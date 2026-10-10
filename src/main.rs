//! The `naive_bbpe` command-line driver.
//!
//! Until step 6 the training corpus was `TEXT`, a 22 KB string literal compiled into the
//! executable — a *compile-time* input. That became a path read at run time. This step
//! closes the other end: a trained tokenizer used to vanish when the process exited, and
//! now it is written to disk and can be loaded back.
//!
//! There is no magic in "command line arguments". When you type
//!
//! ```text
//! naive_bbpe train --input data.txt --vocab-size 50000 --out fw
//! ```
//!
//! the shell hands the process that list of words, and [`std::env::args`] gives it to us
//! as an iterator of `String`. Everything below is a loop over that list. No dependency
//! does this for us — `clap` would, but then the parsing would be a black box.
//!
//! The tokenizer itself lives in `src/lib.rs`; this file only sees its `pub` surface.

use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use naive_bbpe::{
    AllowedSpecial, GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN, Normalizer, Tokenizer,
};

const USAGE: &str = "\
naive_bbpe — a naive byte-level BPE tokenizer

USAGE:
    naive_bbpe train  --input <FILE> --vocab-size <N> --out <PREFIX> [OPTIONS]
    naive_bbpe encode --model <FILE> [--allow-special <POLICY>]
    naive_bbpe decode --model <FILE>

TRAIN OPTIONS:
    --input <FILE>        UTF-8 text file to train on            (required)
    --vocab-size <N>      total vocabulary size, minimum 256     (required)
    --out <PREFIX>        writes <PREFIX>.model and <PREFIX>.vocab  (required)
    --pattern <NAME>      pre-tokenizer: gpt4 | gpt2             [default: gpt4]
    --normalizer <NAME>   none | nfc | nfd                       [default: none]
    --special <TOK>[=ID]  register a special token; repeatable
                          ID defaults to the next free slot at or above vocab-size
    --verbose             report training progress
    --no-report           skip re-reading the input after training to print
                          compression and round trip (the model is saved either way)
    --threads <N>         how many blocks of input to count at once; the model is
                          identical for any N           [default: all cores, at most 64]

ENCODE OPTIONS:
    --allow-special <P>   all | none | none-raise                [default: none-raise]
                          all        -> <|endoftext|> becomes its own token
                          none       -> it encodes as 13 ordinary characters
                          none-raise -> seeing it is an error

    -h, --help            print this message

NOTES:
    vocab-size is the TOTAL table size, not the merge count: the 256 byte tokens exist
    before training starts, so --vocab-size 2000 buys 1744 merges.

    <PREFIX>.model is the real artifact and is all `load` reads. <PREFIX>.vocab is for
    reading with your own eyes and is never loaded back.

    Token ids are written whitespace-separated; `decode` accepts any whitespace, so
    `encode ... | decode ...` round-trips.

    Special tokens are NOT learned by training. They are reserved ids above the learned
    vocabulary, used to mark structure a model needs to see -- document boundaries, chat
    turns -- that must never be confused with text that happens to look the same.";

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
//
// Deliberately pure functions: arguments in, a value out, no I/O and no side effects.
// That is the entire reason they can be tested. Parsing written inline inside `main()`
// would be unreachable from a test — you would have to spawn a process to exercise it.

#[derive(Debug, PartialEq)]
enum Command {
    Train(TrainArgs),
    Encode { model: PathBuf, allowed: AllowedSpecial },
    Decode { model: PathBuf },
    Help,
}

#[derive(Debug, PartialEq)]
struct TrainArgs {
    input: PathBuf,
    vocab_size: u32,
    /// Prefix, not a filename: `.model` and `.vocab` are appended to it.
    out: PathBuf,
    /// The regex source itself, resolved at parse time so an unknown name fails before
    /// we read a potentially enormous input file.
    pattern: &'static str,
    /// The name it was given, kept only so the summary can print "gpt4" rather than the
    /// whole regex.
    pattern_name: String,
    normalizer: Normalizer,
    /// `(token, explicit id)`. `None` means "assign the next free slot", resolved once
    /// vocab_size is known rather than at parse time.
    specials: Vec<(String, Option<u32>)>,
    verbose: bool,
    /// Re-read the input after training to print compression and round trip. On by
    /// default; `--no-report` turns it off, because on a production corpus that second
    /// pass takes longer than training itself (prod.2).
    report: bool,
    /// How many blocks are counted at the same time (prod.3). Only speed and memory
    /// depend on it; the model does not.
    threads: usize,
}

/// Every value-taking flag needs the same "is there actually a next argument?" check.
/// Without it, `naive_bbpe train --input` would panic on an index out of bounds.
fn value_of(args: &[String], i: usize, flag: &str) -> Result<String, String> {
    args.get(i + 1)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

/// `args` excludes the program name — `main` strips it, tests never supply it.
fn parse_args(args: &[String]) -> Result<Command, String> {
    let Some(subcommand) = args.first() else {
        return Err("no subcommand given".to_string());
    };

    match subcommand.as_str() {
        "-h" | "--help" => Ok(Command::Help),
        "train" => parse_train(&args[1..]),
        "encode" => parse_encode(&args[1..]),
        "decode" => Ok(match parse_model_flag(&args[1..], "decode")? {
            Some(model) => Command::Decode { model },
            None => Command::Help,
        }),
        other => Err(format!("unknown subcommand: {other:?}")),
    }
}

/// `encode` and `decode` take exactly one flag. `None` means `--help` was asked for.
fn parse_model_flag(args: &[String], subcommand: &str) -> Result<Option<PathBuf>, String> {
    let mut model: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(value_of(args, i, "--model")?));
                i += 2;
            }
            "-h" | "--help" => return Ok(None),
            other => return Err(format!("unknown flag for {subcommand}: {other:?}")),
        }
    }
    Ok(Some(model.ok_or("--model is required")?))
}

/// `encode` takes a model and a specials policy.
fn parse_encode(args: &[String]) -> Result<Command, String> {
    let mut model: Option<PathBuf> = None;
    // minbpe's default. The only policy that cannot silently do the wrong thing: it neither
    // invents a boundary in someone's prose nor swallows one that was meant.
    let mut allowed = AllowedSpecial::NoneRaise;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(value_of(args, i, "--model")?));
                i += 2;
            }
            "--allow-special" => {
                let raw = value_of(args, i, "--allow-special")?;
                allowed = match raw.as_str() {
                    "all" => AllowedSpecial::All,
                    "none" => AllowedSpecial::None,
                    "none-raise" => AllowedSpecial::NoneRaise,
                    other => {
                        return Err(format!(
                            "unknown --allow-special policy: {other:?} \
                             (expected all, none or none-raise)"
                        ));
                    }
                };
                i += 2;
            }
            "-h" | "--help" => return Ok(Command::Help),
            other => return Err(format!("unknown flag for encode: {other:?}")),
        }
    }
    Ok(Command::Encode {
        model: model.ok_or("--model is required")?,
        allowed,
    })
}

/// Parse one `--special` value: `<TOKEN>` or `<TOKEN>=<ID>`.
///
/// The rule is deliberately narrow: split on the **last** `=`, and treat the suffix as an id
/// **only if it parses as a number**. Otherwise the whole argument is the token.
///
/// That is what makes `--special "<|a=b|>"` work — a special token is an arbitrary string
/// and may perfectly well contain `=`. The price is that `--special "<|x|>=oops"` registers
/// a token literally named `<|x|>=oops` instead of complaining about a malformed id. Given
/// that `--special` echoes every registered token and its id back to you at the end of
/// training, a typo is visible immediately, which is the better trade.
fn parse_special(raw: &str) -> Result<(String, Option<u32>), String> {
    if raw.is_empty() {
        return Err("--special needs a token".to_string());
    }
    match raw.rsplit_once('=') {
        Some((token, id)) if !token.is_empty() => match id.parse::<u32>() {
            Ok(id) => Ok((token.to_string(), Some(id))),
            Err(_) => Ok((raw.to_string(), None)),
        },
        _ => Ok((raw.to_string(), None)),
    }
}

fn parse_train(args: &[String]) -> Result<Command, String> {
    let mut input: Option<PathBuf> = None;
    let mut vocab_size: Option<u32> = None;
    let mut out: Option<PathBuf> = None;
    let mut pattern_name = "gpt4".to_string();
    let mut normalizer = Normalizer::None;
    let mut specials: Vec<(String, Option<u32>)> = Vec::new();
    let mut verbose = false;
    let mut report = true;
    let mut threads = default_threads();

    // Flags that take a value need the NEXT element too, so this is a while loop with a
    // manual index rather than a `for`: some arms advance by 2 and some by 1.
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                input = Some(PathBuf::from(value_of(args, i, "--input")?));
                i += 2;
            }
            "--vocab-size" => {
                let raw = value_of(args, i, "--vocab-size")?;
                vocab_size = Some(
                    raw.parse::<u32>()
                        .map_err(|_| format!("--vocab-size expects a number, got {raw:?}"))?,
                );
                i += 2;
            }
            "--out" => {
                out = Some(PathBuf::from(value_of(args, i, "--out")?));
                i += 2;
            }
            "--pattern" => {
                pattern_name = value_of(args, i, "--pattern")?;
                i += 2;
            }
            "--normalizer" => {
                let raw = value_of(args, i, "--normalizer")?;
                // The library owns this spelling, because the same strings appear in the
                // `.model` file. One definition, two callers.
                normalizer = Normalizer::from_name(&raw)
                    .ok_or_else(|| format!("unknown normalizer: {raw:?}"))?;
                i += 2;
            }
            "--special" => {
                specials.push(parse_special(&value_of(args, i, "--special")?)?);
                i += 2;
            }
            "--verbose" => {
                verbose = true;
                i += 1;
            }
            "--no-report" => {
                report = false;
                i += 1;
            }
            "--threads" => {
                let raw = value_of(args, i, "--threads")?;
                threads = match raw.parse::<usize>() {
                    Ok(n) if n >= 1 => n,
                    _ => return Err(format!("--threads expects a number of at least 1, got {raw:?}")),
                };
                i += 2;
            }
            "-h" | "--help" => return Ok(Command::Help),
            other => return Err(format!("unknown flag: {other:?}")),
        }
    }

    // Resolve the pattern name here, not in `run_train`: a typo should fail immediately,
    // not after reading a gigabyte off disk.
    let pattern = match pattern_name.as_str() {
        "gpt4" => GPT4_SPLIT_PATTERN,
        "gpt2" => GPT2_SPLIT_PATTERN,
        other => return Err(format!("unknown pattern: {other:?} (expected gpt4 or gpt2)")),
    };

    let input = input.ok_or("--input is required")?;
    let vocab_size = vocab_size.ok_or("--vocab-size is required")?;
    let out = out.ok_or("--out is required")?;

    // `train` asserts this too, but a clear message beforehand beats a panic afterwards.
    if vocab_size < 256 {
        return Err(format!(
            "--vocab-size must be at least 256 (the byte tokens), got {vocab_size}"
        ));
    }

    Ok(Command::Train(TrainArgs {
        input,
        vocab_size,
        out,
        pattern,
        pattern_name,
        normalizer,
        specials,
        verbose,
        report,
        threads,
    }))
}

/// `--threads` when it is not given: every core this machine reports, but at most 64.
///
/// Measured at prod.3 on 1 GB: ~97% of counting runs in parallel and ~3% (reading the file,
/// combining the threads' tallies) does not, so past ~64 threads the time barely falls while
/// memory — a block or two in flight per thread — keeps rising. A DEFAULT only: an explicit
/// `--threads 512` is used as given.
fn default_threads() -> usize {
    const DEFAULT_MAX_THREADS: usize = 64;
    std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(DEFAULT_MAX_THREADS)
}

// ---------------------------------------------------------------------------
// Doing the work
// ---------------------------------------------------------------------------

/// `Box<dyn Error>` is a trait object: "some value that implements the `Error` trait,
/// heap-allocated because different errors are different sizes". It lets `?` propagate an
/// `io::Error` and a `String` from the same function without a custom error type.
fn run_train(args: TrainArgs) -> Result<(), Box<dyn Error>> {
    // Check we can WRITE before doing hours of work. Until now `save` ran after training,
    // so a typo in --out meant a full training run was completed and then thrown away with
    // "No such file or directory". At 22 KB that costs a third of a second; at 50k vocab on
    // a real corpus it costs the whole run.
    check_output_writable(&args.out)?;

    // The input is READ IN BLOCKS, never whole (prod.1): a production corpus is far larger
    // than memory. Opening it here, before anything is printed, still makes a missing file
    // the first thing reported. UTF-8 is checked block by block as training reads it.
    let read_error = |e: io::Error| format!("could not read {}: {e}", args.input.display());
    let input = fs::File::open(&args.input).map_err(read_error)?;
    // Bytes only — the size on disk is free, while counting characters (as this line did
    // until prod.1) would mean one extra pass over the whole corpus before training starts.
    let input_bytes = input.metadata().map_err(read_error)?.len();

    println!("input:      {}", args.input.display());
    println!("            {input_bytes} bytes");
    println!("vocab size: {}  ({} merges to learn)", args.vocab_size, args.vocab_size - 256);
    println!("pattern:    {}", args.pattern_name);
    println!("normalizer: {}", args.normalizer.name());
    println!("threads:    {}", args.threads);
    println!();

    let mut tok = Tokenizer::with_config(args.pattern, args.normalizer);
    let started = Instant::now();
    tok.train_from_reader(input, args.vocab_size, args.threads, args.verbose)
        .map_err(read_error)?;
    let elapsed = started.elapsed();

    // Specials are registered AFTER training: their ids must sit above the learned
    // vocabulary, and how large that is depends on whether training ran out of pairs early.
    if !args.specials.is_empty() {
        let resolved = resolve_special_ids(&args.specials, tok.vocab_size() as u32)?;
        tok.register_special_tokens(&resolved)?;
        println!();
        for (token, id) in tok.special_tokens() {
            println!("special:    {token} -> {id}");
        }
    }

    tok.save(&args.out)
        .map_err(|e| format!("could not write {}.model: {e}", args.out.display()))?;

    // Re-encoding the corpus reproduces exactly what training ended with, because encode
    // applies merges in id order and ids were handed out in merge order.
    //
    // Read in blocks again, a second time from the top. Encoding block by block gives the
    // same ids as encoding the whole, because blocks are cut only where the pre-tokenizer
    // cuts; so the token count adds up, and the text round-trips iff every block does.
    //
    // This is a full encode pass over the input, single-threaded: measured at prod.2 as
    // ~2 minutes per GB, against ~46 s to train on the same GB at vocab 300. On a
    // production corpus it is days, hence `--no-report`. The model is already saved above,
    // so skipping this loses nothing training produced.
    let report = if args.report {
        let input = fs::File::open(&args.input).map_err(read_error)?;
        let (mut bytes, mut tokens, mut round_trips) = (0usize, 0usize, true);
        tok.for_each_block(input, |block| {
            let ids = tok.encode_ordinary(block);
            bytes += block.len();
            tokens += ids.len();
            round_trips &= tok.decode(&ids).map_err(io::Error::other)? == block;
            Ok(())
        })
        .map_err(read_error)?;
        Some((bytes, tokens, round_trips))
    } else {
        None
    };
    println!();
    println!("trained in:  {:.2}s", elapsed.as_secs_f64());
    if let Some((bytes, tokens, round_trips)) = report {
        println!(
            "compression: {} bytes -> {} tokens ({:.2}x)",
            bytes,
            tokens,
            bytes as f64 / tokens as f64
        );
        // Under nfc/nfd this is expected to be false: normalization rewrites the text
        // before encode ever sees it, so the original bytes are gone. Reported, never
        // asserted.
        println!(
            "round trip:  {}",
            if round_trips { "ok" } else { "lossy (normalizer is not None)" }
        );
    }
    println!("wrote:       {}.model, {}.vocab", args.out.display(), args.out.display());

    Ok(())
}

/// Fail early if `<prefix>.model` cannot be created. Writing and removing a probe file is
/// the only way to actually know — checking that the parent directory exists would miss
/// permissions, a read-only mount, or a full disk.
fn check_output_writable(prefix: &Path) -> Result<(), Box<dyn Error>> {
    let mut probe = prefix.as_os_str().to_owned();
    probe.push(".model");
    let probe = PathBuf::from(probe);

    // Do not clobber an existing model just to test writability.
    if probe.exists() {
        return Ok(());
    }
    fs::File::create(&probe)
        .map_err(|e| format!("cannot write {}: {e}", probe.display()))?;
    fs::remove_file(&probe).ok();
    Ok(())
}

/// Turn `--special` values into concrete ids. An explicit `=ID` is taken as given; the rest
/// are handed the next free slot at or above the learned vocabulary, in the order they were
/// listed on the command line.
fn resolve_special_ids(
    specials: &[(String, Option<u32>)],
    vocab_size: u32,
) -> Result<Vec<(String, u32)>, String> {
    let mut next = vocab_size;
    let taken: Vec<u32> = specials.iter().filter_map(|(_, id)| *id).collect();
    let mut out = Vec::with_capacity(specials.len());

    for (token, explicit) in specials {
        let id = match explicit {
            Some(id) => *id,
            None => {
                while taken.contains(&next) {
                    next += 1;
                }
                let id = next;
                next += 1;
                id
            }
        };
        out.push((token.clone(), id));
    }
    Ok(out)
}

fn load_model(path: &Path) -> Result<Tokenizer, Box<dyn Error>> {
    Tokenizer::load(path).map_err(|e| format!("could not load {}: {e}", path.display()).into())
}

fn read_stdin() -> Result<String, Box<dyn Error>> {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

fn run_encode(model: PathBuf, allowed: AllowedSpecial) -> Result<(), Box<dyn Error>> {
    let tok = load_model(&model)?;
    let ids = tok.encode(&read_stdin()?, allowed)?;

    // Whitespace-separated on one line. Compact, and `decode` splits on any whitespace so
    // the exact separator never has to be agreed on.
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            write!(out, " ")?;
        }
        write!(out, "{id}")?;
    }
    writeln!(out)?;
    Ok(())
}

fn run_decode(model: PathBuf) -> Result<(), Box<dyn Error>> {
    let tok = load_model(&model)?;
    let input = read_stdin()?;

    let mut ids = Vec::new();
    for field in input.split_whitespace() {
        ids.push(
            field
                .parse::<u32>()
                .map_err(|_| format!("not a token id: {field:?}"))?,
        );
    }

    // The range check that used to live here moved into `decode` in step 8 — it now has to
    // consider special-token ids, which sit above the vocabulary, so only the library can
    // answer "is this id valid?".
    io::stdout().write_all(tok.decode(&ids)?.as_bytes())?;
    Ok(())
}

fn main() {
    // `skip(1)` drops the program's own name, which is always args[0].
    let args: Vec<String> = env::args().skip(1).collect();

    let command = match parse_args(&args) {
        Ok(c) => c,
        Err(message) => {
            // Exit 2 for "you typed it wrong", the long-standing Unix convention, as
            // distinct from 1 for "it ran and failed".
            eprintln!("error: {message}\n\n{USAGE}");
            process::exit(2);
        }
    };

    let result = match command {
        Command::Help => {
            println!("{USAGE}");
            return;
        }
        Command::Train(args) => run_train(args),
        Command::Encode { model, allowed } => run_encode(model, allowed),
        Command::Decode { model } => run_decode(model),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// These are the binary crate's own unit tests — a third test target, alongside the
// library's unit tests and `tests/api.rs`. They exist because `parse_args` is a pure
// function: no process to spawn, no files to create, just a Vec<String> in and a value
// out.

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn train_args(v: &[&str]) -> TrainArgs {
        match parse_args(&args(v)).expect("expected these arguments to parse") {
            Command::Train(a) => a,
            other => panic!("expected a train command, got {other:?}"),
        }
    }

    #[test]
    fn parses_a_full_train_command() {
        let a = train_args(&[
            "train",
            "--input",
            "corpus.txt",
            "--vocab-size",
            "50000",
            "--out",
            "models/fw",
            "--pattern",
            "gpt2",
            "--normalizer",
            "nfc",
            "--verbose",
        ]);
        assert_eq!(a.input, PathBuf::from("corpus.txt"));
        assert_eq!(a.vocab_size, 50000);
        assert_eq!(a.out, PathBuf::from("models/fw"));
        assert_eq!(a.pattern, GPT2_SPLIT_PATTERN);
        assert_eq!(a.normalizer, Normalizer::Nfc);
        assert!(a.verbose);
    }

    #[test]
    fn the_defaults_are_gpt4_none_and_quiet() {
        let a = train_args(&["train", "--input", "c.txt", "--vocab-size", "300", "--out", "m"]);
        assert_eq!(a.pattern, GPT4_SPLIT_PATTERN);
        assert_eq!(a.normalizer, Normalizer::None);
        assert!(!a.verbose);
        assert!(a.report, "the report is on unless --no-report says otherwise");
        assert_eq!(a.threads, default_threads());
        assert!((1..=64).contains(&a.threads), "the default is all cores, at most 64");
    }

    #[test]
    fn threads_is_taken_as_given_even_above_the_default_cap() {
        let a = train_args(&[
            "train", "--input", "c.txt", "--vocab-size", "300", "--out", "m", "--threads", "512",
        ]);
        assert_eq!(a.threads, 512);
        for bad in ["0", "-1", "many"] {
            let v = ["train", "--input", "c.txt", "--vocab-size", "300", "--out", "m", "--threads", bad];
            assert!(parse_args(&args(&v)).is_err(), "--threads {bad} was accepted");
        }
    }

    #[test]
    fn no_report_turns_the_report_off() {
        let a = train_args(&[
            "train", "--input", "c.txt", "--vocab-size", "300", "--out", "m", "--no-report",
        ]);
        assert!(!a.report);
        assert!(!a.verbose, "--no-report is its own flag, not a quieter --verbose");
    }

    #[test]
    fn flags_may_appear_in_any_order() {
        let a = train_args(&[
            "train", "--verbose", "--out", "m", "--vocab-size", "300", "--input", "c.txt",
        ]);
        assert_eq!(a.vocab_size, 300);
        assert!(a.verbose);
    }

    #[test]
    fn every_required_train_flag_is_actually_required() {
        for (missing, rest) in [
            ("--input", vec!["train", "--vocab-size", "300", "--out", "m"]),
            ("--vocab-size", vec!["train", "--input", "c.txt", "--out", "m"]),
            ("--out", vec!["train", "--input", "c.txt", "--vocab-size", "300"]),
        ] {
            let e = parse_args(&args(&rest)).unwrap_err();
            assert!(e.contains(missing), "expected {missing} in message, got: {e}");
        }
    }

    #[test]
    fn vocab_size_must_be_a_number() {
        let e = parse_args(&args(&[
            "train", "--input", "c.txt", "--out", "m", "--vocab-size", "abc",
        ]))
        .unwrap_err();
        assert!(e.contains("number"), "unhelpful message: {e}");
    }

    #[test]
    fn vocab_size_below_256_is_rejected_before_any_file_is_read() {
        // `train` would panic on this. Catching it during parsing means the error arrives
        // before we spend time reading the input.
        let e = parse_args(&args(&[
            "train", "--input", "c.txt", "--out", "m", "--vocab-size", "100",
        ]))
        .unwrap_err();
        assert!(e.contains("256"), "unhelpful message: {e}");
    }

    #[test]
    fn a_flag_with_no_value_errors_instead_of_panicking() {
        // The trap this guards: `args[i + 1]` on the last argument is an index out of
        // bounds. A CLI must never panic on input a user could plausibly type.
        for flags in [
            vec!["train", "--input"],
            vec!["train", "--vocab-size"],
            vec!["train", "--out"],
            vec!["train", "--pattern"],
            vec!["train", "--normalizer"],
            vec!["encode", "--model"],
            vec!["decode", "--model"],
        ] {
            let e = parse_args(&args(&flags)).unwrap_err();
            assert!(e.contains("needs a value"), "unhelpful message for {flags:?}: {e}");
        }
    }

    #[test]
    fn unknown_flags_and_subcommands_are_rejected() {
        assert!(parse_args(&args(&["train", "--turbo"])).unwrap_err().contains("unknown flag"));
        assert!(parse_args(&args(&["encode", "--turbo"])).unwrap_err().contains("unknown flag"));
        assert!(parse_args(&args(&["frobnicate"])).unwrap_err().contains("unknown subcommand"));
        assert!(parse_args(&args(&[])).unwrap_err().contains("no subcommand"));
    }

    #[test]
    fn unknown_pattern_and_normalizer_names_are_rejected() {
        let base = ["train", "--input", "c.txt", "--vocab-size", "300", "--out", "m"];

        let mut a = base.to_vec();
        a.extend(["--pattern", "gpt9"]);
        assert!(parse_args(&args(&a)).unwrap_err().contains("unknown pattern"));

        let mut a = base.to_vec();
        a.extend(["--normalizer", "nfkc"]);
        assert!(parse_args(&args(&a)).unwrap_err().contains("unknown normalizer"));
    }

    #[test]
    fn parses_encode_and_decode() {
        assert_eq!(
            parse_args(&args(&["encode", "--model", "fw.model"])).unwrap(),
            // The default policy is the cautious one: seeing a special token is an error
            // rather than a silent guess in either direction.
            Command::Encode {
                model: PathBuf::from("fw.model"),
                allowed: AllowedSpecial::NoneRaise
            }
        );
        assert_eq!(
            parse_args(&args(&["decode", "--model", "fw.model"])).unwrap(),
            Command::Decode { model: PathBuf::from("fw.model") }
        );
        assert!(parse_args(&args(&["encode"])).unwrap_err().contains("--model is required"));
    }

    #[test]
    fn help_is_recognised_everywhere() {
        for a in [
            vec!["--help"],
            vec!["-h"],
            vec!["train", "--help"],
            vec!["encode", "--help"],
            vec!["decode", "-h"],
        ] {
            assert_eq!(parse_args(&args(&a)).unwrap(), Command::Help, "for {a:?}");
        }
    }

    // -- reading the input file -------------------------------------------------

    // -- special tokens ---------------------------------------------------------

    #[test]
    fn parses_special_tokens_with_and_without_explicit_ids() {
        let a = train_args(&[
            "train", "--input", "c.txt", "--vocab-size", "300", "--out", "m",
            "--special", "<|endoftext|>",
            "--special", "<|fim_prefix|>=70000",
        ]);
        assert_eq!(
            a.specials,
            vec![
                ("<|endoftext|>".to_string(), None),
                ("<|fim_prefix|>".to_string(), Some(70000)),
            ]
        );
    }

    #[test]
    fn a_special_containing_an_equals_sign_splits_on_the_last_one() {
        // A token containing `=` is fine: the suffix is only an id if it is a number.
        assert_eq!(parse_special("<|a=b|>").unwrap(), ("<|a=b|>".to_string(), None));
        assert_eq!(parse_special("<|a=b|>=99").unwrap(), ("<|a=b|>".to_string(), Some(99)));
        assert_eq!(parse_special("<|endoftext|>").unwrap(), ("<|endoftext|>".to_string(), None));
        // The documented consequence: a non-numeric suffix is part of the token, not a
        // malformed id. Training echoes every token and id back, so this is visible.
        assert_eq!(
            parse_special("<|x|>=oops").unwrap(),
            ("<|x|>=oops".to_string(), None)
        );
        assert!(parse_special("").unwrap_err().contains("needs a token"));
    }

    #[test]
    fn unassigned_special_ids_start_at_the_vocabulary_and_skip_taken_ones() {
        let specials = vec![
            ("<|a|>".to_string(), None),
            ("<|b|>".to_string(), Some(2000)),
            ("<|c|>".to_string(), None),
        ];
        // vocab_size 2000, and 2000 is explicitly claimed, so the automatic ones step over it.
        assert_eq!(
            resolve_special_ids(&specials, 2000).unwrap(),
            vec![
                ("<|a|>".to_string(), 2001),
                ("<|b|>".to_string(), 2000),
                ("<|c|>".to_string(), 2002),
            ]
        );
    }

    #[test]
    fn encode_takes_an_allow_special_policy() {
        for (flag, expected) in [
            ("all", AllowedSpecial::All),
            ("none", AllowedSpecial::None),
            ("none-raise", AllowedSpecial::NoneRaise),
        ] {
            assert_eq!(
                parse_args(&args(&["encode", "--model", "m", "--allow-special", flag])).unwrap(),
                Command::Encode { model: PathBuf::from("m"), allowed: expected }
            );
        }
        assert!(parse_args(&args(&["encode", "--model", "m", "--allow-special", "maybe"]))
            .unwrap_err()
            .contains("unknown --allow-special"));
        // `decode` has no such flag: ids are unambiguous, only text is.
        assert!(parse_args(&args(&["decode", "--model", "m", "--allow-special", "all"]))
            .unwrap_err()
            .contains("unknown flag"));
    }

    #[test]
    fn an_unwritable_output_prefix_fails_before_training_starts() {
        // The wart this fixes: `save` ran AFTER training, so a typo in --out meant a full
        // run finished and was then thrown away. At 50k vocab that is hours.
        let e = check_output_writable(Path::new("no/such/dir/model")).unwrap_err();
        assert!(e.to_string().contains("cannot write"), "unhelpful message: {e}");

        // A usable prefix passes and leaves nothing behind.
        let ok = std::env::temp_dir().join("naive_bbpe_writable_probe");
        check_output_writable(&ok).unwrap();
        assert!(!ok.with_extension("model").exists(), "the probe file was not cleaned up");
    }

    #[test]
    fn the_corpus_fixture_reads_back_at_its_known_size() {
        // The fixture is `TEXT` as it was before step 6, plus one trailing newline.
        let text = fs::read_to_string("tests/data/unicode_article.txt").unwrap();
        assert_eq!(text.len(), 22815);
        assert!(text.ends_with("code points\n"));
    }

    #[test]
    fn a_missing_input_file_is_an_error_not_a_panic() {
        let args = train_args(&[
            "train", "--input", "no/such/file.txt", "--vocab-size", "300", "--out", "m",
        ]);
        let e = run_train(args).unwrap_err();
        assert!(e.to_string().contains("could not read"), "unhelpful message: {e}");
    }

    #[test]
    fn a_non_utf8_input_file_is_an_error_not_a_panic() {
        // 0xFF can never appear in valid UTF-8. `read_to_string` rejects the whole file.
        let path = std::env::temp_dir().join("naive_bbpe_invalid_utf8.bin");
        fs::write(&path, [0xFFu8, 0xFE, 0xFD]).unwrap();
        let args = train_args(&[
            "train",
            "--input",
            path.to_str().unwrap(),
            "--vocab-size",
            "300",
            "--out",
            "m",
        ]);
        let e = run_train(args).unwrap_err();
        assert!(e.to_string().contains("could not read"), "unhelpful message: {e}");
        fs::remove_file(&path).ok();
    }

    #[test]
    fn a_missing_model_file_is_an_error_not_a_panic() {
        let e = load_model(&PathBuf::from("no/such/model.model")).unwrap_err();
        assert!(e.to_string().contains("could not load"), "unhelpful message: {e}");
    }
}
