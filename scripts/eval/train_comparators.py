#!/usr/bin/env python3
"""Train naive_bbpe, rustbpe, HuggingFace and sentencepiece on the SAME corpus.

WHY THIS EXISTS
---------------
Every test in this project so far is internal: hand-written inputs checked against our
own expectations. A systematic bug -- in `PairIndex`, in the tie-break, in the
pre-tokenizer -- would pass all of them, and would also pass a `.model` self-diff,
because both sides of that diff come out of the same code.

The only way to find that class of bug is to train an independent implementation on the
same data and compare. This script produces the four models; `bench.py` measures them.

WHAT "THE SAME" MEANS, AND WHERE IT BREAKS DOWN
-----------------------------------------------
Comparing tokenizers trained on different data, or with different pre-tokenizers, is
meaningless -- it measures the difference in setup, not the difference in algorithm. So
this script holds three things fixed: the corpus file, the vocabulary size, and the
pre-tokenizer regex. The regex is read out of *our own* `.model` file rather than
written down twice, so the two cannot drift apart.

That works for rustbpe and HuggingFace. It does NOT work for sentencepiece:

    sentencepiece has no regex pre-tokenizer at all. It offers flags
    (split_by_whitespace, split_by_unicode_script, split_by_number) and nothing that
    accepts a pattern. There is no way to give it GPT4_SPLIT_PATTERN.

So sentencepiece is a genuinely different comparison, and the report has to say so: it
is compared on whole-system output only, never on merge lists. It earns its place as a
third independent implementation -- it catches things the other two and we might all get
wrong the same way -- not as a like-for-like control.

VOCABULARY SIZE MEANS FOUR SLIGHTLY DIFFERENT THINGS
-----------------------------------------------------
    ours      total, i.e. 256 byte tokens + (vocab_size - 256) merges
    rustbpe   the same
    HF        "including all tokens and alphabet" -- matches ours only once
              initial_alphabet is forced to all 256 bytes, which is what
              ByteLevel.alphabet() provides. Without it the alphabet is only the
              characters that happen to appear in the corpus.
    SP        total including <unk>. bos/eos/pad are disabled so they cost nothing,
              but <unk> still occupies one slot, so sentencepiece ends up with
              exactly ONE fewer learned piece. At vocab 2000 that is 0.05%.
              It is reported rather than hidden.

ROUND TRIP GATES EVERYTHING
---------------------------
A tokenizer that quietly drops information gets better compression for free: less text
to encode means fewer tokens, which looks like a win. sentencepiece does exactly this by
default -- `remove_extra_whitespaces=True` deletes code indentation -- so a model that
fails `decode(encode(text)) == text` is not written at all. Every trainer here is gated
on it.

USAGE
-----
    python3 scripts/eval/train_comparators.py \\
        --corpus data/book.txt --vocab-size 2000 --out models/eval
"""

import argparse
import base64
import io
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def read_pattern(model_path: Path) -> str:
    """Pull the pre-tokenizer regex out of a naive_bbpe `.model` file.

    Line 2 of the format, verbatim. Reading it rather than hard-coding it here is the
    point: if the pattern ever changes in `lib.rs`, every comparator follows
    automatically instead of silently comparing against a stale copy.
    """
    with model_path.open("rb") as fh:
        lines = fh.read().decode("utf-8").split("\n")
    if lines[0] != "naive_bbpe v1":
        raise SystemExit(f"{model_path} is not a naive_bbpe v1 model")
    return lines[1]


def corpus_chunks(text: str) -> list[str]:
    """The corpus as the trainers want it, losing nothing.

    `keepends=True` matters more than it looks. `text.split("\\n")` throws the newlines
    away, so the comparators would train on a corpus with every line ending deleted
    while our binary trains on the file as it actually is. That is a different corpus,
    and it would show up as a fertility difference that has nothing to do with the
    algorithm.
    """
    return text.splitlines(keepends=True)


def write_ranks(path: Path, ranks: list[tuple[bytes, int]]) -> None:
    """Write `<base64 of token bytes> <rank>` per line, ordered by rank.

    This is tiktoken's own on-disk format, chosen for one reason: a token is an
    arbitrary byte string and need not be valid UTF-8 -- a merge can join half of a
    multi-byte character. That is the same fact that makes our vocab a `Vec<Vec<u8>>`
    rather than a `Vec<String>`. Base64 survives it; text does not.

    rustbpe has no save method of its own (its whole API is train/encode/decode/
    get_mergeable_ranks/get_pattern/vocab_size), so without this its training would be
    thrown away the moment the process exits.
    """
    with path.open("w", encoding="ascii") as fh:
        for token, rank in sorted(ranks, key=lambda kv: kv[1]):
            fh.write(f"{base64.b64encode(token).decode('ascii')} {rank}\n")


def byte_decoder() -> dict[str, int]:
    """The GPT-2 byte <-> unicode table, inverted.

    HuggingFace's ByteLevel pre-tokenizer does not hand BPE raw bytes. It maps each of
    the 256 byte values onto a single printable character first, so that the rest of the
    pipeline only ever sees valid text. The consequence is that HF's vocabulary keys are
    those stand-in characters, not the bytes they represent, and comparing them against
    ours means undoing the mapping.

    The table is defined by GPT-2 and reproduced here because `tokenizers` does not
    expose the inverse.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapped = printable[:]
    spare = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + spare)
            spare += 1
    return {chr(code): byte for byte, code in zip(printable, mapped)}


def check_round_trip(name: str, original: str, decoded: str) -> None:
    """Refuse to accept a tokenizer that cannot reproduce its input."""
    if decoded == original:
        return
    detail = f"lengths {len(original)} in, {len(decoded)} out"
    for i, (a, b) in enumerate(zip(original, decoded)):
        if a != b:
            detail = f"first difference at character {i}: {original[i-30:i+30]!r} became {decoded[i-30:i+30]!r}"
            break
    raise SystemExit(f"{name}: ROUND TRIP FAILED -- refusing to write the model.\n  {detail}")


# ---------------------------------------------------------------------------
# The four trainers
# ---------------------------------------------------------------------------


def train_ours(binary: Path, corpus: Path, vocab_size: int, prefix: Path) -> Path:
    """Shell out to the real CLI.

    Deliberately a subprocess rather than a binding: this measures the binary that
    actually ships, driven exactly the way a person would drive it, rather than some
    private path that only the benchmark uses.
    """
    result = subprocess.run(
        [str(binary), "train", "--input", str(corpus),
         "--vocab-size", str(vocab_size), "--out", str(prefix)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ours: `naive_bbpe train` failed ({result.returncode})\n{result.stderr}")

    model = Path(f"{prefix}.model")
    # Round trip through the CLI too -- encode and decode are what a user touches.
    encoded = subprocess.run([str(binary), "encode", "--model", str(model)],
                             stdin=corpus.open("rb"), capture_output=True)
    decoded = subprocess.run([str(binary), "decode", "--model", str(model)],
                             input=encoded.stdout, capture_output=True)
    check_round_trip("ours", corpus.read_text(encoding="utf-8"),
                     decoded.stdout.decode("utf-8"))
    return model


def train_rustbpe(text: str, vocab_size: int, pattern: str, prefix: Path) -> int:
    import rustbpe

    tok = rustbpe.Tokenizer()
    tok.train_from_iterator(corpus_chunks(text), vocab_size, pattern=pattern)

    # Confirm the pre-tokenizer really is ours before anything is compared. Merge lists
    # from different pre-tokenizers are not comparable, and a silently-defaulted pattern
    # would make every downstream number wrong in a way no metric would reveal.
    if tok.get_pattern() != pattern:
        raise SystemExit("rustbpe: pattern was not applied; comparison would be meaningless")

    check_round_trip("rustbpe", text, tok.decode(tok.encode(text)))
    write_ranks(Path(f"{prefix}.ranks"), tok.get_mergeable_ranks())
    return tok.vocab_size


def train_hf(text: str, vocab_size: int, pattern: str, prefix: Path) -> int:
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    # Our regex first, then ByteLevel. The order matters: Split cuts the text into the
    # same chunks our pre-tokenizer produces, and ByteLevel then turns each chunk's
    # bytes into stand-in characters. `use_regex=False` stops ByteLevel applying its own
    # built-in GPT-2 split on top of ours, which would chunk the text twice.
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()

    tok.train_from_iterator(corpus_chunks(text), trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=False,
        special_tokens=[],
        # Force all 256 bytes into the alphabet. Without this the alphabet is only the
        # characters present in the corpus, and "vocab_size" would not mean what it
        # means for the other three.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    ))

    check_round_trip("hf", text, tok.decode(tok.encode(text).ids))
    tok.save(f"{prefix}.json")

    # Also emit the common ranks format, with the ByteLevel mapping undone, so the
    # merge-list comparison in 11.5 is a direct diff rather than a translation exercise.
    inverse = byte_decoder()
    write_ranks(Path(f"{prefix}.ranks"),
                [(bytes(inverse[ch] for ch in piece), rank)
                 for piece, rank in tok.get_vocab().items()])
    return tok.get_vocab_size()


def train_sp(text: str, vocab_size: int, prefix: Path) -> int:
    import sentencepiece as spm

    buffer = io.BytesIO()
    spm.SentencePieceTrainer.Train(
        sentence_iterator=iter(corpus_chunks(text)),
        model_writer=buffer,
        model_type="bpe",
        vocab_size=vocab_size,
        # --- the three documented traps, all off ---
        add_dummy_prefix=False,          # else a marker is glued to every input: +1 token each
        remove_extra_whitespaces=False,  # else runs of spaces collapse and indentation is DELETED
        normalization_rule_name="identity",  # the default is nmt_nfkc, not plain NFKC
        # --- and four more that the notes did not have ---
        byte_fallback=True,              # unseen characters become bytes, not <unk>
        character_coverage=1.0,          # keep every character, not 99.95% of them
        allow_whitespace_only_pieces=True,  # WITHOUT THIS a run of spaces cannot be
                                            # represented and round trip fails on code
        hard_vocab_limit=False,          # stop short rather than erroring, as ours does
        # Unused specials would each cost a vocabulary slot. <unk> cannot be disabled.
        unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1,
        minloglevel=2,
    )

    sp = spm.SentencePieceProcessor(model_proto=buffer.getvalue())
    check_round_trip("sentencepiece", text, sp.decode(sp.encode(text)))
    Path(f"{prefix}.model").write_bytes(buffer.getvalue())
    return sp.get_piece_size()


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("models/eval"))
    ap.add_argument("--binary", type=Path, default=Path("target/release/naive_bbpe"))
    ap.add_argument("--only", nargs="*", choices=["ours", "rustbpe", "hf", "sp"],
                    help="train only these; default is all four")
    args = ap.parse_args()

    if not args.binary.exists():
        raise SystemExit(f"{args.binary} not found -- run `cargo build --release` first")

    # `naive_bbpe train --out` fails if the directory does not exist, so create it here.
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.corpus.stem
    wanted = args.only or ["ours", "rustbpe", "hf", "sp"]

    def prefix(name: str) -> Path:
        return args.out / f"{name}-{stem}-{args.vocab_size}"

    print(f"corpus     {args.corpus}  ({args.corpus.stat().st_size:,} bytes)")
    print(f"vocab size {args.vocab_size}")
    print(f"output     {args.out}/\n")

    # Ours goes first: it produces the `.model` the pattern is read from, so every other
    # trainer is configured from our own file rather than a second copy of the regex.
    sizes: dict[str, int] = {}
    if "ours" in wanted:
        started = time.time()
        model = train_ours(args.binary, args.corpus, args.vocab_size, prefix("ours"))
        print(f"ours           ok  {time.time() - started:6.1f}s  -> {model.name}")
    else:
        model = Path(f"{prefix('ours')}.model")
        if not model.exists():
            raise SystemExit(f"{model} is needed for the pattern; train `ours` first")

    pattern = read_pattern(model)
    print(f"pattern    {pattern[:60]}...\n" if len(pattern) > 60 else f"pattern    {pattern}\n")

    text = args.corpus.read_text(encoding="utf-8")

    for name, run in (
        ("rustbpe", lambda: train_rustbpe(text, args.vocab_size, pattern, prefix("rustbpe"))),
        ("hf", lambda: train_hf(text, args.vocab_size, pattern, prefix("hf"))),
        ("sp", lambda: train_sp(text, args.vocab_size, prefix("sp"))),
    ):
        if name not in wanted:
            continue
        started = time.time()
        sizes[name] = run()
        print(f"{name:<14} ok  {time.time() - started:6.1f}s  "
              f"-> {prefix(name).name}.*   vocab {sizes[name]}")

    print("\nall round trips passed")
    if sizes.get("sp") is not None and sizes["sp"] != args.vocab_size:
        print(f"note: sentencepiece reports {sizes['sp']} pieces "
              f"({args.vocab_size - sizes['sp']:+d} vs the target)")


if __name__ == "__main__":
    sys.exit(main())
