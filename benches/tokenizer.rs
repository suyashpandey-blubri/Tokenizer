//! Regression tripwires, not a performance investigation.
//!
//! WHY THESE EXIST NOW, AT THE END OF STEP 11
//! -------------------------------------------
//! Step 12 is a whole-codebase refactor: `Tokenizer` splits into `BpeModel` and
//! `BpeTrainer`, and a boundary mask gets threaded through the merge path. The project's
//! equivalence gate proves the refactor produces a byte-identical `.model` — and says
//! nothing at all about speed. A vtable hop landing in a hot loop, or one extra
//! allocation per merge, sails straight past a byte-for-byte output check.
//!
//! So this is the baseline step 12 gets measured against, not housekeeping.
//!
//! WHAT IS DELIBERATELY NOT HERE
//! ------------------------------
//! No 50k training run. A benchmark that takes minutes per sample is a benchmark nobody
//! runs, and one nobody runs catches nothing. No `data/` either: it is gitignored and may
//! not exist on another machine. Everything here uses `tests/data/unicode_article.txt`,
//! which is checked in and `include_str!`d, so `cargo bench` works on a fresh clone with
//! no setup.
//!
//! PUBLIC API ONLY, AND THAT IS A CONSTRAINT NOT A CHOICE
//! -------------------------------------------------------
//! `benches/` is a separate crate, exactly like `tests/api.rs`, so it sees only `pub`
//! items. `merge` and `PairIndex` are private and stay private — widening the library's
//! API for benchmarking convenience would be the tail wagging the dog. The consequence is
//! that `train` covers them indirectly: if `merge` or `PairIndex::apply_chunk_change`
//! regresses, `train` is where it shows up, just with less precision about where.
//!
//! READING THE RESULTS
//! -------------------
//! Criterion compares against the previous run ON THIS MACHINE, stored in
//! `target/criterion/` — which is gitignored, so there is no shared baseline and this is
//! not a CI gate on its own. It is a local tripwire. Turning it into a CI gate needs a
//! stored baseline or `critcmp`, which is a step-13 decision.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use naive_bbpe::Tokenizer;

/// 22,815 bytes of mixed-script prose — the step-6 fixture, checked in. Compiled into the
/// binary rather than read at runtime so file I/O never lands inside a measurement.
const ARTICLE: &str = include_str!("../tests/data/unicode_article.txt");

/// The whole training path: pre-tokenize, deduplicate, build the `PairIndex`, then merge.
///
/// Two vocabulary sizes because they stress different things. Step 10 made the per-merge
/// cost flat, so 512 and 2000 should differ roughly by the extra merges and nothing more —
/// if that ratio ever grows, the incremental-stats win is being undone and `train/2000`
/// will show it before `train/512` does.
fn bench_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("train");
    group.throughput(Throughput::Bytes(ARTICLE.len() as u64));
    for vocab_size in [512u32, 2000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            &vocab_size,
            |b, &size| {
                b.iter(|| {
                    let mut tok = Tokenizer::new();
                    tok.train(black_box(ARTICLE), size, false);
                    tok
                })
            },
        );
    }
    group.finish();
}

/// The pre-tokenizer alone — `fancy_regex::find_iter` over the corpus, nothing else.
///
/// Isolated because step 15 changes this exact regex (`\p{L}++` → `[\p{L}\p{M}]++`) and
/// a possessive-quantifier change is precisely the kind of edit that can quietly cost an
/// order of magnitude. Measuring it separately means that step gets an unambiguous answer.
fn bench_split(c: &mut Criterion) {
    let tok = Tokenizer::new();
    let mut group = c.benchmark_group("split");
    group.throughput(Throughput::Bytes(ARTICLE.len() as u64));
    group.bench_function("gpt4", |b| {
        b.iter(|| black_box(tok.split(black_box(ARTICLE)).len()))
    });
    group.finish();
}

/// Encoding, which is `encode_chunk` for every chunk.
///
/// Step 19's target: `encode_chunk` rebuilds a full `HashMap` of pair statistics on every
/// merge iteration within a chunk — O(len²) with a hash-map constant. This is the number
/// that step has to move, so it wants a baseline recorded before anything else changes.
fn bench_encode(c: &mut Criterion) {
    let mut tok = Tokenizer::new();
    tok.train(ARTICLE, 2000, false);

    let mut group = c.benchmark_group("encode");
    group.throughput(Throughput::Bytes(ARTICLE.len() as u64));
    group.bench_function("article", |b| {
        b.iter(|| black_box(tok.encode_ordinary(black_box(ARTICLE)).len()))
    });
    group.finish();
}

/// Decoding: a vocabulary lookup per id, then one lossy UTF-8 pass over the bytes.
///
/// Cheap and unlikely to regress on its own, but it is the other half of the round trip
/// and costs almost nothing to watch. If the refactor ever turns `vocab` from a `Vec`
/// indexed by id into something with a hash lookup, this is where it shows.
fn bench_decode(c: &mut Criterion) {
    let mut tok = Tokenizer::new();
    tok.train(ARTICLE, 2000, false);
    let ids = tok.encode_ordinary(ARTICLE);

    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Bytes(ARTICLE.len() as u64));
    group.bench_function("article", |b| {
        b.iter(|| black_box(tok.decode(black_box(&ids)).unwrap().len()))
    });
    group.finish();
}

criterion_group!(benches, bench_train, bench_split, bench_encode, bench_decode);
criterion_main!(benches);
