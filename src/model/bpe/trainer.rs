//! Training: corpus in, a [`BpeModel`] out.
//!
//! THE SPLIT, AND WHERE THE LINE FALLS
//! ------------------------------------
//! [`BpeTrainer`] is the machinery; [`BpeModel`] is what it produces. They were one type
//! until 12.3b, and separating them is justified from this file rather than by analogy:
//! `train` built `tally`, `chunks`, `counts` and a `PairIndex`, and **discarded every one of
//! them on return**. That scaffolding was already temporary in fact; it simply had nowhere
//! to live, so it lived on the type that also encodes.
//!
//! The line falls where the types change. `Tokenizer::train` keeps normalize → split →
//! tally, because those are `text/` work on `&str`. `BpeTrainer::train` takes ids and
//! returns a model, and never sees a character. The conversion between them is one line.
//!
//! Concretely, three things follow. A trained model is now produced by a *value*, not left
//! behind by a mutation, so a half-finished training run is no longer representable. Steps
//! 15–16 add roughly six training strategies and zero new models — each is a new trainer
//! returning the same `BpeModel`, rather than another way to mutate the type that encodes.
//! And `PairIndex`'s 127 lines of scaffolding no longer share a type with `decode`.
//!
//! `target_vocab_size`, NOT `vocab_size`
//! -------------------------------------
//! The trainer's number is a **target**: "learn until the table is this big." The model's
//! [`BpeModel::vocab_size`] is a **fact**: how big it actually is. They differ whenever
//! training runs out of pairs — `tiny.txt` asked for 400 and produced 268 — so giving them
//! the same name would invite exactly the wrong assumption.
//!
//! WHAT 12.3b DELIBERATELY DID NOT DO
//! -----------------------------------
//! `BpeTrainer` is `pub(crate)`. `Tokenizer::train` is still `&mut self` and still the only
//! way in from outside the crate, so the public API is byte-for-byte what it was. Making the
//! trainer/model split visible to callers is a step-13 decision.

use std::collections::HashMap;
use std::time::Instant;

use crate::Tokenizer;
use crate::model::bpe::model::{BYTE_ALPHABET, BpeModel, merge};
use crate::model::bpe::pair_index::PairIndex;
// `render_token` lives in `serialize.rs` — `write_vocab` is its main caller. The trainer
// uses it only for the sampled verbose report.
use crate::model::bpe::serialize::render_token;

/// The training machinery: configuration, plus the loop that turns pre-tokenized chunks
/// into a [`BpeModel`].
///
/// Holds no tables. Everything it builds — the `PairIndex`, the mutable chunk list — is
/// local to [`train`](BpeTrainer::train) and dropped when it returns.
pub(crate) struct BpeTrainer {
    /// The TARGET table size, including the 256 byte tokens. Training may finish under it
    /// if the corpus runs out of pairs; see the module docs.
    target_vocab_size: u32,
    verbose: bool,
    /// Started at construction rather than at the top of `train`, so the elapsed times in
    /// the verbose report still cover normalization and pre-tokenization — which is what
    /// they covered before 12.3b, and which is the point of printing them: telling a slow
    /// run from a hung one means counting the setup too.
    started: Instant,
}

impl BpeTrainer {
    pub(crate) fn new(target_vocab_size: u32, verbose: bool) -> Self {
        assert!(
            target_vocab_size >= BYTE_ALPHABET,
            "vocab_size must leave room for the 256 byte tokens"
        );
        BpeTrainer {
            target_vocab_size,
            verbose,
            started: Instant::now(),
        }
    }

    /// The merge loop. `chunks` are pre-tokenized and deduplicated; `counts[i]` is how many
    /// times `chunks[i]` occurred in the corpus.
    ///
    /// Takes `chunks` by value because it rewrites them in place on every merge, and the
    /// caller has no use for them afterwards.
    pub(crate) fn train(&self, mut chunks: Vec<Vec<u32>>, counts: Vec<u64>) -> BpeModel {
        let mut model = BpeModel::untrained();
        let new_tokens = self.target_vocab_size - BYTE_ALPHABET;

        // `verbose` used to print one line per merge. That was fine for 20 merges and is
        // useless for 49744 — so it now reports roughly 20 times over the whole run,
        // always including the first and last. Elapsed time matters more than the merge
        // itself once a run is measured in minutes: it is the only way to tell a slow
        // run from a hung one.
        let report_every = (new_tokens / 20).max(1);

        // Pair counts, the chunk index and the priority queue, built once and then kept
        // up to date. Before step 10 the equivalent of `stats` was rebuilt from scratch on
        // every iteration by scanning every chunk.
        let mut index = PairIndex::build(&chunks, &counts);

        for i in 0..new_tokens {
            // Python: `max(stats, key=stats.get)` — but taken off a heap rather than by
            // scanning, because `stats` has millions of entries and scanning it 50,000
            // times is minutes of work on its own.
            //
            // The ordering has not changed: it is still "highest count, ties to the
            // smallest pair", the rule introduced in step 2 because Rust HashMaps have no
            // iteration order and ties had to break deterministically.
            let Some((p0, p1)) = index.pop_best() else {
                // Reachable because chunks are short: they can all shrink to length 1.
                if self.verbose {
                    println!(
                        "no pairs left; stopping early at vocab size {}",
                        BYTE_ALPHABET + i
                    );
                }
                break;
            };

            // Mint the new token. One call rather than five lines, and it derives the id
            // itself — before 12.3a this was five lines here and five more in `load`, each
            // computing the id its own way (loop counter here, line index there).
            //
            // It has to happen BEFORE the chunks are rewritten, because `merge` needs the
            // id to substitute in. It used to sit after, which was possible only because
            // `idx` was computed separately from the table it belonged to. Nothing between
            // here and the loop reads the model, so the observable order is unchanged.
            let idx = model.push_merge((p0, p1));

            // Only the chunks that actually contain this pair are touched. Everything else
            // is untouched and its contribution to `stats` stays correct — which is the
            // entire point of the step.
            //
            // `counts` is never touched: merging rewrites what a chunk contains, never how
            // many times it occurs, so the two vectors stay aligned by construction.
            for ci in index.chunks_containing((p0, p1)) {
                let old = std::mem::take(&mut chunks[ci]);
                let new = merge(&old, (p0, p1), idx);
                index.apply_chunk_change(ci, &old, &new, counts[ci]);
                chunks[ci] = new;
            }
            #[cfg(test)]
            index.assert_consistent(&chunks, &counts, &format!("merging {:?}", (p0, p1)));

            if self.verbose && (i == 0 || i + 1 == new_tokens || (i + 1) % report_every == 0) {
                println!(
                    "  merge {:>6}/{}  {:>5} + {:<5} -> {:<5} {:>12}  [{:.1}s]",
                    i + 1,
                    new_tokens,
                    p0,
                    p1,
                    idx,
                    format!(
                        "{:?}",
                        render_token(model.token(idx).expect("just pushed"))
                    ),
                    self.started.elapsed().as_secs_f64(),
                );
            }
        }
        model
    }
}

impl Tokenizer {
    /// Train on `text`, replacing whatever this tokenizer had learned before.
    ///
    /// This half is the *text* pipeline — normalize, pre-tokenize, deduplicate — and the
    /// byte-to-id conversion at its end is the boundary between `text/` and `model/`. The
    /// merge loop itself is [`BpeTrainer::train`].
    pub fn train(&mut self, text: &str, vocab_size: u32, verbose: bool) {
        // Constructed first because it starts the clock, and the verbose report's elapsed
        // times are meant to include normalization and pre-tokenization. It also carries
        // the `vocab_size >= 256` assertion, so that still fires before any work is done.
        let trainer = BpeTrainer::new(vocab_size, verbose);

        let text = self.normalizer.normalize(text);

        // Pre-tokenize, then DEDUPLICATE. Merges are applied within a chunk and never
        // across one, so no token can span a boundary — that part is unchanged. What is
        // new is that identical chunks are collapsed.
        //
        // Natural text is enormously repetitive: 1 GB of English pre-tokenizes to roughly
        // 200 million chunks but only ~3 million *distinct* ones, because `" the"` alone
        // occurs tens of millions of times. Storing and re-processing every copy on every
        // iteration is the same work done over and over for no new information.
        //
        // Tallying into a `HashMap<&str, u64>` borrows from `text`, so the chunk strings
        // themselves are never copied — only counted.
        let mut tally: HashMap<&str, u64> = HashMap::new();
        let mut total_chunks: u64 = 0;
        for chunk in self.split(&text) {
            *tally.entry(chunk).or_insert(0) += 1;
            total_chunks += 1;
        }

        // Two parallel vectors rather than a map of chunk -> count: `merge` rebuilds the
        // ids on every iteration, and rebuilding a HashMap's keys is far more expensive
        // than rewriting a Vec in place. `counts[i]` belongs to `chunks[i]`.
        let mut chunks: Vec<Vec<u32>> = Vec::with_capacity(tally.len());
        let mut counts: Vec<u64> = Vec::with_capacity(tally.len());
        for (chunk, count) in tally {
            // THE ONE LINE where text becomes ids. Everything above is `text/`; everything
            // the trainer does below is `model/`.
            chunks.push(chunk.bytes().map(|b| b as u32).collect());
            counts.push(count);
        }

        if verbose {
            println!(
                "  pre-tokenized into {total_chunks} chunks, {} unique ({:.1}x less work)",
                chunks.len(),
                total_chunks as f64 / chunks.len().max(1) as f64,
            );
        }

        self.model = trainer.train(chunks, counts);
    }
}
