//! Training: corpus in, `merges` and `vocab` out.
//!
//! WHY THIS IS AN `impl Tokenizer` BLOCK IN A CHILD MODULE
//! -------------------------------------------------------
//! It moved here at 12.2b **without a single visibility change**, because in Rust a
//! private item is visible in its own module and every *descendant* of it. `Tokenizer`'s
//! fields are declared in `lib.rs`, the crate root; this file is a descendant; so
//! `self.merges` and `self.vocab` simply work. Contrast 12.2a, where free functions moved
//! the other way — `lib.rs` is the *parent* of `model/bpe/`, cannot see in, and every
//! moved item needed `pub(crate)`.
//!
//! WHAT 12.3 DOES TO THIS FILE
//! ----------------------------
//! `train` becomes `BpeTrainer::train(corpus) -> BpeModel`. The body survives nearly
//! verbatim — the tally, the dedup, the `PairIndex` loop — what changes is the signature
//! and where the results land. Moving it now, unchanged, is what lets 12.3 be a pure type
//! change with no file motion mixed into the diff.
//!
//! THE STAGES, IN ORDER
//! --------------------
//! normalize (`text/`) -> split (`text/`) -> **tally and deduplicate** -> ids -> merge loop.
//! The byte-to-id conversion is one line, and it is the boundary between `text/` and here.
//! Deduplication is step 9: 1 GB of English pre-tokenizes to ~200M chunks but only ~3M
//! *distinct* ones, and identical chunks contribute their count with a single addition.

use std::collections::HashMap;
use std::time::Instant;

use crate::Tokenizer;
use crate::model::bpe::model::{BYTE_ALPHABET, merge};
use crate::model::bpe::pair_index::PairIndex;
// `render_token` lives in `serialize.rs` — `write_vocab` is its main caller. `train` uses
// it only for the sampled verbose report.
use crate::model::bpe::serialize::render_token;

impl Tokenizer {
    /// All of training, in one place: text in, both tables filled.
    pub fn train(&mut self, text: &str, vocab_size: u32, verbose: bool) {
        assert!(
            vocab_size >= BYTE_ALPHABET,
            "vocab_size must leave room for the 256 byte tokens"
        );
        let new_tokens = vocab_size - BYTE_ALPHABET;

        // `verbose` used to print one line per merge. That was fine for 20 merges and is
        // useless for 49744 — so it now reports roughly 20 times over the whole run,
        // always including the first and last. Elapsed time matters more than the merge
        // itself once a run is measured in minutes: it is the only way to tell a slow
        // run from a hung one.
        let started = Instant::now();
        let report_every = (new_tokens / 20).max(1);

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
                if verbose {
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
            let idx = self.model.push_merge((p0, p1));

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

            if verbose && (i == 0 || i + 1 == new_tokens || (i + 1) % report_every == 0) {
                println!(
                    "  merge {:>6}/{}  {:>5} + {:<5} -> {:<5} {:>12}  [{:.1}s]",
                    i + 1,
                    new_tokens,
                    p0,
                    p1,
                    idx,
                    format!(
                        "{:?}",
                        render_token(self.model.token(idx).expect("just pushed"))
                    ),
                    started.elapsed().as_secs_f64(),
                );
            }
        }
    }
}
