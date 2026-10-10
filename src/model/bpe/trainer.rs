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
use std::io::{self, Read};
use std::sync::mpsc;
use std::time::Instant;

use crate::Tokenizer;
use crate::text::pretokenizer::BLOCK_BYTES;
use crate::model::bpe::model::{BYTE_ALPHABET, BpeModel, merge};
use crate::model::bpe::pair_index::PairIndex;
use crate::text::boundary::CorpusBoundaries;
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
    /// `bounds[i]` says which gaps inside `chunks[i]` a merge may cross. Today every entry
    /// is `Unrestricted` — B0 forbids nothing — and the equivalence gate is what proves the
    /// threading is inert. 15.4 is where non-trivial masks start arriving.
    pub(crate) fn train(
        &self,
        mut chunks: Vec<Vec<u32>>,
        mut bounds: CorpusBoundaries,
        counts: Vec<u64>,
    ) -> BpeModel {
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
        let mut index = PairIndex::build(&chunks, &counts, &bounds);

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
                let old_bounds = bounds.for_chunk(ci).clone();
                // `merge` returns the rewritten chunk AND its gap mask, because a merge
                // consumes a gap and renumbers the ones to its right. The index has to see
                // both masks: `before` is counted under the old one, `after` under the new.
                let (new, new_bounds) = merge(&old, (p0, p1), idx, &old_bounds);
                index.apply_chunk_change(
                    ci,
                    &old,
                    &old_bounds,
                    &new,
                    &new_bounds,
                    counts[ci],
                );
                chunks[ci] = new;
                bounds.set_chunk(ci, new_bounds);
            }
            #[cfg(test)]
            index.assert_consistent(&chunks, &counts, &bounds, &format!("merging {:?}", (p0, p1)));

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

/// Pre-tokenized chunks, DEDUPLICATED: each distinct chunk once, with how often it occurred.
///
/// Merges are applied within a chunk and never across one, so identical chunks can be
/// collapsed. Natural text is enormously repetitive: 1 GB of English pre-tokenizes to
/// roughly 200 million chunks but only ~1.6 million *distinct* ones, because `" the"` alone
/// occurs tens of millions of times. Storing and re-processing every copy on every
/// iteration is the same work done over and over for no new information.
///
/// THE KEYS ARE OWNED `String`s, NOT `&str` (prod.1)
/// ------------------------------------------------
/// Until prod.1 the tally was a `HashMap<&str, u64>` borrowing from the text, so no chunk
/// was ever copied — but the whole text had to stay in memory until counting ended, and at
/// 1.2 TB it cannot. Owning the keys lets the text arrive in blocks that are counted and
/// dropped one at a time, so memory is set by the number of DISTINCT chunks, not by the
/// size of the corpus. The cost is one copy of each distinct chunk.
struct Tally {
    counts: HashMap<String, u64>,
    /// Every chunk seen, repeats included — for the verbose report only.
    total: u64,
}

impl Tally {
    fn new() -> Self {
        Tally {
            counts: HashMap::new(),
            total: 0,
        }
    }

    /// Normalize, pre-tokenize and count one piece of text. Called once with the whole
    /// corpus by [`Tokenizer::train`], once per block by [`Tokenizer::train_from_reader`];
    /// the two give the same tally because blocks are cut only where the pre-tokenizer
    /// would cut anyway (see `last_safe_cut`).
    fn add(&mut self, tok: &Tokenizer, text: &str) {
        let text = tok.normalizer.normalize(text);
        for chunk in tok.split(&text) {
            // `get_mut` first, so a chunk already counted costs no allocation. `entry` would
            // need an owned `String` key up front, copying every one of the 200 million.
            match self.counts.get_mut(chunk) {
                Some(count) => *count += 1,
                None => {
                    self.counts.insert(chunk.to_owned(), 1);
                }
            }
            self.total += 1;
        }
    }

    /// Add another tally's counts into this one. This is how the private tallies the
    /// counting threads build are combined into the main one (prod.3). The keys are MOVED,
    /// not copied: a chunk new to this tally takes the other tally's `String` as it is.
    fn absorb(&mut self, other: Tally) {
        for (chunk, count) in other.counts {
            match self.counts.get_mut(&chunk) {
                Some(mine) => *mine += count,
                None => {
                    self.counts.insert(chunk, count);
                }
            }
        }
        self.total += other.total;
    }
}

impl Tokenizer {
    /// Train on `text`, replacing whatever this tokenizer had learned before.
    ///
    /// This half is the *text* pipeline — normalize, pre-tokenize, deduplicate — and the
    /// byte-to-id conversion at its end is the boundary between `text/` and `model/`. The
    /// merge loop itself is `BpeTrainer::train`.
    pub fn train(&mut self, text: &str, vocab_size: u32, verbose: bool) {
        // Constructed first because it starts the clock, and the verbose report's elapsed
        // times are meant to include normalization and pre-tokenization. It also carries
        // the `vocab_size >= 256` assertion, so that still fires before any work is done.
        let trainer = BpeTrainer::new(vocab_size, verbose);

        // The whole text as ONE piece — never cut, so this gives the same chunks for any
        // pattern, not only the two `train_from_reader` is proven for.
        let mut tally = Tally::new();
        tally.add(self, text);
        self.train_on(trainer, tally, verbose);
    }

    /// Train on everything `reader` yields, without ever holding all of it in memory:
    /// the input is read in ~64 MB blocks, each counted and dropped. Learns exactly what
    /// [`train`](Tokenizer::train) would learn on the same text.
    ///
    /// Errors if the input is not valid UTF-8, if reading fails, or if this tokenizer's
    /// pattern is not [`GPT4_SPLIT_PATTERN`](crate::GPT4_SPLIT_PATTERN) or
    /// [`GPT2_SPLIT_PATTERN`](crate::GPT2_SPLIT_PATTERN) — the only two the block cut is
    /// proven safe for. On an error nothing is learned and the old model is kept.
    ///
    /// `threads` is how many blocks are counted at the same time (0 is taken as 1). It
    /// changes only speed and memory — the model learned is byte-identical for every value.
    /// Learning the merges afterwards is single-threaded whatever it is.
    pub fn train_from_reader<R: Read>(
        &mut self,
        reader: R,
        vocab_size: u32,
        threads: usize,
        verbose: bool,
    ) -> io::Result<()> {
        self.train_from_reader_in_blocks(reader, vocab_size, threads, BLOCK_BYTES, verbose)
    }

    /// [`train_from_reader`](Tokenizer::train_from_reader) with the block size as a
    /// parameter, so tests can use tiny blocks and get many batches out of a small text.
    pub(crate) fn train_from_reader_in_blocks<R: Read>(
        &mut self,
        reader: R,
        vocab_size: u32,
        threads: usize,
        block_bytes: usize,
        verbose: bool,
    ) -> io::Result<()> {
        let trainer = BpeTrainer::new(vocab_size, verbose);
        let tally = self.count_in_parallel(reader, threads.max(1), block_bytes)?;
        self.train_on(trainer, tally, verbose);
        Ok(())
    }

    /// PARALLEL COUNTING (prod.3) — the only place in the crate that uses threads, through
    /// the `rayon` crate.
    ///
    /// Counting one block never needs anything from another block, and adding counts gives
    /// the same total in any order, so blocks can be counted on different threads and the
    /// result is exactly the tally one thread would have built. Learning the merges cannot
    /// be split this way — each merge depends on the one before — and is untouched.
    ///
    /// HOW IT RUNS
    /// -----------
    /// `rayon` keeps a POOL of `threads` worker threads. The calling thread does not count;
    /// it reads and combines:
    ///
    /// 1. it reads the next block and hands it to the pool as a TASK (`scope.spawn`). An
    ///    idle worker picks it up and normalizes, splits and counts it into its OWN tally;
    /// 2. a finished task sends its tally back over a channel (`mpsc`, a queue between
    ///    threads in Rust's standard library);
    /// 3. once `2 × threads` blocks are in flight, the calling thread waits for one tally,
    ///    adds it into the main one (`absorb`), and only then reads another block. That cap
    ///    is what bounds memory: about one block being counted per worker plus as many
    ///    waiting, never the whole input.
    ///
    /// So reading, counting and combining all happen at the same time, on different threads.
    ///
    /// Why each task has a private tally rather than all of them sharing one: a shared table
    /// can be changed by only one thread at a time, and each block holds ~12 million chunk
    /// occurrences, so the workers would mostly wait on each other. Private tallies need no
    /// coordination. Their cost is `absorb`, and a 64 MB block has only ~269,000 *distinct*
    /// chunks, so combining is ~45× less work than counting (measured at prod.3).
    ///
    /// Why `rayon` and not `std::thread`: either would do this design. `rayon` is what
    /// HuggingFace `tokenizers` and rustbpe use, it hands tasks to whichever worker is free,
    /// and switching to `std::thread::scope` later would touch only this function.
    fn count_in_parallel<R: Read>(
        &self,
        reader: R,
        threads: usize,
        block_bytes: usize,
    ) -> io::Result<Tally> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            // The single-threaded path ran the regex on the main thread, whose stack is
            // 8 MB; a spawned thread's default is 2 MB. Matching it keeps the pre-tokenizer
            // running under the same conditions it always has.
            .stack_size(8 << 20)
            .build()
            .map_err(io::Error::other)?;
        let (send, receive) = mpsc::channel::<Tally>();
        let mut tally = Tally::new();
        let mut in_flight = 0usize;

        // `in_place_scope` runs this closure on the CALLING thread, and does not return
        // until every task spawned inside it has finished — which is what lets each task
        // borrow `self` (for the normalizer and pre-tokenizer) instead of copying it. If
        // reading fails part-way, it still waits for the tasks already started.
        let read = pool.in_place_scope(|scope| {
            self.pattern.for_each_block(reader, block_bytes, |block| {
                // The reader reuses its buffer, so the block is copied out to be handed to
                // a task: ~10 ms per 64 MB, next to ~2 s to count it.
                let block = block.to_owned();
                let send = send.clone();
                scope.spawn(move |_| {
                    // ---- runs on a worker thread ----
                    let mut mine = Tally::new();
                    mine.add(self, &block);
                    send.send(mine).expect("the receiver outlives every task");
                });
                in_flight += 1;
                if in_flight == 2 * threads {
                    tally.absorb(receive.recv().expect("a task is in flight"));
                    in_flight -= 1;
                }
                Ok(())
            })
        });
        read?;
        // The scope has returned, so every task has finished and sent its tally; dropping
        // the last sender lets the loop below end once the channel is empty.
        drop(send);
        for mine in receive {
            tally.absorb(mine);
        }
        Ok(tally)
    }

    /// Hand a finished tally to the trainer. Shared by both entry points, so everything
    /// after counting is one code path.
    fn train_on(&mut self, trainer: BpeTrainer, tally: Tally, verbose: bool) {
        // Two parallel vectors rather than a map of chunk -> count: `merge` rebuilds the
        // ids on every iteration, and rebuilding a HashMap's keys is far more expensive
        // than rewriting a Vec in place. `counts[i]` belongs to `chunks[i]`.
        let total_chunks = tally.total;
        let tally = tally.counts;
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

        // Boundaries are derived from TEXT, so this loop — the one place the chunk text is
        // still in hand — is where 15.4 will build real masks. B0 forbids nothing, and
        // `AllUnrestricted` says so without allocating anything per chunk.
        self.model = trainer.train(chunks, CorpusBoundaries::AllUnrestricted, counts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::boundary::Boundaries;

    /// prod.3: the tally built by the worker threads is EXACTLY the one-thread tally —
    /// every chunk's count, and the total. Compared directly rather than through the merges
    /// they produce, because a counting error can leave the winning pairs unchanged on a
    /// small text and still be wrong.
    #[test]
    fn parallel_counting_gives_the_tally_one_thread_gives() {
        const CASES: &[&str] = include!("../../../tests/data/round_trip_cases.rs");
        // ".\nx" puts a legal block cut after every case, so tiny blocks give many tasks.
        let text: String = CASES.iter().map(|c| format!("{c}.\nx{c} \n")).collect();
        let tok = Tokenizer::new();
        let mut one = Tally::new();
        one.add(&tok, &text);
        for block_bytes in [7, 64, 1000] {
            for threads in [1, 2, 3, 8] {
                let many = tok.count_in_parallel(text.as_bytes(), threads, block_bytes).unwrap();
                assert_eq!(many.counts, one.counts, "{block_bytes}-byte blocks, {threads} threads");
                assert_eq!(many.total, one.total, "{block_bytes}-byte blocks, {threads} threads");
            }
        }
    }

    /// THE LIVE-SEAM TEST for the training loop itself — added at 12.6, because the
    /// post-12.4b audit showed three mutations of this loop that no test noticed. Every
    /// other caller goes through `Tokenizer::train`, which always passes `AllUnrestricted`,
    /// so this has to call `BpeTrainer::train` directly to hand it a real mask.
    ///
    /// ```text
    ///   ids:   [a][a][a][a]
    ///   gaps:     0  1  2
    ///   mask:     x  ok x
    /// ```
    ///
    /// `(a, a)` has exactly one permitted occurrence, so it is merged once, into
    /// `[a][aa][a]`. Both remaining gaps are still forbidden, so training stops at ONE merge.
    /// Without the mask the same chunk trains two (`aa`, then `aaaa`).
    ///
    /// What each mutation does to this, and what notices:
    ///
    /// - **merge is handed `Unrestricted` instead of the mask** — it crosses the forbidden
    ///   gaps and trains exactly as if there were no mask: two merges. The merge list says so.
    /// - **the new mask is never written back** — the chunk is now 3 ids with 2 gaps but is
    ///   still paired with the old 3-gap mask. `assert_consistent` recounts after the merge
    ///   under that stale mask, disagrees with the running totals, and panics.
    /// - **the new chunk is counted under the OLD mask** — old gap 1 is permitted, so
    ///   `(aa, a)` gets counted when it should not. The recount disagrees; left to run, it
    ///   would also learn a second merge.
    #[test]
    fn training_respects_a_mask_across_merges() {
        let a = 97;
        let chunk = vec![a, a, a, a];
        let trainer = BpeTrainer::new(BYTE_ALPHABET + 4, false); // room for 4; needs fewer

        let open = trainer.train(vec![chunk.clone()], CorpusBoundaries::AllUnrestricted, vec![1]);
        assert_eq!(
            open.merges_in_learned_order(),
            vec![(256, (a, a)), (257, (256, 256))],
            "sanity: with nothing forbidden, aaaa trains aa then aaaa"
        );

        let masked = CorpusBoundaries::PerChunk(vec![Boundaries::Mask(vec![false, true, false])]);
        let restricted = trainer.train(vec![chunk], masked, vec![1]);
        assert_eq!(
            restricted.merges_in_learned_order(),
            vec![(256, (a, a))],
            "only the middle gap may be merged, so exactly one merge is possible"
        );
    }
}
