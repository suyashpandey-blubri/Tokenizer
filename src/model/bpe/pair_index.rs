//! Incremental pair statistics — step 10's 47x, and the reason training is flat in vocab
//! size rather than linear.
//!
//! WHAT IT REPLACED
//! ----------------
//! Before step 10 the trainer rebuilt the whole pair table from every chunk on every
//! iteration. 7.5 MB at vocab 32k took 94.72 s; with this it takes 2.02 s.
//!
//! THE STRUCTURE
//! -------------
//! `stats` (pair -> count) kept alive across iterations, `where_` (pair -> which chunks
//! contain it) so only affected chunks are touched, and a `BinaryHeap` with **lazy stale
//! refresh** — pop, validate against `stats`, re-push if the entry is out of date. rustbpe
//! and huggingface/tokenizers both do this; minbpe deliberately does not.
//!
//! `pairs_of` LIVES HERE, not in a trainer file. Measured at 12.2: its only three call
//! sites are `build` and `apply_chunk_change`, both in this file. Putting it elsewhere
//! would split a function from its only caller.
//!
//! THE TWO CLASSIC DELTA BUGS, ELIMINATED BY CONSTRUCTION
//! ------------------------------------------------------
//! *Neighbour pairs*: `[x,a,a,y] -> [x,aa,y]` also destroys `(x,a)` and `(a,y)`.
//! *Overlapping runs*: `"aaaa"` holds 3 `(a,a)` but a merge replaces only 2.
//! Both are avoided by recomputing the whole pair set of an affected chunk rather than
//! hand-computing deltas. Chunks average ~6 ids, so recomputation costs nothing.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// The pairs inside one chunk, with how many times each occurs.
///
/// Note this counts *overlapping* occurrences: `[a,a,a]` yields `(a,a) -> 2`, even though
/// `merge` will only replace one of them. That is deliberate and matches what the
/// from-scratch loop counted, so the two agree.
pub(crate) fn pairs_of(ids: &[u32]) -> HashMap<(u32, u32), i64> {
    let mut pairs = HashMap::new();
    for window in ids.windows(2) {
        *pairs.entry((window[0], window[1])).or_insert(0) += 1;
    }
    pairs
}

/// The step-10 bookkeeping: pair counts kept **alive across merges**, plus an index of
/// which chunks contain which pair, plus a priority queue over the counts.
///
/// # Why
///
/// Up to step 9, `train` rebuilt `stats` from scratch every iteration by scanning every
/// chunk. But merging `(" t", "h")` only changes chunks that *contain* that pair — perhaps
/// 50,000 of 3,000,000. The other 98% of `stats` was already correct and was recomputed
/// anyway. This keeps it and patches only what moved.
///
/// # The cost
///
/// A from-scratch `stats` **cannot be stale**; this one is correct only if every update is.
/// A missed update does not crash — it silently picks slightly wrong winners and trains a
/// subtly worse tokenizer. [`Self::assert_consistent`] exists to make that failure loud in
/// tests, and the step's real check is that `.model` files come out byte-identical.
///
/// # The simplification that removes most of the risk
///
/// [`Self::apply_chunk_change`] does **not** hand-compute deltas the way rustbpe and
/// HuggingFace do. It recomputes the affected chunk's whole pair set before and after, and
/// diffs the two. A chunk is one pre-token — six or so ids — so this costs nothing, and it
/// makes two classic bugs impossible by construction:
///
/// - **neighbour pairs**: `[x,a,a,y] -> [x,aa,y]` destroys `(x,a)` and `(a,y)` as well as
///   `(a,a)`. Recomputing the set cannot miss them.
/// - **overlapping runs**: `"aaaa"` has three `(a,a)` occurrences but `merge` replaces only
///   two of them, so the delta is not `-3`. Recomputing sidesteps the arithmetic entirely.
///
/// What is left to get wrong is weighting and the `where_` bookkeeping, both narrow.
pub(crate) struct PairIndex {
    /// Pair -> total weighted count across the whole corpus. `i64` rather than `u64`
    /// because updates subtract, and an intermediate step can legitimately reach zero.
    stats: HashMap<(u32, u32), i64>,
    /// Pair -> indices of the chunks that currently contain it. This is what turns "scan
    /// three million chunks" into "touch fifty thousand".
    where_: HashMap<(u32, u32), HashSet<usize>>,
    /// Max-heap over `(count, Reverse(pair))`.
    ///
    /// That key is not new: it is exactly the expression `train` has used since step 2,
    /// where it existed because Rust's HashMap has no iteration order and ties had to be
    /// broken deterministically. It also happens to be rustbpe's heap ordering — count
    /// descending, tie-break pair ascending.
    ///
    /// Entries go stale as counts change. Rather than reheapifying on every update, stale
    /// entries are detected and repaired when popped; see [`Self::pop_best`].
    heap: BinaryHeap<(i64, Reverse<(u32, u32)>)>,
}

impl PairIndex {
    pub(crate) fn build(chunks: &[Vec<u32>], counts: &[u64]) -> Self {
        let mut stats: HashMap<(u32, u32), i64> = HashMap::new();
        let mut where_: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();

        for (i, (chunk, &weight)) in chunks.iter().zip(counts).enumerate() {
            for (pair, n) in pairs_of(chunk) {
                *stats.entry(pair).or_insert(0) += n * weight as i64;
                where_.entry(pair).or_default().insert(i);
            }
        }

        let heap = stats.iter().map(|(&pair, &count)| (count, Reverse(pair))).collect();
        PairIndex { stats, where_, heap }
    }

    /// The most frequent remaining pair, or `None` when nothing is left to merge.
    ///
    /// The heap can hold entries whose counts have since changed, so each pop is validated
    /// against `stats`: a pair that has vanished is dropped, and one whose count moved is
    /// pushed back with the current value and the pop retried. This is rustbpe's lazy
    /// refresh, and it is what makes updating a count O(1) instead of O(log n).
    pub(crate) fn pop_best(&mut self) -> Option<(u32, u32)> {
        while let Some((count, Reverse(pair))) = self.heap.pop() {
            let current = self.stats.get(&pair).copied().unwrap_or(0);
            if current <= 0 {
                continue; // the pair is gone; this entry is a leftover
            }
            if count != current {
                self.heap.push((current, Reverse(pair))); // stale count, requeue and retry
                continue;
            }
            return Some(pair);
        }
        None
    }

    pub(crate) fn chunks_containing(&self, pair: (u32, u32)) -> Vec<usize> {
        self.where_.get(&pair).map(|s| s.iter().copied().collect()).unwrap_or_default()
    }

    /// Record that chunk `index` changed from `old` to `new`.
    pub(crate) fn apply_chunk_change(&mut self, index: usize, old: &[u32], new: &[u32], weight: u64) {
        let before = pairs_of(old);
        let after = pairs_of(new);
        let weight = weight as i64;

        for (&pair, &n) in &before {
            // `where_` first: a pair present before and absent after loses this chunk.
            if !after.contains_key(&pair)
                && let Some(set) = self.where_.get_mut(&pair)
            {
                set.remove(&index);
                if set.is_empty() {
                    self.where_.remove(&pair);
                }
            }
            let entry = self.stats.entry(pair).or_insert(0);
            *entry -= n * weight;
            if *entry <= 0 {
                self.stats.remove(&pair);
            }
        }

        for (&pair, &n) in &after {
            if !before.contains_key(&pair) {
                self.where_.entry(pair).or_default().insert(index);
            }
            let entry = self.stats.entry(pair).or_insert(0);
            *entry += n * weight;
            // Any pair whose count went up needs a heap entry. Duplicates are harmless:
            // `pop_best` filters them.
            self.heap.push((*entry, Reverse(pair)));
        }
    }

    /// Rebuild from scratch and assert the incremental state matches.
    ///
    /// This is the check that would have caught every bug listed on the type above. It is
    /// O(corpus) so it is never run during real training — only from tests, on corpora
    /// small enough that running it after *every* merge is still instant.
    #[cfg(test)]
    pub(crate) fn assert_consistent(&self, chunks: &[Vec<u32>], counts: &[u64], after: &str) {
        let fresh = PairIndex::build(chunks, counts);
        assert_eq!(self.stats, fresh.stats, "stats drifted after {after}");
        assert_eq!(self.where_, fresh.where_, "where_ drifted after {after}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `apply_chunk_change` tests drive a REAL merge rather than hand-writing the
    // "after" chunk, so they cross into the sibling module. That is deliberate: hand-
    // writing it would let the test and `merge` disagree about overlapping runs, which is
    // exactly the bug these two tests exist to catch.
    use crate::model::bpe::model::merge;

    #[test]
    fn pairs_of_counts_overlapping_occurrences() {
        // Deliberately counts overlaps, matching what the from-scratch loop counted.
        // `merge` will only replace two of the three, but that is a different question.
        assert_eq!(pairs_of(&[97, 97, 97, 97]).get(&(97, 97)), Some(&3));
        assert_eq!(pairs_of(&[1, 2, 3]).len(), 2);
        assert!(pairs_of(&[1]).is_empty());
        assert!(pairs_of(&[]).is_empty());
    }

    #[test]
    fn pair_index_build_weights_by_chunk_count() {
        let chunks = vec![vec![1, 2], vec![1, 2]];
        let counts = vec![10, 5];
        let index = PairIndex::build(&chunks, &counts);
        assert_eq!(index.stats.get(&(1, 2)), Some(&15));
        assert_eq!(index.where_[&(1, 2)], [0, 1].into_iter().collect());
    }

    #[test]
    fn pop_best_returns_the_highest_count_with_ties_to_the_smallest_pair() {
        let chunks = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let counts = vec![10, 10, 3]; // (1,2) and (3,4) tie at 10
        let mut index = PairIndex::build(&chunks, &counts);
        // The step-2 tie-break rule, now expressed as a heap ordering.
        assert_eq!(index.pop_best(), Some((1, 2)));
    }

    #[test]
    fn pop_best_discards_stale_heap_entries() {
        // The lazy-refresh path. A pair's count changes while an old entry is still queued;
        // popping must notice and use the current value rather than the stale one.
        let chunks = vec![vec![1, 2], vec![3, 4]];
        let counts = vec![5, 3];
        let mut index = PairIndex::build(&chunks, &counts);

        // (1,2) disappears from chunk 0, which should demote it below (3,4).
        index.apply_chunk_change(0, &[1, 2], &[9], 5);
        assert_eq!(index.pop_best(), Some((3, 4)), "a stale entry won the pop");

        // And a pair that is gone entirely is skipped rather than returned with count 0.
        index.apply_chunk_change(1, &[3, 4], &[9], 3);
        assert_eq!(index.pop_best(), None);
    }

    #[test]
    fn apply_chunk_change_handles_neighbour_pairs() {
        // `[x,a,a,y] -> [x,aa,y]` destroys (x,a) and (a,y) as well as (a,a), and creates
        // (x,aa) and (aa,y). Hand-computing deltas is where implementations forget the
        // neighbours; recomputing the whole pair set cannot.
        let chunks = vec![vec![7, 1, 1, 8]];
        let counts = vec![1];
        let mut index = PairIndex::build(&chunks, &counts);

        let merged = merge(&chunks[0], (1, 1), 256);
        assert_eq!(merged, vec![7, 256, 8]);
        index.apply_chunk_change(0, &chunks[0], &merged, 1);

        index.assert_consistent(&[merged], &counts, "a neighbour-pair merge");
        for gone in [(7, 1), (1, 1), (1, 8)] {
            assert!(!index.stats.contains_key(&gone), "{gone:?} should be gone");
        }
        assert_eq!(index.stats.get(&(7, 256)), Some(&1));
        assert_eq!(index.stats.get(&(256, 8)), Some(&1));
    }

    #[test]
    fn apply_chunk_change_handles_overlapping_runs() {
        // "aaaa": pairs_of says (a,a) occurs 3 times, but merge produces [aa,aa] and
        // replaces only two. The delta is -3, not -2 — which is exactly why this
        // recomputes rather than subtracting one per replacement.
        let chunks = vec![vec![1, 1, 1, 1]];
        let counts = vec![4];
        let mut index = PairIndex::build(&chunks, &counts);
        assert_eq!(index.stats.get(&(1, 1)), Some(&12)); // 3 occurrences x weight 4

        let merged = merge(&chunks[0], (1, 1), 256);
        assert_eq!(merged, vec![256, 256]);
        index.apply_chunk_change(0, &chunks[0], &merged, 4);

        index.assert_consistent(&[merged], &counts, "an overlapping-run merge");
        assert!(!index.stats.contains_key(&(1, 1)));
        assert_eq!(index.stats.get(&(256, 256)), Some(&4));
    }
}
