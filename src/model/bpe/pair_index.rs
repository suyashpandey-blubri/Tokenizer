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

use crate::text::boundary::{Boundaries, CorpusBoundaries};

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// The pairs inside one chunk, with how many times each occurs.
///
/// Note this counts *overlapping* occurrences: `[a,a,a]` yields `(a,a) -> 2`, even though
/// `merge` will only replace one of them. That is deliberate and matches what the
/// from-scratch loop counted, so the two agree.
pub(crate) fn pairs_of(ids: &[u32], bounds: &Boundaries) -> HashMap<(u32, u32), i64> {
    let mut pairs = HashMap::new();
    // `windows(2)` yields one window per GAP, so the window index IS the gap index:
    // window `i` spans `ids[i] | ids[i+1]`. Skipping a forbidden gap here is the whole
    // point of threading boundaries into COUNTING and not only into applying — an
    // occurrence the trainer may not merge must not contribute to the pair's score, or
    // the pair wins on evidence it cannot act on and the ranking is silently wrong.
    for (gap, window) in ids.windows(2).enumerate() {
        if !bounds.permits(gap) {
            continue;
        }
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
    pub(crate) fn build(chunks: &[Vec<u32>], counts: &[u64], bounds: &CorpusBoundaries) -> Self {
        let mut stats: HashMap<(u32, u32), i64> = HashMap::new();
        let mut where_: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();

        for (i, (chunk, &weight)) in chunks.iter().zip(counts).enumerate() {
            for (pair, n) in pairs_of(chunk, bounds.for_chunk(i)) {
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
    /// `old_bounds` and `new_bounds` are the gap masks before and after the merge, and they
    /// are genuinely different objects: a merge consumes one gap and renumbers every gap to
    /// its right, so `before` must be counted under the old mask and `after` under the new.
    /// Using one for both would miscount the delta the moment boundaries stop being uniform.
    pub(crate) fn apply_chunk_change(
        &mut self,
        index: usize,
        old: &[u32],
        old_bounds: &Boundaries,
        new: &[u32],
        new_bounds: &Boundaries,
        weight: u64,
    ) {
        let before = pairs_of(old, old_bounds);
        let after = pairs_of(new, new_bounds);
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
    pub(crate) fn assert_consistent(
        &self,
        chunks: &[Vec<u32>],
        counts: &[u64],
        bounds: &CorpusBoundaries,
        after: &str,
    ) {
        let fresh = PairIndex::build(chunks, counts, bounds);
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
        assert_eq!(pairs_of(&[97, 97, 97, 97], &Boundaries::Unrestricted).get(&(97, 97)), Some(&3));
        assert_eq!(pairs_of(&[1, 2, 3], &Boundaries::Unrestricted).len(), 2);
        assert!(pairs_of(&[1], &Boundaries::Unrestricted).is_empty());
        assert!(pairs_of(&[], &Boundaries::Unrestricted).is_empty());
    }

    /// THE LIVE-SEAM TEST for counting. Everything production does today passes
    /// `Unrestricted`, so without this the boundary threading would compile, pass every
    /// other test, and be **completely inert** — a seam nobody had shown to do anything.
    ///
    /// This is the half of B3 that fails *silently* if it is missed. Suppose gap 1 is
    /// forbidden but `pairs_of` counts it anyway: the trainer sees `(97, 97)` occurring
    /// three times, ranks it first on that evidence, and then `merge` legally applies only
    /// two of them. The winner was chosen on occurrences it was never allowed to touch, and
    /// nothing crashes — the vocabulary is just quietly worse.
    #[test]
    fn a_forbidden_gap_is_not_counted() {
        //   ids:   [97][97][97][97]
        //   gaps:     0   1   2
        let ids = [97, 97, 97, 97];
        assert_eq!(
            pairs_of(&ids, &Boundaries::Unrestricted).get(&(97, 97)),
            Some(&3),
            "sanity: all three gaps counted when nothing is forbidden"
        );

        let one_forbidden = Boundaries::Mask(vec![true, false, true]);
        assert_eq!(
            pairs_of(&ids, &one_forbidden).get(&(97, 97)),
            Some(&2),
            "forbidding gap 1 must drop the count to 2"
        );

        let all_forbidden = Boundaries::Mask(vec![false, false, false]);
        assert!(
            pairs_of(&ids, &all_forbidden).is_empty(),
            "a chunk with every gap forbidden contributes no pairs at all"
        );
    }

    /// The same claim one level up: a forbidden gap must not reach `stats` or `where_`
    /// either, or `pop_best` would still rank on it.
    #[test]
    fn pair_index_build_respects_forbidden_gaps() {
        let chunks = vec![vec![97, 97, 97, 97]];
        let counts = vec![10];

        let open = PairIndex::build(&chunks, &counts, &CorpusBoundaries::AllUnrestricted);
        assert_eq!(open.stats.get(&(97, 97)), Some(&30), "3 gaps x weight 10");

        let restricted = CorpusBoundaries::PerChunk(vec![Boundaries::Mask(vec![
            true, false, true,
        ])]);
        let closed = PairIndex::build(&chunks, &counts, &restricted);
        assert_eq!(closed.stats.get(&(97, 97)), Some(&20), "2 gaps x weight 10");

        let shut = CorpusBoundaries::PerChunk(vec![Boundaries::Mask(vec![false; 3])]);
        let none = PairIndex::build(&chunks, &counts, &shut);
        assert!(
            none.stats.is_empty() && none.where_.is_empty(),
            "nothing counted means nothing indexed"
        );
    }

    #[test]
    fn pair_index_build_weights_by_chunk_count() {
        let chunks = vec![vec![1, 2], vec![1, 2]];
        let counts = vec![10, 5];
        let index = PairIndex::build(&chunks, &counts, &CorpusBoundaries::AllUnrestricted);
        assert_eq!(index.stats.get(&(1, 2)), Some(&15));
        assert_eq!(index.where_[&(1, 2)], [0, 1].into_iter().collect());
    }

    #[test]
    fn pop_best_returns_the_highest_count_with_ties_to_the_smallest_pair() {
        let chunks = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let counts = vec![10, 10, 3]; // (1,2) and (3,4) tie at 10
        let mut index = PairIndex::build(&chunks, &counts, &CorpusBoundaries::AllUnrestricted);
        // The step-2 tie-break rule, now expressed as a heap ordering.
        assert_eq!(index.pop_best(), Some((1, 2)));
    }

    #[test]
    fn pop_best_discards_stale_heap_entries() {
        // The lazy-refresh path. A pair's count changes while an old entry is still queued;
        // popping must notice and use the current value rather than the stale one.
        let chunks = vec![vec![1, 2], vec![3, 4]];
        let counts = vec![5, 3];
        let mut index = PairIndex::build(&chunks, &counts, &CorpusBoundaries::AllUnrestricted);

        // (1,2) disappears from chunk 0, which should demote it below (3,4).
        index.apply_chunk_change(
            0,
            &[1, 2],
            &Boundaries::Unrestricted,
            &[9],
            &Boundaries::Unrestricted,
            5,
        );
        assert_eq!(index.pop_best(), Some((3, 4)), "a stale entry won the pop");

        // And a pair that is gone entirely is skipped rather than returned with count 0.
        index.apply_chunk_change(
            1,
            &[3, 4],
            &Boundaries::Unrestricted,
            &[9],
            &Boundaries::Unrestricted,
            3,
        );
        assert_eq!(index.pop_best(), None);
    }

    #[test]
    fn apply_chunk_change_handles_neighbour_pairs() {
        // `[x,a,a,y] -> [x,aa,y]` destroys (x,a) and (a,y) as well as (a,a), and creates
        // (x,aa) and (aa,y). Hand-computing deltas is where implementations forget the
        // neighbours; recomputing the whole pair set cannot.
        let chunks = vec![vec![7, 1, 1, 8]];
        let counts = vec![1];
        let mut index = PairIndex::build(&chunks, &counts, &CorpusBoundaries::AllUnrestricted);

        let merged = merge(&chunks[0], (1, 1), 256, &Boundaries::Unrestricted).0;
        assert_eq!(merged, vec![7, 256, 8]);
        index.apply_chunk_change(
            0,
            &chunks[0],
            &Boundaries::Unrestricted,
            &merged,
            &Boundaries::Unrestricted,
            1,
        );

        index.assert_consistent(
            &[merged],
            &counts,
            &CorpusBoundaries::AllUnrestricted,
            "a neighbour-pair merge",
        );
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
        let mut index = PairIndex::build(&chunks, &counts, &CorpusBoundaries::AllUnrestricted);
        assert_eq!(index.stats.get(&(1, 1)), Some(&12)); // 3 occurrences x weight 4

        let merged = merge(&chunks[0], (1, 1), 256, &Boundaries::Unrestricted).0;
        assert_eq!(merged, vec![256, 256]);
        index.apply_chunk_change(
            0,
            &chunks[0],
            &Boundaries::Unrestricted,
            &merged,
            &Boundaries::Unrestricted,
            4,
        );

        index.assert_consistent(
            &[merged],
            &counts,
            &CorpusBoundaries::AllUnrestricted,
            "an overlapping-run merge",
        );
        assert!(!index.stats.contains_key(&(1, 1)));
        assert_eq!(index.stats.get(&(256, 256)), Some(&4));
    }
}
