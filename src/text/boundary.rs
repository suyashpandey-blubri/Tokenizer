//! Which positions inside a chunk a merge is allowed to cross.
//!
//! WHY THIS EXISTS BEFORE ANYTHING NEEDS IT
//! -----------------------------------------
//! Steps 15.4 (B3, constrained BPE) and 15.5 (B4, morphology-aware boundaries) both ask the
//! same question — *may these two adjacent tokens be joined?* — and the expensive part of
//! answering it is not the data structure. It is that the question has to be asked in
//! **six** places, and one group of them fails silently:
//!
//! | | site | failure mode if missed |
//! |---|---|---|
//! | counting | `PairIndex::build`, `apply_chunk_change` ×2 | **silent** — the trainer picks a winning pair whose count is mostly occurrences it is not allowed to apply, so the merge ranking is wrong and nothing crashes |
//! | applying | `BpeTrainer::train`, `encode_chunk` | loud — round-trip tests fail |
//!
//! Threading those six is knowable today and independent of whatever B3 turns out to need.
//! The *representation* is not, which is why this file knows what a boundary **is** and
//! nothing about where boundaries come from. No akshara logic, no morpheme logic, nothing
//! Indic — that is 15.1–15.5, behind a research gate.
//!
//! GAPS, NOT POSITIONS
//! -------------------
//! A chunk of `n` ids has `n - 1` gaps. Gap `i` sits between `ids[i]` and `ids[i + 1]`:
//!
//! ```text
//!   ids:    [ a ][ b ][ c ][ d ]
//!   gaps:       0    1    2
//! ```
//!
//! Merging `(b, c)` consumes gap 1 and the chunk becomes `[a][bc][d]` with gaps `0, 1` —
//! where the new gap 1 is the old gap 2. So applying a merge **removes the gap it crossed**
//! and renumbers everything to its right. [`merge`](crate::model::bpe::model::merge) is where
//! that happens: it returns the rewritten ids and their renumbered mask together (12.4b).
//!
//! IF THIS TURNS OUT TO BE THE WRONG ABSTRACTION
//! ----------------------------------------------
//! Removing it is a bounded edit, deliberately: this file, the `bounds` parameter on
//! `pairs_of` / `get_stats_into` / `merge` / `PairIndex`, the `CorpusBoundaries` argument to
//! `BpeTrainer::train`, and the two lines that pass in the unrestricted case
//! (`Tokenizer::train`, `encode_chunk`). Nothing else knows it exists, and `Tokenizer`'s public API does not mention it. Written down
//! because step 15's research may well redefine what a boundary has to express, and this
//! seam should not become load-bearing by accident.

/// Which gaps in one chunk a merge may cross.
///
/// Two variants rather than always carrying a mask, because B0 — everything the project
/// does today — never forbids anything, and paying a `Vec<bool>` per chunk for a mask that
/// is uniformly `true` would be a real cost on a 1 GB corpus for no information.
/// [`permits`](Boundaries::permits) on `Unrestricted` is a constant `true`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Boundaries {
    /// Every gap is mergeable. The B0 case, and the only one production constructs today.
    Unrestricted,
    /// `mask[i]` says whether a merge may cross gap `i`. Length is `ids.len() - 1`.
    ///
    /// Constructed only by tests at 12.4 — it exists so the seam can be *shown* to change
    /// behaviour rather than merely compile. A seam with one trivial implementation that
    /// nothing ever exercises differently has not been verified.
    ///
    /// **`allow(dead_code)` is a tripwire, not housekeeping.** Production constructs only
    /// `Unrestricted`, so without it the build warns. **Delete this attribute at 15.4**,
    /// when a text-to-mask function starts producing these for real. If it is still here
    /// after step 15, the seam never found a consumer and this whole file should go —
    /// an outcome the step-15 research must stay free to reach.
    #[allow(dead_code)]
    Mask(Vec<bool>),
}

impl Boundaries {
    /// May a merge cross gap `i`?
    ///
    /// Out-of-range answers `false`: a gap that does not exist cannot be crossed, and
    /// returning `true` there would turn an indexing bug into a silent extra merge.
    pub(crate) fn permits(&self, gap: usize) -> bool {
        match self {
            Boundaries::Unrestricted => true,
            Boundaries::Mask(mask) => mask.get(gap).copied().unwrap_or(false),
        }
    }
}

/// Borrowed to hand out for the uniform case, so `AllUnrestricted` stores nothing at all.
const UNRESTRICTED: Boundaries = Boundaries::Unrestricted;

/// The boundaries for a whole deduplicated corpus.
///
/// WHY THIS EXISTS RATHER THAN A PLAIN `Vec<Boundaries>`
/// -----------------------------------------------------
/// It was a `Vec<Boundaries>` first, and that was **measured at +11.2% peak RSS** on
/// `rust.txt` (5376 KB → 5980 KB) — 24 bytes per distinct chunk, for a vector whose every
/// entry said "nothing is forbidden". At 1 GB, where dedup leaves ~3 million distinct
/// chunks, that is ~72 MB spent to store the same word three million times.
///
/// Paying that for a feature with no consumer yet is not a cost B0 should carry, so the
/// uniform case is hoisted to a variant of its own and allocates nothing. `PerChunk` is the
/// shape 15.4 needs, unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CorpusBoundaries {
    /// No chunk restricts anything — the B0 case. **Zero storage.**
    AllUnrestricted,
    /// One entry per chunk, indexed exactly as `chunks` is.
    #[allow(dead_code)] // tripwire — see the note on `Boundaries::Mask`
    PerChunk(Vec<Boundaries>),
}

impl CorpusBoundaries {
    /// The boundaries for chunk `i`.
    pub(crate) fn for_chunk(&self, i: usize) -> &Boundaries {
        match self {
            CorpusBoundaries::AllUnrestricted => &UNRESTRICTED,
            CorpusBoundaries::PerChunk(per) => &per[i],
        }
    }

    /// Replace chunk `i`'s mask after a merge rewrote it.
    ///
    /// A no-op for `AllUnrestricted`, and correctly so: merging cannot make an unrestricted
    /// chunk restricted, so there is nothing to record. That is also why the B0 path still
    /// allocates nothing per chunk even though the trainer now writes back on every merge.
    pub(crate) fn set_chunk(&mut self, i: usize, bounds: Boundaries) {
        match self {
            CorpusBoundaries::AllUnrestricted => {
                debug_assert_eq!(
                    bounds,
                    Boundaries::Unrestricted,
                    "an unrestricted chunk cannot become restricted by merging"
                );
            }
            CorpusBoundaries::PerChunk(per) => per[i] = bounds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unrestricted_permits_every_gap_including_absurd_ones() {
        // No length is carried, so there is nothing to be out of range OF. That is the
        // point: the B0 case costs no storage and no bookkeeping.
        let b = Boundaries::Unrestricted;
        assert!(b.permits(0));
        assert!(b.permits(9_999));
    }

    #[test]
    fn a_mask_forbids_exactly_the_gaps_it_says() {
        //   ids:    [ a ][ b ][ c ][ d ]
        //   gaps:       0    1    2
        let b = Boundaries::Mask(vec![true, false, true]);
        assert!(b.permits(0));
        assert!(!b.permits(1), "gap 1 is forbidden");
        assert!(b.permits(2));
    }

    #[test]
    fn a_gap_past_the_end_is_forbidden_not_permitted() {
        // The safe direction. If this answered `true`, an off-by-one in a caller would
        // become an extra merge nobody asked for rather than a visible panic or refusal.
        let b = Boundaries::Mask(vec![true, true]);
        assert!(!b.permits(2));
        assert!(!b.permits(usize::MAX));
    }
}
