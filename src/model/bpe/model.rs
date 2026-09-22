//! The trained artifact's machinery: the id space, and the two operations that work on a
//! sequence of ids.
//!
//! WHAT IS HERE AND WHY
//! --------------------
//! Split by *who calls it*, not by what the code is:
//!
//! - `merge` is **shared** — `train` applies a learned merge, `encode_chunk` replays one.
//! - `get_stats_into` is **encode-only**. Training counts pairs through `PairIndex`
//!   instead, incrementally; this is the O(len²) rebuild that step 19 targets.
//! - `BYTE_ALPHABET` / `initial_vocab` are the id space itself.
//!
//! `pairs_of` is NOT here despite looking like a sibling of `get_stats_into`: measured at
//! 12.2, its only three call sites are all inside `impl PairIndex`, so it lives there.
//!
//! At 12.3 these become inherent methods on `BpeModel`. They are free functions today
//! because that is what they were in `lib.rs`, and 12.2 moves code without changing it.

use std::collections::HashMap;

/// The byte alphabet: one token per possible byte value, minted before training starts.
/// This is *the* reason byte-level BPE has no out-of-vocabulary case — every input byte
/// already has an id, so there is nothing to fall back to.
///
/// It is a named constant because the number is load-bearing in four places across two
/// functions written independently of each other: `with_config` and `load` both build the
/// initial table, and `train` and `load` both mint ids as `BYTE_ALPHABET + i`. `load`
/// carries its own copy of arithmetic `train` also performs — the two must agree forever,
/// and until now nothing in the code said so.
pub(crate) const BYTE_ALPHABET: u32 = 256;

/// The untrained vocabulary: `vocab[i] == [i]` for every byte value.
///
/// Extracted because `with_config` and `load` held two verbatim copies of it. `load` is the
/// one that matters: it must reconstruct *exactly* the state `train` started from, or
/// replaying the merge list rebuilds a different table than the one that was saved.
pub(crate) fn initial_vocab() -> Vec<Vec<u8>> {
    (0..BYTE_ALPHABET).map(|i| vec![i as u8]).collect()
}

/// Counting all number of occurring pairs, accumulating into a caller-supplied map.
/// Python: `for pair in zip(ids, ids[1:]): counts[pair] = counts.get(pair, 0) + 1`
/// Takes `&mut` rather than returning a fresh map because with a pre-tokenizer the counts
/// have to be tallied across every chunk before picking a winner — the direct analogue of
/// minbpe's optional `counts` parameter.
///
/// `weight` is how many times this chunk occurs in the corpus. Before step 9 every chunk
/// was stored once per occurrence and this was effectively always 1; now identical chunks
/// are collapsed into one entry carrying a count, so `" the"` contributes its 30 million
/// occurrences with a single addition instead of 30 million of them. The resulting counts
/// are *identical* either way — that is what makes the optimisation exact rather than an
/// approximation.
///
/// The counter is `u64` because a weighted total can exceed `u32`: 1 GB of English holds
/// on the order of 10^9 pairs, and one common pair can account for a large share of them.
pub(crate) fn get_stats_into(counts: &mut HashMap<(u32, u32), u64>, ids: &[u32], weight: u64) {
    for window in ids.windows(2) {
        let pair = (window[0], window[1]);
        *counts.entry(pair).or_insert(0) += weight;
    }
}

/// After getting the highest pair we mint a new token for it and replace every
/// non-overlapping occurrence, scanning left to right.
/// `i + 1 < ids.len()` is the `i < len(ids)-1` guard: the last element can't start a
/// pair. It's written as an addition because `len()` is unsigned, so `len() - 1` would
/// underflow on an empty input. The outer loop still runs to the final index, because
/// that last element still has to be copied to the output.
pub(crate) fn merge(ids: &[u32], pair: (u32, u32), idx: u32) -> Vec<u32> {
    let mut newids = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            newids.push(idx);
            i += 2;
        } else {
            newids.push(ids[i]);
            i += 1;
        }
    }
    newids
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- merge ------------------------------------------------------------------
    // The function whose behaviour prompted the most questions.

    #[test]
    fn merge_replaces_non_overlapping_pairs() {
        // The case traced by hand in a scratch program and then thrown away.
        assert_eq!(merge(&[97, 97, 97, 98], (97, 97), 256), vec![256, 97, 98]);
    }

    #[test]
    fn merge_does_not_skip_the_first_element() {
        // The `i + 1` question: does the guard make the scan start at index 1? No —
        // `i` starts at 0 and `ids[i]` is always examined. If it did skip, this would
        // come back as [120, 121, 5, 5].
        assert_eq!(merge(&[120, 121, 5, 5], (120, 121), 300), vec![300, 5, 5]);
    }

    #[test]
    fn merge_copies_the_last_element() {
        // The last element can't start a pair, but it still has to reach the output.
        assert_eq!(merge(&[1, 2, 3], (1, 2), 256), vec![256, 3]);
    }

    #[test]
    fn merge_on_empty_input_does_not_underflow() {
        // This is why the guard is written `i + 1 < ids.len()` and not `i < ids.len() - 1`:
        // `len()` is unsigned, so `0 - 1` would panic in debug and wrap in release.
        assert_eq!(merge(&[], (1, 2), 256), Vec::<u32>::new());
    }

    #[test]
    fn merge_on_single_element_is_identity() {
        assert_eq!(merge(&[42], (42, 42), 256), vec![42]);
    }

    #[test]
    fn merge_with_no_match_is_identity() {
        assert_eq!(merge(&[1, 2, 3], (9, 9), 256), vec![1, 2, 3]);
    }

    #[test]
    fn merge_of_an_overlapping_run_takes_pairs_left_to_right() {
        // [97,97,97] contains (97,97) at index 0 AND index 1, but they overlap. Scanning
        // left to right takes the first and leaves the leftover 97 alone.
        assert_eq!(merge(&[97, 97, 97], (97, 97), 256), vec![256, 97]);
        assert_eq!(merge(&[97, 97, 97, 97], (97, 97), 256), vec![256, 256]);
    }

    // -- get_stats_into ---------------------------------------------------------

    #[test]
    fn get_stats_counts_adjacent_pairs() {
        let mut counts = HashMap::new();
        get_stats_into(&mut counts, &[1, 2, 3], 1);
        assert_eq!(counts.get(&(1, 2)), Some(&1));
        assert_eq!(counts.get(&(2, 3)), Some(&1));
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn get_stats_counts_repeats() {
        let mut counts = HashMap::new();
        get_stats_into(&mut counts, &[1, 1, 1], 1);
        // Unlike `merge`, counting does NOT skip overlaps: (1,1) occurs at index 0 and 1.
        assert_eq!(counts.get(&(1, 1)), Some(&2));
    }

    #[test]
    fn get_stats_accumulates_across_calls() {
        // The whole reason this takes `&mut` instead of returning a fresh map: with a
        // pre-tokenizer, counts must be tallied across every chunk before picking a winner.
        let mut counts = HashMap::new();
        get_stats_into(&mut counts, &[1, 2], 1);
        get_stats_into(&mut counts, &[1, 2], 1);
        get_stats_into(&mut counts, &[2, 3], 1);
        assert_eq!(counts.get(&(1, 2)), Some(&2));
        assert_eq!(counts.get(&(2, 3)), Some(&1));
    }

    #[test]
    fn get_stats_weights_each_chunk_by_its_occurrence_count() {
        // The heart of step 9: one weighted addition must equal N unweighted ones.
        let mut weighted = HashMap::new();
        get_stats_into(&mut weighted, &[1, 2, 3], 30_000_000);

        let mut repeated = HashMap::new();
        for _ in 0..1000 {
            get_stats_into(&mut repeated, &[1, 2, 3], 30_000);
        }
        assert_eq!(weighted, repeated);
        assert_eq!(weighted.get(&(1, 2)), Some(&30_000_000));
    }

    #[test]
    fn get_stats_counts_exceed_u32() {
        // Why the counter widened from u32 to u64. u32::MAX is about 4.3 billion; a single
        // common pair in a large corpus can pass that, and wrapping would silently pick the
        // wrong merge.
        let mut counts = HashMap::new();
        get_stats_into(&mut counts, &[1, 2], u32::MAX as u64 + 1000);
        assert_eq!(counts.get(&(1, 2)), Some(&(u32::MAX as u64 + 1000)));
    }

    #[test]
    fn get_stats_on_too_short_input_yields_nothing() {
        let mut counts = HashMap::new();
        get_stats_into(&mut counts, &[], 1);
        get_stats_into(&mut counts, &[7], 1);
        assert!(counts.is_empty(), "a slice shorter than 2 has no pairs");
    }

    // -- PairIndex: the step-10 bookkeeping --------------------------------------
}
