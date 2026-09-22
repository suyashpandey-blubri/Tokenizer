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

use crate::text::boundary::Boundaries;

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
fn initial_vocab() -> Vec<Vec<u8>> {
    (0..BYTE_ALPHABET).map(|i| vec![i as u8]).collect()
}

/// A trained (or untrained) byte-pair vocabulary: the merge list and the table it derives.
///
/// WHY THIS IS A TYPE AND NOT TWO FIELDS ON `Tokenizer`
/// ----------------------------------------------------
/// The two fields are not independent. `vocab` is *derived* from `merges` — so completely
/// that the `.model` file does not store it at all, and `load` rebuilds it by replaying the
/// merge list. Any code that pushed to `vocab` without inserting into `merges` would break
/// that silently, and every `.model` written afterwards would be wrong.
///
/// Before 12.3a that invariant was enforced by nothing. `lib.rs`, `trainer.rs` and
/// `serialize.rs` could all reach both fields, and two of them carried their own hand-written
/// copy of "mint an id, concatenate the two halves, push" — `train` deriving the id from its
/// loop counter, `load` from its line index. Two independent statements of one rule, which
/// is the same shape as the duplication 12.0 removed one level down.
///
/// Now the fields are private to this module and [`push_merge`](BpeModel::push_merge) is the
/// only way to add anything, so the id cannot be derived wrongly and the two tables cannot
/// drift apart.
///
/// `PartialEq` because several tests ask "did `load` reproduce what `train` built?" — which
/// is the invariant stated as an assertion.
///
/// It is `pub(crate)`. Promoting it to public API would be the thing that finally lets a
/// caller hold a trained model that provably cannot be retrained underneath them; that is a
/// step-13 decision, and **12.3 does not deliver it** — from outside the crate nothing has
/// changed.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct BpeModel {
    /// (left_id, right_id) -> new_id. Ids are handed out 256, 257, ... in merge order, so
    /// the id doubles as the merge rank — no separate ranks table needed.
    merges: HashMap<(u32, u32), u32>,
    /// vocab[id] = the byte string that id expands to. `Vec<u8>` is Python's `bytes`; a
    /// token need not be valid UTF-8, because a merge can join half of a multi-byte
    /// character to something else. Ids are dense 0..vocab_size, so a Vec indexed by id
    /// beats a HashMap.
    vocab: Vec<Vec<u8>>,
}

impl BpeModel {
    /// The starting state: one token per possible byte, nothing learned.
    ///
    /// Note *untrained*, not *empty* — it already holds 256 tokens, which is exactly why
    /// byte-level BPE has no out-of-vocabulary case. Both `Tokenizer::with_config` and
    /// `load` begin here, and before 12.3a each assembled it from two halves by hand.
    pub(crate) fn untrained() -> Self {
        BpeModel {
            merges: HashMap::new(),
            vocab: initial_vocab(),
        }
    }

    /// How many tokens this model knows: 256 byte tokens plus one per learned merge.
    ///
    /// A method rather than a public field because `.len()` needs `vocab` in scope, and
    /// anything that can see `vocab` can also call `vocab.push` — which is the one thing
    /// this type exists to prevent.
    pub(crate) fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// How many merges were learned. Equals `vocab_size() - BYTE_ALPHABET`, by construction.
    pub(crate) fn merge_count(&self) -> usize {
        self.merges.len()
    }

    /// The byte string an id expands to, or `None` if the id is out of range.
    ///
    /// `Option` rather than an index because `decode` takes ids from a file or a pipe, not
    /// only from our own `encode`. Indexing directly was an out-of-bounds panic until step 8.
    pub(crate) fn token(&self, id: u32) -> Option<&[u8]> {
        self.vocab.get(id as usize).map(Vec::as_slice)
    }

    /// Every token in id order, for `.vocab` rendering.
    pub(crate) fn tokens(&self) -> impl ExactSizeIterator<Item = &[u8]> {
        self.vocab.iter().map(Vec::as_slice)
    }

    /// The id a pair merges into, or `None` if this pair was never learned.
    ///
    /// This is `encode`'s hot lookup, and the id it returns *is* the merge rank — which is
    /// what lets `encode_chunk` pick "lowest id" to mean "earliest merge".
    pub(crate) fn rank(&self, pair: (u32, u32)) -> Option<u32> {
        self.merges.get(&pair).copied()
    }

    /// The merge list in the order training learned it, as `(new_id, (p0, p1))`.
    ///
    /// `merges` is a `HashMap` and has no order of its own. Sorting by the minted id
    /// recovers it — and the order is not cosmetic: the `.model` file stores each new id
    /// **implicitly, as the line index**, so writing the lines in a different order does not
    /// produce a slightly different model, it produces a differently *trained* one, because
    /// every later merge would then name different tokens.
    pub(crate) fn merges_in_learned_order(&self) -> Vec<(u32, (u32, u32))> {
        let mut by_id: Vec<(u32, (u32, u32))> =
            self.merges.iter().map(|(&pair, &idx)| (idx, pair)).collect();
        by_id.sort_unstable();
        by_id
    }

    /// Learn one merge: mint the next id, derive its byte string, record both. Returns the
    /// new id.
    ///
    /// **The only mutation this type allows**, and it deliberately does not let the caller
    /// choose the id. `train` used to compute it from its loop counter and `load` from its
    /// line index; here it comes from `merges.len()`, which cannot disagree with itself.
    /// Building the byte string in the same call is what keeps `vocab` derived from
    /// `merges` rather than merely consistent with it by habit.
    ///
    /// Because ids are appended as they are minted, they stay dense and in learned order by
    /// construction — no separate vocab-building pass and nothing to sort.
    ///
    /// The next id is `vocab.len()`, **not** `BYTE_ALPHABET + merges.len()`. The two agree
    /// on anything `train` produces, and they diverge on a corrupt `.model` that names the
    /// same pair twice: `merges.insert` would overwrite and the count would not grow, so the
    /// following merge would reuse an id that `vocab` had already handed out. `vocab.len()`
    /// is the table's actual size and cannot drift from itself, which also matches what
    /// `train` and `load` each computed by hand before 12.3a.
    pub(crate) fn push_merge(&mut self, pair: (u32, u32)) -> u32 {
        let idx = self.vocab.len() as u32;
        self.merges.insert(pair, idx);
        let mut bytes = self.vocab[pair.0 as usize].clone();
        bytes.extend_from_slice(&self.vocab[pair.1 as usize]);
        self.vocab.push(bytes);
        idx
    }
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
/// `bounds` matters here for the same reason it matters in `pairs_of`: this is how
/// `encode_chunk` DISCOVERS candidate pairs, so a pair that only occurs at forbidden gaps
/// must never become a candidate. Missing it would let encode apply a merge that training
/// could not have learned at that position.
pub(crate) fn get_stats_into(
    counts: &mut HashMap<(u32, u32), u64>,
    ids: &[u32],
    weight: u64,
    bounds: &Boundaries,
) {
    // The match is hoisted OUT of the loop on purpose. Asking `bounds.permits(gap)` per gap
    // cost a measured +6.6% on `encode/article` at 12.4b, because `encode_chunk` rebuilds
    // these stats on every merge iteration — so the check landed in an O(len^2) inner loop.
    // Specialising the two cases costs one duplicated three-line body and gives B0 a loop
    // with no boundary work in it at all.
    match bounds {
        Boundaries::Unrestricted => {
            for window in ids.windows(2) {
                let pair = (window[0], window[1]);
                *counts.entry(pair).or_insert(0) += weight;
            }
        }
        // Window index == gap index; see `merge`.
        Boundaries::Mask(mask) => {
            for (gap, window) in ids.windows(2).enumerate() {
                if !mask.get(gap).copied().unwrap_or(false) {
                    continue;
                }
                let pair = (window[0], window[1]);
                *counts.entry(pair).or_insert(0) += weight;
            }
        }
    }
}

/// After getting the highest pair we mint a new token for it and replace every
/// non-overlapping occurrence, scanning left to right.
/// `i + 1 < ids.len()` is the `i < len(ids)-1` guard: the last element can't start a
/// pair. It's written as an addition because `len()` is unsigned, so `len() - 1` would
/// underflow on an empty input. The outer loop still runs to the final index, because
/// that last element still has to be copied to the output.
/// `bounds` gates each occurrence: an occurrence starting at `i` crosses gap `i`, so it is
/// only replaced if that gap is permitted. Returns the rewritten ids **and the gap mask that
/// belongs to them** — merging consumes a gap and renumbers every gap to its right, so the
/// two have to be produced together or they fall out of step immediately.
///
/// The mask arithmetic is one expression in both branches. After a step, `i` has already
/// advanced past whatever was emitted, so the gap that follows the emitted element in the
/// *output* is input gap `i - 1`:
///
/// ```text
///   copy  ids[i]            i += 1   ->  following gap is input gap i-1  (== old i)
///   merge ids[i], ids[i+1]  i += 2   ->  following gap is input gap i-1  (== old i+1)
/// ```
///
/// and the final element has no gap after it, which `i < ids.len()` takes care of.
pub(crate) fn merge(
    ids: &[u32],
    pair: (u32, u32),
    idx: u32,
    bounds: &Boundaries,
) -> (Vec<u32>, Boundaries) {
    let mut newids = Vec::with_capacity(ids.len());

    // Two loops rather than one with a check inside it, for the reason spelled out on
    // `get_stats_into`: this is called once per merge iteration inside `encode_chunk`, so a
    // per-position enum match is in the hottest path the crate has. The B0 arm below is
    // byte-for-byte the pre-12.4 loop.
    let Boundaries::Mask(mask) = bounds else {
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
        return (newids, Boundaries::Unrestricted);
    };

    let permits = |gap: usize| mask.get(gap).copied().unwrap_or(false);
    let mut newmask = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 && permits(i) {
            newids.push(idx);
            i += 2;
        } else {
            newids.push(ids[i]);
            i += 1;
        }
        if i < ids.len() {
            newmask.push(permits(i - 1));
        }
    }
    (newids, Boundaries::Mask(newmask))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- merge ------------------------------------------------------------------
    // The function whose behaviour prompted the most questions.

    #[test]
    fn merge_replaces_non_overlapping_pairs() {
        // The case traced by hand in a scratch program and then thrown away.
        assert_eq!(merge(&[97, 97, 97, 98], (97, 97), 256, &Boundaries::Unrestricted).0, vec![256, 97, 98]);
    }

    #[test]
    fn merge_does_not_skip_the_first_element() {
        // The `i + 1` question: does the guard make the scan start at index 1? No —
        // `i` starts at 0 and `ids[i]` is always examined. If it did skip, this would
        // come back as [120, 121, 5, 5].
        assert_eq!(merge(&[120, 121, 5, 5], (120, 121), 300, &Boundaries::Unrestricted).0, vec![300, 5, 5]);
    }

    #[test]
    fn merge_copies_the_last_element() {
        // The last element can't start a pair, but it still has to reach the output.
        assert_eq!(merge(&[1, 2, 3], (1, 2), 256, &Boundaries::Unrestricted).0, vec![256, 3]);
    }

    #[test]
    fn merge_on_empty_input_does_not_underflow() {
        // This is why the guard is written `i + 1 < ids.len()` and not `i < ids.len() - 1`:
        // `len()` is unsigned, so `0 - 1` would panic in debug and wrap in release.
        assert_eq!(merge(&[], (1, 2), 256, &Boundaries::Unrestricted).0, Vec::<u32>::new());
    }

    #[test]
    fn merge_on_single_element_is_identity() {
        assert_eq!(merge(&[42], (42, 42), 256, &Boundaries::Unrestricted).0, vec![42]);
    }

    #[test]
    fn merge_with_no_match_is_identity() {
        assert_eq!(merge(&[1, 2, 3], (9, 9), 256, &Boundaries::Unrestricted).0, vec![1, 2, 3]);
    }

    #[test]
    fn merge_of_an_overlapping_run_takes_pairs_left_to_right() {
        // [97,97,97] contains (97,97) at index 0 AND index 1, but they overlap. Scanning
        // left to right takes the first and leaves the leftover 97 alone.
        assert_eq!(merge(&[97, 97, 97], (97, 97), 256, &Boundaries::Unrestricted).0, vec![256, 97]);
        assert_eq!(merge(&[97, 97, 97, 97], (97, 97), 256, &Boundaries::Unrestricted).0, vec![256, 256]);
    }

    /// LIVE-SEAM TEST, applying side: a forbidden gap must block the merge.
    ///
    /// Everything production does passes `Unrestricted`, so without this the boundary
    /// argument to `merge` would compile and be entirely decorative.
    #[test]
    fn merge_refuses_a_forbidden_gap() {
        //   ids:   [97][97][97][97]      merging (97,97)
        //   gaps:     0   1   2
        let ids = [97, 97, 97, 97];

        // Unrestricted: left-to-right, non-overlapping, so gaps 0 and 2 are taken.
        assert_eq!(
            merge(&ids, (97, 97), 256, &Boundaries::Unrestricted).0,
            vec![256, 256]
        );

        // Forbid gap 0. The scan then finds the occurrence at gap 1 instead — which is a
        // DIFFERENT merge, not merely a missing one, and is exactly why the trainer must
        // not count occurrences it cannot apply.
        assert_eq!(
            merge(&ids, (97, 97), 256, &Boundaries::Mask(vec![false, true, true])).0,
            vec![97, 256, 97]
        );

        // Forbid everything: the chunk comes back untouched.
        assert_eq!(
            merge(&ids, (97, 97), 256, &Boundaries::Mask(vec![false; 3])).0,
            ids.to_vec()
        );
    }

    /// LIVE-SEAM TEST, and the subtlest claim in 12.4b: **a merge renumbers the gaps to its
    /// right**, so the mask that comes back must be shifted, not merely shortened.
    ///
    /// If this were wrong the failure would be quiet and cumulative — every subsequent merge
    /// in the same chunk would consult the wrong gap, so boundaries would drift further out
    /// of alignment the longer training ran.
    #[test]
    fn merge_shifts_the_mask_left_of_what_it_consumed() {
        //   ids:   [ a ][ b ][ c ][ d ]        merging (b, c) consumes gap 1
        //   gaps:     0    1    2
        //   out:   [ a ][ bc ][ d ]
        //   gaps:     0'    1'                 0' == old 0,  1' == old 2
        let ids = [1, 2, 3, 4];

        // Distinguishable values, so a shift is visible and a truncation is not enough:
        // old gap 2 is `false`, and it must land at new gap 1.
        let (out, mask) = merge(
            &ids,
            (2, 3),
            256,
            &Boundaries::Mask(vec![true, true, false]),
        );
        assert_eq!(out, vec![1, 256, 4]);
        assert_eq!(
            mask,
            Boundaries::Mask(vec![true, false]),
            "new gap 1 must carry OLD gap 2's flag, not old gap 1's"
        );

        // A chunk that merges down to one token has no gaps left.
        let (out, mask) = merge(&[1, 2], (1, 2), 256, &Boundaries::Mask(vec![true]));
        assert_eq!(out, vec![256]);
        assert_eq!(mask, Boundaries::Mask(vec![]));

        // Unrestricted in, unrestricted out — and no allocation on the way through.
        let (_, mask) = merge(&ids, (2, 3), 256, &Boundaries::Unrestricted);
        assert_eq!(mask, Boundaries::Unrestricted);
    }

    /// The encode-side counterpart of `a_forbidden_gap_is_not_counted`: `get_stats_into` is
    /// how `encode_chunk` DISCOVERS candidates, so a forbidden gap must not produce one.
    #[test]
    fn get_stats_skips_forbidden_gaps() {
        let mut counts = HashMap::new();
        get_stats_into(&mut counts, &[97, 97, 97, 97], 1, &Boundaries::Unrestricted);
        assert_eq!(counts.get(&(97, 97)), Some(&3));

        let mut counts = HashMap::new();
        get_stats_into(
            &mut counts,
            &[97, 97, 97, 97],
            1,
            &Boundaries::Mask(vec![true, false, true]),
        );
        assert_eq!(counts.get(&(97, 97)), Some(&2));
    }

    // -- get_stats_into ---------------------------------------------------------

    #[test]
    fn get_stats_counts_adjacent_pairs() {
        let mut counts = HashMap::new();
        get_stats_into(&mut counts, &[1, 2, 3], 1, &Boundaries::Unrestricted);
        assert_eq!(counts.get(&(1, 2)), Some(&1));
        assert_eq!(counts.get(&(2, 3)), Some(&1));
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn get_stats_counts_repeats() {
        let mut counts = HashMap::new();
        get_stats_into(&mut counts, &[1, 1, 1], 1, &Boundaries::Unrestricted);
        // Unlike `merge`, counting does NOT skip overlaps: (1,1) occurs at index 0 and 1.
        assert_eq!(counts.get(&(1, 1)), Some(&2));
    }

    #[test]
    fn get_stats_accumulates_across_calls() {
        // The whole reason this takes `&mut` instead of returning a fresh map: with a
        // pre-tokenizer, counts must be tallied across every chunk before picking a winner.
        let mut counts = HashMap::new();
        get_stats_into(&mut counts, &[1, 2], 1, &Boundaries::Unrestricted);
        get_stats_into(&mut counts, &[1, 2], 1, &Boundaries::Unrestricted);
        get_stats_into(&mut counts, &[2, 3], 1, &Boundaries::Unrestricted);
        assert_eq!(counts.get(&(1, 2)), Some(&2));
        assert_eq!(counts.get(&(2, 3)), Some(&1));
    }

    #[test]
    fn get_stats_weights_each_chunk_by_its_occurrence_count() {
        // The heart of step 9: one weighted addition must equal N unweighted ones.
        let mut weighted = HashMap::new();
        get_stats_into(&mut weighted, &[1, 2, 3], 30_000_000, &Boundaries::Unrestricted);

        let mut repeated = HashMap::new();
        for _ in 0..1000 {
            get_stats_into(&mut repeated, &[1, 2, 3], 30_000, &Boundaries::Unrestricted);
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
        get_stats_into(&mut counts, &[1, 2], u32::MAX as u64 + 1000, &Boundaries::Unrestricted);
        assert_eq!(counts.get(&(1, 2)), Some(&(u32::MAX as u64 + 1000)));
    }

    #[test]
    fn get_stats_on_too_short_input_yields_nothing() {
        let mut counts = HashMap::new();
        get_stats_into(&mut counts, &[], 1, &Boundaries::Unrestricted);
        get_stats_into(&mut counts, &[7], 1, &Boundaries::Unrestricted);
        assert!(counts.is_empty(), "a slice shorter than 2 has no pairs");
    }

    // -- PairIndex: the step-10 bookkeeping --------------------------------------
}
