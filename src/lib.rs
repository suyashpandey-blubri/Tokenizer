//! A naive byte-level BPE tokenizer.
//!
//! This is the **library crate**. It has no `main`; it compiles to something other code
//! links against. `src/main.rs` is a separate binary crate that consumes it.
//!
//! `pub` is the line between API and internals, and this is the first point in the project
//! where that line exists at all. What is public:
//!
//! - [`Tokenizer`] and its `new` / `with_config` / `train` / `encode` / `encode_ordinary` /
//!   `decode` / `split` / `save` / `load` / `register_special_tokens`
//! - [`Normalizer`] and its `normalize`, and [`AllowedSpecial`]
//! - [`GPT4_SPLIT_PATTERN`] and [`GPT2_SPLIT_PATTERN`]
//!
//! What is deliberately private: `merge`, `get_stats_into`, `encode_chunk` and `show` are
//! how training happens, not what a caller does; and `Tokenizer`'s fields stay private so
//! `merges` and `vocab` cannot drift out of sync behind our back.
//!
//! Tests live in two places for a reason. The `#[cfg(test)] mod tests` at the bottom of
//! this file is a *unit* test module: same file, so it can reach the private items above.
//! `tests/api.rs` is an *integration* test: a separate crate that sees only `pub` items,
//! which is what makes it able to catch a forgotten `pub` — something a unit test in here
//! structurally cannot notice.

use std::collections::HashMap;
use std::fmt;

mod model;
mod special_tokens;
mod text;

// Moved into `model/bpe/` at 12.2a. They need `pub(crate)` there and a `use` here for the
// same reason: a private item is visible in its own module and all DESCENDANTS of it, so
// `lib.rs` — the parent — cannot see into a child.
//
// The reverse direction is why 12.2b could move `train`, `save` and `load` out of this file
// with NO visibility change at all: they are `impl Tokenizer` blocks, the fields are
// declared here, and `model/bpe/` is a descendant. What is left in this file after that is
// the type itself, the encode and decode paths, and the public special-token methods — whose
// state has lived in `special_tokens.rs` since 12.8b.
use model::bpe::model::{BpeModel, get_stats_into, merge};
use text::boundary::Boundaries;

// `pub use` is a `use` plus "and let callers outside the crate see it too". These three
// are the items `main.rs` and `tests/api.rs` import by name, so re-exporting them here is
// what keeps `naive_bbpe::Normalizer` resolving after the move — and it simultaneously
// brings the short names into scope for this file, which uses all three itself.
pub use text::normalizer::Normalizer;
pub use text::pretokenizer::{GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN};

// Plain `use`, because `PreTokenizer` is `pub(crate)`: only this file needs to name it,
// as the type of a private field. `pub use` on it would not merely be redundant, it would
// not compile — a `pub(crate)` item cannot be re-exported publicly.
use text::pretokenizer::PreTokenizer;

// Split the same two ways as above: `AllowedSpecial` (12.8a) is public API that `main.rs` and
// `tests/api.rs` import by name, so it is re-exported; `SpecialTokens` (12.8b) is
// `pub(crate)`, the type of a private field.
pub use special_tokens::AllowedSpecial;
use special_tokens::SpecialTokens;


// ---------------------------------------------------------------------------
// The tokenizer
// ---------------------------------------------------------------------------

/// A trained byte-level BPE tokenizer: the learned merges, the text pipeline in front of them
/// (normalizer and pre-tokenizer), and any special tokens. `train` builds the merges,
/// `encode` and `decode` use them, and `save` and `load` persist all of it.
pub struct Tokenizer {
    /// What training produced: the merge list and the table derived from it. One field
    /// rather than two since 12.3a, because the two were never independent — see
    /// [`BpeModel`].
    model: BpeModel,
    /// Applied to the text before encoding. Defaults to `None`, matching GPT-4.
    normalizer: Normalizer,
    /// The pre-tokenizer, compiled once at construction rather than per call.
    pattern: PreTokenizer,

    /// Special tokens, both directions plus the regex that finds them. One field rather than
    /// three since 12.8b, for the same reason as `model` — see [`SpecialTokens`].
    specials: SpecialTokens,
}

/// Deriving `Debug` would print every one of `vocab`'s entries — 2000 of them after a
/// modest training run, and 50000 after a real one. This prints the shape instead, which
/// is what anyone looking at a `Tokenizer` in an error message actually wants.
impl fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.model.vocab_size())
            .field("merges", &self.model.merge_count())
            .field("normalizer", &self.normalizer.name())
            .field("pattern", &self.pattern.as_str())
            .finish()
    }
}

/// Required now that `new` is public: Rust convention is that a type with a no-argument
/// constructor also implements `Default`, so generic code written against `Default` can
/// build one. This became a clippy warning the moment `new` gained its `pub` — a small,
/// concrete example of what going public actually obliges you to provide.
impl Default for Tokenizer {
    fn default() -> Self {
        Tokenizer::new()
    }
}

impl Tokenizer {
    /// The untrained state: nothing learned, and one token per possible byte.
    /// Every byte already has a token before training starts, which is why byte-level
    /// BPE has no out-of-vocabulary case.
    pub fn new() -> Self {
        Tokenizer::with_config(GPT4_SPLIT_PATTERN, Normalizer::None)
    }

    /// How many tokens this tokenizer knows: 256 byte tokens plus one per learned merge.
    /// `vocab` itself stays private — handing out the table would let a caller desynchronise
    /// it from `merges` — but the *size* is something a caller legitimately needs, e.g. to
    /// check an id before passing it to [`decode`](Tokenizer::decode).
    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    pub fn with_config(pattern: &str, normalizer: Normalizer) -> Self {
        Tokenizer {
            model: BpeModel::untrained(),
            normalizer,
            pattern: PreTokenizer::new(pattern).expect("split pattern failed to compile"),
            specials: SpecialTokens::none(),
        }
    }

    /// Register special tokens, replacing any already registered (minbpe's semantics).
    ///
    /// Four things are validated, each because it would otherwise fail somewhere far away
    /// from the mistake:
    ///
    /// - **id must be >= `vocab_size()`** — otherwise the special would shadow a token BPE
    ///   learned, and `decode` would hand back the wrong one with no error anywhere.
    /// - **no newlines in the token** — the `.model` format is one special per line, so a
    ///   newline would corrupt the file on save and misparse on load.
    /// - **non-empty** — an empty alternation branch matches at every position.
    /// - **ids unique** — two specials sharing an id makes `decode` ambiguous.
    pub fn register_special_tokens(&mut self, specials: &[(String, u32)]) -> Result<(), String> {
        // `?` returns before the assignment, so a rejected registration leaves the previous
        // specials in place — as it did when the three fields were assigned one by one.
        self.specials = SpecialTokens::new(specials, self.model.vocab_size() as u32)?;
        Ok(())
    }

    /// The registered special tokens, lowest id first.
    pub fn special_tokens(&self) -> Vec<(String, u32)> {
        self.specials.list()
    }

    /// The pre-tokenizer stage: cut the text into chunks that BPE is not allowed to
    /// merge across.
    pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        self.pattern.split(text)
    }


    /// Encode text with **no regard for special tokens** — `<|endoftext|>` comes out as
    /// those 13 ordinary characters. This is the whole encode pipeline as it existed before
    /// step 8, and it is minbpe's `encode_ordinary`.
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        let text = self.normalizer.normalize(text);
        self.split(&text)
            .into_iter()
            .flat_map(|chunk| self.encode_chunk(chunk))
            .collect()
    }

    /// Encode text, handling special tokens according to `allowed`. minbpe's `encode`.
    ///
    /// Specials are found and cut out **before** normalization and pre-tokenization, not
    /// after. A special token is a literal marker rather than prose, so it should not be
    /// NFC-folded, and it must not be handed to the split regex — which would happily chop
    /// `<|endoftext|>` into `<|`, `endoftext`, `|>` and merge the pieces into ordinary
    /// vocabulary.
    ///
    /// Returns `Err` only under [`AllowedSpecial::NoneRaise`], and only when a special
    /// actually appears in the text.
    pub fn encode(&self, text: &str, allowed: AllowedSpecial) -> Result<Vec<u32>, String> {
        // No specials registered, or none permitted: the fast path is also the whole path.
        let Some(pattern) = self.specials.pattern() else {
            return Ok(self.encode_ordinary(text));
        };
        match allowed {
            AllowedSpecial::None => return Ok(self.encode_ordinary(text)),
            AllowedSpecial::NoneRaise => {
                if let Ok(Some(m)) = pattern.find(text) {
                    return Err(format!(
                        "special token {:?} appears in the input at byte {}, but specials are \
                         not allowed (use AllowedSpecial::All to encode it as a token, or \
                         AllowedSpecial::None to encode it as ordinary text)",
                        m.as_str(),
                        m.start()
                    ));
                }
                return Ok(self.encode_ordinary(text));
            }
            AllowedSpecial::All => {}
        }

        // Walk the matches and slice the text between them. Python's `re.split` with a
        // capture group keeps the delimiters; Rust's `Regex::split` drops them, so doing it
        // by hand is both necessary and clearer about what is kept.
        let mut ids = Vec::new();
        let mut cursor = 0;
        for found in pattern.find_iter(text) {
            let m = found.map_err(|e| format!("special-token scan failed: {e}"))?;
            if m.start() > cursor {
                ids.extend(self.encode_ordinary(&text[cursor..m.start()]));
            }
            // `find_iter` only yields matches of the alternation, so this is always a hit.
            ids.push(self.specials.id(m.as_str()));
            cursor = m.end();
        }
        if cursor < text.len() {
            ids.extend(self.encode_ordinary(&text[cursor..]));
        }
        Ok(ids)
    }

    /// BPE over a single pre-tokenized chunk. This is the whole of the old `encode`.
    fn encode_chunk(&self, text: &str) -> Vec<u32> {
        let mut tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        // The encode-side twin of the line in `Tokenizer::train`: this is where text
        // becomes ids, so it is the only place a mask for this chunk can be derived.
        // BOTH sides must agree — boundaries are positional, so if training respects a
        // boundary and encoding does not, encoding emits tokens training could never have
        // learned at that position. B0 forbids nothing; 15.4 replaces this line too.
        let mut bounds = Boundaries::Unrestricted;

        // Running this loop till merging is possible.
        while tokens.len() >= 2 {
            // Weight 1: encoding looks at one piece of text, so there is nothing to
            // deduplicate and the counts are only used as a set of "which pairs exist".
            let mut stats: HashMap<(u32, u32), u64> = HashMap::new();
            get_stats_into(&mut stats, &tokens, 1, &bounds);

            // Python: `min(stats, key=lambda p: merges.get(p, float('inf')))`, then
            // `if pair not in merges: break`. Here we filter to the pairs that are in
            // merges and take the smallest id — ids are handed out in merge order, so
            // the lowest id is the earliest-learned merge. `None` is the `break` case.
            let best = stats
                .keys()
                .filter_map(|&pair| self.model.rank(pair).map(|idx| (idx, pair)))
                .min();

            match best {
                Some((idx, pair)) => {
                    let (t, b) = merge(&tokens, pair, idx, &bounds);
                    tokens = t;
                    bounds = b;
                }
                None => break, // we have come to a point where no more merges can be made
            }
        }
        tokens
    }

    // -- persistence ------------------------------------------------------------


    /// Python: `b"".join(vocab[idx] for idx in ids).decode('utf-8', errors='replace')`
    ///
    /// Lookup order is vocab, then specials, then error. Special ids live *above*
    /// `vocab.len()`, so the two ranges never overlap and the order is a matter of which is
    /// commoner rather than which wins.
    ///
    /// This returned a bare `String` before step 8 and indexed `vocab` directly, so an id
    /// from outside the table was an index-out-of-bounds panic. That was survivable while
    /// ids could only come from our own `encode`; with special tokens they can also come
    /// from a file or a pipe, so an unknown id is now an ordinary error. The bounds check
    /// `run_decode` was doing by hand moves in here, where it belongs.
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let mut tokens: Vec<u8> = Vec::new();
        for &idx in ids {
            if let Some(bytes) = self.model.token(idx) {
                tokens.extend_from_slice(bytes);
            } else if let Some(special) = self.specials.token(idx) {
                tokens.extend_from_slice(special.as_bytes());
            } else {
                return Err(format!(
                    "unknown token id {idx}: this model has {} vocabulary entries and {} \
                     special tokens",
                    self.model.vocab_size(),
                    self.specials.len()
                ));
            }
        }
        // Lossy == errors='replace': a slice of ids can end mid-character.
        Ok(String::from_utf8_lossy(&tokens).into_owned())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// `#[cfg(test)]` means "only compile this when running tests". `cargo test` builds it;
// `cargo build` / `cargo run` do not, so none of this ends up in the binary. It is the
// compiler-enforced version of the `PRINT PURPOSES ONLY` comment fences above.
//
// These are UNIT tests: they live in the same file as the code, so they can reach private
// items — `merge`, `get_stats_into`, `encode_chunk`, and the struct's private fields. An
// integration test in `tests/` could not; it would only see `pub` items, and this crate has
// none. That distinction is what step 5 is about.
//
// Every assertion here was previously checked by hand in a throwaway program, by reading
// 5000 lines of output, or by running the binary twice and diffing. None of it was recorded
// anywhere until now.
#[cfg(test)]
mod tests {
    use super::*;

    // These were crate-root imports until 12.2b. The production code in this file no longer
    // touches the filesystem — `save`/`load` live in `model/bpe/serialize.rs` now — but the
    // persistence tests DELIBERATELY stayed here, because they drive `tok.save()` and
    // `Tokenizer::load()` (claims about the public type, not about the writer), and their
    // helpers are shared with the special-tokens section below. So the imports follow them
    // into the test module rather than staying at the crate root where they would be dead.
    use std::fs;
    use std::io;
    use std::path::Path;

    // `a_merged_token_need_not_be_valid_utf8` asserts that a token can be invalid UTF-8,
    // which is a fact about `vocab`, not about serialization — so it stays here and reaches
    // across for the renderer.
    use crate::model::bpe::serialize::render_token;

    /// A small, repetitive corpus. Deliberately NOT `TEXT`: a test that learns 1744 merges
    /// is a slow test, and slow tests get skipped. 900 bytes is enough for every property
    /// here and runs in milliseconds.
    fn corpus() -> String {
        "the quick brown fox jumps over the lazy dog. ".repeat(20)
    }

    /// Trained once on `corpus()`. Most tests want the same tokenizer.
    fn trained() -> Tokenizer {
        let mut tok = Tokenizer::new();
        tok.train(&corpus(), 320, false);
        tok
    }



    #[test]
    fn training_a_corpus_of_pure_repeats() {
        // The overlapping-run case driven through the real training loop, where each merge
        // feeds the next. `assert_consistent` runs after every one of them.
        let mut tok = Tokenizer::new();
        tok.train(&"aaaa aaaa ".repeat(40), 300, false);
        assert_eq!(tok.decode(&tok.encode_ordinary("aaaa")).unwrap(), "aaaa");
    }

    // -- the pre-tokenizer ------------------------------------------------------

    #[test]
    fn split_attaches_a_space_to_the_following_word() {
        // ` ?\p{L}+` — the space goes with the word AFTER it, not before. This is the
        // reason ("e", " ") can never end up in one chunk.
        assert_eq!(Tokenizer::new().split("Hello world"), vec!["Hello", " world"]);
    }

    #[test]
    fn split_chops_digits_into_groups_of_three() {
        // `\p{N}{1,3}`. This is why "2026" became "202" + "6" in the demo output.
        assert_eq!(Tokenizer::new().split("2026"), vec!["202", "6"]);
        assert_eq!(Tokenizer::new().split("1234567"), vec!["123", "456", "7"]);
    }

    #[test]
    fn split_isolates_contractions() {
        // `'(?i:[sdmt]|ll|ve|re)` — and `(?i:` makes it case-insensitive, so the shouting
        // version splits the same way.
        assert_eq!(Tokenizer::new().split("isn't"), vec!["isn", "'t"]);
        assert_eq!(Tokenizer::new().split("ISN'T"), vec!["ISN", "'T"]);
    }

    #[test]
    fn split_is_lossless() {
        // Whatever the arms do, concatenating the chunks must give the input back. This
        // is the property the round-trip depends on, and it holds for every input.
        for s in ROUND_TRIP_CASES {
            assert_eq!(Tokenizer::new().split(s).concat(), *s, "split lost data on {s:?}");
        }
    }

    #[test]
    fn pretokenizer_forbids_merges_across_word_boundaries() {
        // THE decisive check for step 3, previously done by grepping 5000 lines by eye.
        // (101, 32) is ("e", " "). Before the pre-tokenizer it was the single most frequent
        // pair in the corpus; after it, the two bytes land in different chunks and the
        // merge is unreachable. If this ever fails, the pre-tokenizer is not wired into
        // `train`.
        let tok = trained();
        assert!(
            tok.model.rank((101, 32)).is_none(),
            "learned a merge spanning a word boundary"
        );
    }

    // -- training ---------------------------------------------------------------

    #[test]
    fn training_is_deterministic() {
        // Rust HashMaps iterate in a randomised order, so `max_by_key` over `stats` would
        // pick a different pair among ties on every run and train a different tokenizer.
        // The `(count, Reverse(pair))` key makes the maximum unique. This replaces
        // "run the binary twice and diff 5000 lines".
        let (a, b) = (trained(), trained());
        assert!(a.model == b.model);
    }

    #[test]
    fn training_again_replaces_what_was_learned() {
        // Pins a behaviour step 12 changed WITHOUT meaning to, found by the post-12.4b
        // audit. Before 12.3b a second `train` kept the old merges and numbered the new ones
        // from 256 again, so two tokens shared each id and decode returned garbage. Since
        // 12.3b it replaces the model, which is correct — and nothing in the repo retrains,
        // so without this test the old behaviour could come back unnoticed.
        //
        // Public methods only, on purpose: that lets the same test be run against the
        // pre-step-12 code, where it fails.
        let second = "pack my box with five dozen liquor jugs. ".repeat(20);

        let mut retrained = Tokenizer::new();
        retrained.train(&corpus(), 320, false);
        retrained.train(&second, 300, false);

        let mut fresh = Tokenizer::new();
        fresh.train(&second, 300, false);

        assert_eq!(retrained.vocab_size(), fresh.vocab_size(), "the first run's merges survived");
        let ids = retrained.encode_ordinary(&second);
        assert_eq!(ids, fresh.encode_ordinary(&second));
        assert_eq!(retrained.decode(&ids).unwrap(), second);
    }

    #[test]
    fn repeating_the_corpus_does_not_change_what_is_learned() {
        // The strongest available check on the weighting, because it is a property rather
        // than a fixture: repeating a corpus multiplies every chunk count by the same
        // factor, so every pair's count scales uniformly and the argmax cannot move. If
        // weights were applied wrongly — dropped, or applied once instead of per chunk —
        // the two runs would diverge.
        //
        // `corpus()` ends in a space, so the repetitions join cleanly and no new chunk is
        // created at the seams.
        let base = corpus();
        let mut once = Tokenizer::new();
        once.train(&base, 320, false);

        let mut ten_times = Tokenizer::new();
        ten_times.train(&base.repeat(10), 320, false);

        assert!(once.model == ten_times.model);
    }

    #[test]
    fn deduplication_actually_collapses_a_repetitive_corpus() {
        // A corpus of one repeated sentence has very few distinct chunks. Before step 9
        // every occurrence was stored separately; the memory win is exactly this ratio.
        let tok = Tokenizer::new();
        let text = corpus();
        let total = tok.split(&text).len();
        let unique: std::collections::HashSet<&str> = tok.split(&text).into_iter().collect();

        assert!(total > 150, "expected a decent number of chunks, got {total}");
        assert!(
            unique.len() * 8 < total,
            "expected heavy duplication: {total} chunks but {} unique",
            unique.len()
        );
    }

    #[test]
    fn chunk_order_does_not_affect_what_is_learned() {
        // Step 9 made chunk order NONDETERMINISTIC: chunks come out of a HashMap, whose
        // iteration order is randomised per process. That is safe only because nothing
        // downstream depends on order — `stats` accumulates by addition, the winner is
        // picked by a total ordering, and `merge` treats each chunk independently.
        //
        // Simulating it directly: feeding the same multiset of chunks in a different order
        // must give the same tables. Reversing the corpus sentence-wise keeps every chunk
        // and every count while changing the order they are first seen in.
        // Every sentence is prefixed with a space so that each contributes exactly the same
        // chunks wherever it lands. (An earlier version of this test just reversed the
        // sentence order, which quietly changed the corpus: the FIRST word has no leading
        // space, so moving a sentence to the front swaps a " the" for a "the".)
        let sentences = ["the cat sat.", "a dog ran.", "the cat ran.", "a dog sat."];
        let forward: String =
            (0..12).flat_map(|_| sentences.iter()).map(|s| format!(" {s}")).collect();
        let backward: String =
            (0..12).flat_map(|_| sentences.iter().rev()).map(|s| format!(" {s}")).collect();
        assert_ne!(forward, backward, "the two corpora must differ in order");

        let mut a = Tokenizer::new();
        a.train(&forward, 300, false);
        let mut b = Tokenizer::new();
        b.train(&backward, 300, false);
        assert!(a.model == b.model, "learned tables depend on chunk order");
    }

    #[test]
    fn training_to_vocab_size_256_learns_nothing() {
        let mut tok = Tokenizer::new();
        tok.train(&corpus(), 256, false);
        assert_eq!(tok.model.merge_count(), 0);
        assert_eq!(tok.model.vocab_size(), 256);
    }

    #[test]
    #[should_panic(expected = "vocab_size must leave room")]
    fn training_below_256_is_rejected() {
        Tokenizer::new().train(&corpus(), 100, false);
    }

    #[test]
    fn training_stops_early_when_no_pairs_remain() {
        // The `else { break }` branch. It has never executed — not in any demo run, not
        // even at vocab_size 2000 over 22 KB. This is the first time that code runs.
        //
        // "a b c" splits into ["a", " b", " c"]. Only two pairs exist in the whole corpus,
        // so after two merges every chunk is a single token and there is nothing left to
        // count. Asking for 44 more merges must stop at 2, not panic and not loop.
        let mut tok = Tokenizer::new();
        tok.train("a b c", 300, false);
        assert_eq!(
            tok.model.merge_count(),
            2,
            "should have run out of pairs after 2 merges"
        );
        assert_eq!(tok.model.vocab_size(), 258);
    }

    #[test]
    fn new_token_ids_are_dense_and_start_at_256() {
        // Ids are handed out 256, 257, ... as merges are learned, which is what lets the
        // id double as the merge rank and lets `vocab` be a Vec instead of a HashMap.
        let tok = trained();
        assert_eq!(tok.model.vocab_size(), 256 + tok.model.merge_count());

        let mut ids: Vec<u32> = tok
            .model
            .merges_in_learned_order()
            .into_iter()
            .map(|(idx, _)| idx)
            .collect();
        ids.sort_unstable();
        let expected: Vec<u32> = (256..(256 + tok.model.merge_count() as u32)).collect();
        assert_eq!(ids, expected, "ids must be exactly 256..256+n with no gaps");
    }

    #[test]
    fn vocab_entries_are_concatenations_of_their_parents() {
        // vocab[idx] == vocab[p0] ++ vocab[p1], for every merge. This is the invariant
        // `decode` relies on, and the reason vocab never needs to be stored on disk —
        // it can always be rebuilt by replaying the merges.
        let tok = trained();
        for (idx, (p0, p1)) in tok.model.merges_in_learned_order() {
            let mut expected = tok.model.token(p0).unwrap().to_vec();
            expected.extend_from_slice(tok.model.token(p1).unwrap());
            assert_eq!(tok.model.token(idx).unwrap(), expected, "vocab[{idx}] is not p0 ++ p1");
        }
    }

    #[test]
    fn every_byte_has_a_token_before_training() {
        // Why byte-level BPE has no out-of-vocabulary case.
        let tok = Tokenizer::new();
        assert_eq!(tok.model.vocab_size(), 256);
        for i in 0..256usize {
            assert_eq!(tok.model.token(i as u32).unwrap(), [i as u8]);
        }
    }

    #[test]
    fn a_merged_token_need_not_be_valid_utf8() {
        // A merge can join the first half of a multi-byte character to something else.
        // In the real run, merge 262 was (226, 128) — two bytes from the middle of a
        // UTF-8 sequence. This is why vocab is Vec<Vec<u8>> and not Vec<String>: a
        // String literally could not hold that token.
        assert_eq!(render_token(&[226, 128]), "\u{fffd}");
        assert_eq!(render_token(&[104, 105]), "hi");
    }

    // -- encode / decode --------------------------------------------------------

    /// The 29 inputs that must survive encode -> decode. Shared verbatim with
    /// `tests/api.rs` via `include!`, which pastes the literal in at compile time.
    /// See `tests/data/round_trip_cases.rs` for why it is done this way.
    const ROUND_TRIP_CASES: &[&str] = include!("../tests/data/round_trip_cases.rs");

    #[test]
    fn round_trip_holds_for_tricky_inputs() {
        // Scripts entirely absent from the training corpus still round-trip — they just
        // compress badly, staying one token per byte. That is the no-OOV property.
        let tok = trained();
        for s in ROUND_TRIP_CASES {
            assert_eq!(*s, tok.decode(&tok.encode_ordinary(s)).unwrap(), "round trip failed for {s:?}");
        }
    }

    #[test]
    fn round_trip_holds_before_training_too() {
        // With no merges learned, encode is just "bytes widened to u32" and decode is
        // "concatenate the byte tokens". The identity still has to hold.
        let tok = Tokenizer::new();
        for s in ROUND_TRIP_CASES {
            assert_eq!(*s, tok.decode(&tok.encode_ordinary(s)).unwrap(), "round trip failed for {s:?}");
        }
    }

    #[test]
    fn an_untrained_tokenizer_encodes_to_raw_bytes() {
        assert_eq!(Tokenizer::new().encode_ordinary("hi"), vec![104, 105]);
        // 4 bytes for one character, because there are no merges yet.
        assert_eq!(Tokenizer::new().encode_ordinary("🐈").len(), "🐈".len());
    }

    #[test]
    fn training_compresses_its_own_corpus() {
        let tok = trained();
        let text = corpus();
        let ids = tok.encode_ordinary(&text);
        assert!(ids.len() < text.len(), "training learned nothing useful");
        assert_eq!(tok.decode(&ids).unwrap(), text);
    }

    #[test]
    fn encode_applies_merges_in_learned_order() {
        // `encode` picks the lowest merge id available at each step. Ids are handed out in
        // merge order, so lowest id == earliest-learned merge — which is exactly the order
        // `train` applied them. That is why re-encoding the training text reproduces what
        // training ended with.
        let tok = trained();
        let ids = tok.encode_ordinary(&corpus());
        for w in ids.windows(2) {
            // Any pair still present in the output must be one `encode` could not merge.
            assert!(
                tok.model.rank((w[0], w[1])).is_none(),
                "encode stopped while a known merge {:?} was still applicable",
                (w[0], w[1])
            );
        }
    }

    #[test]
    fn decoding_nothing_gives_nothing() {
        assert_eq!(trained().decode(&[]).unwrap(), "");
    }

    #[test]
    fn decode_replaces_invalid_utf8_rather_than_failing() {
        // A slice of ids can end mid-character — e.g. taking the first byte of "é" alone.
        // `from_utf8_lossy` is Python's errors='replace'.
        assert_eq!(Tokenizer::new().decode(&[195]).unwrap(), "\u{fffd}");
    }

    // -- the normalizer ---------------------------------------------------------
    //
    // The two PURE `Normalizer` tests moved to `text/normalizer.rs` at 12.1, next to the
    // code. These two stay because they are claims about a `Tokenizer` that happens to
    // have a normalizer, not about the normalizer itself.

    #[test]
    fn normalization_breaks_the_round_trip_and_that_is_correct() {
        // Asserting the BREAKAGE, not the fix. Normalizer and pre-tokenizer are encode-side
        // only — decode knows about neither — so a normalizer rewrites the text before
        // encode ever sees it and the original bytes are gone for good. This is inherent,
        // not a bug, and is exactly why GPT-4 does no normalization and why the default
        // here is `None`.
        let decomposed = "e\u{0301}";
        let precomposed = "\u{00e9}";

        let mut nfc = Tokenizer::with_config(GPT4_SPLIT_PATTERN, Normalizer::Nfc);
        nfc.train(&corpus(), 320, false);
        assert_ne!(nfc.decode(&nfc.encode_ordinary(decomposed)).unwrap(), decomposed);
        assert_eq!(nfc.decode(&nfc.encode_ordinary(precomposed)).unwrap(), precomposed);

        let mut nfd = Tokenizer::with_config(GPT4_SPLIT_PATTERN, Normalizer::Nfd);
        nfd.train(&corpus(), 320, false);
        assert_ne!(nfd.decode(&nfd.encode_ordinary(precomposed)).unwrap(), precomposed);
        assert_eq!(nfd.decode(&nfd.encode_ordinary(decomposed)).unwrap(), decomposed);
    }

    #[test]
    fn the_default_normalizer_is_none() {
        assert_eq!(Tokenizer::new().normalizer, Normalizer::None);
    }

    #[test]
    fn the_gpt2_pattern_also_compiles_and_trains() {
        // The other pattern is otherwise never exercised. It does NOT chop digits into
        // groups of three — that arm is a GPT-4 addition.
        let tok = Tokenizer::with_config(GPT2_SPLIT_PATTERN, Normalizer::None);
        assert_eq!(tok.split("2026"), vec!["2026"]);
        let mut tok = tok;
        tok.train(&corpus(), 300, false);
        assert_eq!(tok.decode(&tok.encode_ordinary("Hello world")).unwrap(), "Hello world");
    }

    // -- persistence ------------------------------------------------------------

    /// Tests run in parallel, so each one needs its own path.
    fn temp_prefix(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("naive_bbpe_tests");
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn save_then_load_reproduces_the_tokenizer_exactly() {
        let tok = trained();
        let prefix = temp_prefix("roundtrip");
        tok.save(&prefix).unwrap();

        let mut model = prefix.as_os_str().to_owned();
        model.push(".model");
        let loaded = Tokenizer::load(Path::new(&model)).unwrap();

        // `vocab` is NOT stored in the file — it is rebuilt by replaying the merges. That
        // it comes back identical is the proof that merges really are the ground truth.
        assert!(tok.model == loaded.model);
        assert_eq!(tok.normalizer, loaded.normalizer);
        assert_eq!(tok.pattern.as_str(), loaded.pattern.as_str());
    }

    #[test]
    fn a_loaded_tokenizer_encodes_and_decodes_identically() {
        // The behavioural version of the test above: same tables means same behaviour.
        let tok = trained();
        let prefix = temp_prefix("behaviour");
        tok.save(&prefix).unwrap();
        let mut model = prefix.as_os_str().to_owned();
        model.push(".model");
        let loaded = Tokenizer::load(Path::new(&model)).unwrap();

        for s in ROUND_TRIP_CASES {
            assert_eq!(tok.encode_ordinary(s), loaded.encode_ordinary(s), "encode differs for {s:?}");
            assert_eq!(*s, loaded.decode(&loaded.encode_ordinary(s)).unwrap(), "round trip failed for {s:?}");
        }
    }

    #[test]
    fn the_normalizer_survives_save_and_load() {
        // The reason the `.model` format carries a field minbpe's does not. A normalizer
        // rewrites text before encoding, so a model loaded without it would silently
        // encode differently from the tokenizer that was trained — no error, just wrong.
        for normalizer in [Normalizer::None, Normalizer::Nfc, Normalizer::Nfd] {
            let mut tok = Tokenizer::with_config(GPT4_SPLIT_PATTERN, normalizer);
            tok.train(&corpus(), 300, false);

            let prefix = temp_prefix(&format!("norm_{}", normalizer.name()));
            tok.save(&prefix).unwrap();
            let mut model = prefix.as_os_str().to_owned();
            model.push(".model");
            let loaded = Tokenizer::load(Path::new(&model)).unwrap();

            assert_eq!(loaded.normalizer, normalizer);
            assert_eq!(tok.encode_ordinary("e\u{0301}"), loaded.encode_ordinary("e\u{0301}"));
        }
    }

    #[test]
    fn the_split_pattern_survives_save_and_load() {
        let mut tok = Tokenizer::with_config(GPT2_SPLIT_PATTERN, Normalizer::None);
        tok.train(&corpus(), 300, false);
        let prefix = temp_prefix("pattern");
        tok.save(&prefix).unwrap();
        let mut model = prefix.as_os_str().to_owned();
        model.push(".model");
        let loaded = Tokenizer::load(Path::new(&model)).unwrap();

        assert_eq!(loaded.pattern.as_str(), GPT2_SPLIT_PATTERN);
        // The observable consequence: GPT-2 does not chop digits into groups of three.
        assert_eq!(loaded.split("2026"), vec!["2026"]);
    }

    #[test]
    fn the_model_file_has_the_documented_shape() {
        let tok = trained();
        let prefix = temp_prefix("shape");
        tok.save(&prefix).unwrap();
        let mut model = prefix.as_os_str().to_owned();
        model.push(".model");
        let text = fs::read_to_string(Path::new(&model)).unwrap();
        let lines: Vec<&str> = text.lines().collect();

        assert_eq!(lines[0], "naive_bbpe v1");
        assert_eq!(lines[1], GPT4_SPLIT_PATTERN);
        assert_eq!(lines[2], "none");
        assert_eq!(lines[3], "0"); // special tokens, until step 8
        assert_eq!(lines.len(), 4 + tok.model.merge_count());

        // Merge lines are in learned order, so line 4 is the first merge ever made.
        let first = tok
            .model
            .merges_in_learned_order()
            .into_iter()
            .find(|&(idx, _)| idx == 256)
            .unwrap()
            .1;
        assert_eq!(lines[4], format!("{} {}", first.0, first.1));
    }

    #[test]
    fn saving_the_same_tokenizer_twice_gives_identical_files() {
        // `merges` is a HashMap with no order of its own, so `save` has to sort by minted
        // id. Without that, two saves of the same tokenizer would differ.
        let tok = trained();
        let (a, b) = (temp_prefix("stable_a"), temp_prefix("stable_b"));
        tok.save(&a).unwrap();
        tok.save(&b).unwrap();

        for ext in [".model", ".vocab"] {
            let (mut pa, mut pb) = (a.as_os_str().to_owned(), b.as_os_str().to_owned());
            pa.push(ext);
            pb.push(ext);
            assert_eq!(
                fs::read_to_string(Path::new(&pa)).unwrap(),
                fs::read_to_string(Path::new(&pb)).unwrap(),
                "{ext} files differ between two saves"
            );
        }
    }


    #[test]
    fn the_vocab_file_stays_one_line_per_token_even_with_newline_tokens() {
        // The escaping above, proven end to end. This corpus is nothing but blank lines,
        // so BPE is forced to learn a token made of newlines — exactly the case that
        // would break a naive writer.
        let mut tok = Tokenizer::new();
        tok.train(&"a\n\n\nb\n\n\n".repeat(30), 300, false);
        assert!(
            tok.model.tokens().any(|v| v.len() > 1 && v.iter().all(|&b| b == b'\n')),
            "expected the corpus to produce a newline-only token"
        );

        let prefix = temp_prefix("newline_tokens");
        tok.save(&prefix).unwrap();
        let mut vocab_path = prefix.as_os_str().to_owned();
        vocab_path.push(".vocab");
        let text = fs::read_to_string(Path::new(&vocab_path)).unwrap();

        // 4 header comment lines + one line per token, and not one more.
        assert_eq!(text.lines().count(), 4 + tok.model.vocab_size());
    }

    // -- malformed models -------------------------------------------------------
    //
    // Every one of these is a corrupt FILE, not a bug in the code, so each must come back
    // as an `InvalidData` error with an explanation — never a panic.

    fn load_from_text(name: &str, contents: &str) -> io::Result<Tokenizer> {
        let path = temp_prefix(name);
        fs::write(&path, contents).unwrap();
        Tokenizer::load(&path)
    }

    #[test]
    fn a_file_with_the_wrong_header_is_rejected() {
        let e = load_from_text("bad_header", "not a model\n").unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
        assert!(e.to_string().contains("not a naive_bbpe model"), "{e}");
    }

    #[test]
    fn a_truncated_file_is_rejected() {
        // Header present, everything after it missing.
        let e = load_from_text("truncated", "naive_bbpe v1\n").unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
        assert!(e.to_string().contains("file ended before"), "{e}");
    }

    #[test]
    fn an_unknown_normalizer_name_is_rejected() {
        let text = format!("naive_bbpe v1\n{GPT4_SPLIT_PATTERN}\nnfkc\n0\n");
        let e = load_from_text("bad_norm", &text).unwrap_err();
        assert!(e.to_string().contains("unknown normalizer"), "{e}");
    }

    #[test]
    fn a_pattern_that_does_not_compile_is_rejected() {
        // `with_config` would panic on this. `load` must not, because the pattern came
        // from a file rather than from our own source code.
        let e = load_from_text("bad_pattern", "naive_bbpe v1\n[unclosed\nnone\n0\n").unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
        assert!(e.to_string().contains("does not compile"), "{e}");
    }

    #[test]
    fn a_malformed_merge_line_is_rejected() {
        let base = format!("naive_bbpe v1\n{GPT4_SPLIT_PATTERN}\nnone\n0\n");

        let e = load_from_text("no_space", &format!("{base}97\n")).unwrap_err();
        assert!(e.to_string().contains("expected"), "{e}");

        let e = load_from_text("not_a_number", &format!("{base}97 xyz\n")).unwrap_err();
        assert!(e.to_string().contains("not a token id"), "{e}");
    }

    #[test]
    fn a_merge_referring_to_a_token_that_does_not_exist_yet_is_rejected() {
        // The check that stops a corrupt file from indexing out of bounds. Token 999 is
        // not defined at the point this merge is replayed — only 0..256 are.
        let text = format!("naive_bbpe v1\n{GPT4_SPLIT_PATTERN}\nnone\n0\n97 999\n");
        let e = load_from_text("forward_ref", &text).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
        assert!(e.to_string().contains("does not exist yet"), "{e}");
    }

    #[test]
    fn a_model_file_that_repeats_a_pair_still_round_trips() {
        // The ONE corrupt file in this section that `load` accepts rather than rejects:
        // nothing checks for a merge line that repeats an earlier one. Rejecting it is a
        // behaviour change, deferred to step 13. Until then this pins what it does today.
        //
        // It is also the reason `push_merge` mints `vocab.len()` and NOT
        // `BYTE_ALPHABET + merges.len()`. The repeated pair overwrites its map entry, so
        // `merges.len()` stops growing: with the wrong rule the third merge would be given
        // id 257 — already `aa` — and "bb" would decode as "aa". Added at 12.6; the
        // post-12.4b audit showed that mutation passing every test.
        let text = format!("naive_bbpe v1\n{GPT4_SPLIT_PATTERN}\nnone\n0\n97 97\n97 97\n98 98\n");
        let tok = load_from_text("repeated_pair", &text).unwrap();

        assert_eq!(tok.vocab_size(), 259, "256 bytes + three merge lines");
        let ids = tok.encode_ordinary("bb");
        assert_eq!(ids, vec![258], "bb is the third merge line, so id 258");
        assert_eq!(tok.decode(&ids).unwrap(), "bb");
        assert_eq!(tok.decode(&tok.encode_ordinary("aa")).unwrap(), "aa");
    }

    // -- special tokens ---------------------------------------------------------

    fn with_specials() -> Tokenizer {
        let mut tok = trained();
        let ids = tok.vocab_size() as u32;
        tok.register_special_tokens(&[
            ("<|endoftext|>".to_string(), ids),
            ("<|fim_prefix|>".to_string(), ids + 1),
        ])
        .unwrap();
        tok
    }

    #[test]
    fn a_special_token_encodes_to_exactly_one_id() {
        // The point of a special token: 13 characters collapse to a single id that no
        // amount of ordinary text could ever produce.
        let tok = with_specials();
        let ids = tok.encode("<|endoftext|>", AllowedSpecial::All).unwrap();
        assert_eq!(ids, vec![tok.vocab_size() as u32]);
    }

    #[test]
    fn specials_are_cut_out_and_the_text_around_them_is_encoded_normally() {
        let tok = with_specials();
        let eot = tok.vocab_size() as u32;

        let ids = tok.encode("a<|endoftext|>b", AllowedSpecial::All).unwrap();
        assert_eq!(ids.first(), tok.encode_ordinary("a").first());
        assert!(ids.contains(&eot));
        assert_eq!(ids.last(), tok.encode_ordinary("b").last());

        // Leading, trailing and adjacent specials — the boundary cases in the cursor loop.
        assert_eq!(tok.encode("<|endoftext|><|endoftext|>", AllowedSpecial::All).unwrap(),
                   vec![eot, eot]);
        assert_eq!(tok.encode("", AllowedSpecial::All).unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn the_three_allowed_special_policies_differ() {
        let tok = with_specials();
        let text = "hi <|endoftext|> there";

        // All: one token.
        let all = tok.encode(text, AllowedSpecial::All).unwrap();
        assert!(all.contains(&(tok.vocab_size() as u32)));

        // None: ordinary text, so identical to encode_ordinary and much longer.
        let none = tok.encode(text, AllowedSpecial::None).unwrap();
        assert_eq!(none, tok.encode_ordinary(text));
        assert!(none.len() > all.len());

        // NoneRaise: an error naming the token and where it was found.
        let e = tok.encode(text, AllowedSpecial::NoneRaise).unwrap_err();
        assert!(e.contains("<|endoftext|>"), "unhelpful message: {e}");
    }

    #[test]
    fn none_raise_is_silent_when_no_special_is_present() {
        let tok = with_specials();
        assert_eq!(
            tok.encode("ordinary text", AllowedSpecial::NoneRaise).unwrap(),
            tok.encode_ordinary("ordinary text")
        );
    }

    #[test]
    fn a_special_round_trips_through_decode() {
        let tok = with_specials();
        for text in ["<|endoftext|>", "a<|endoftext|>b", "<|fim_prefix|>x<|endoftext|>"] {
            let ids = tok.encode(text, AllowedSpecial::All).unwrap();
            assert_eq!(tok.decode(&ids).unwrap(), text, "round trip failed for {text:?}");
        }
    }

    #[test]
    fn special_tokens_are_regex_escaped() {
        // `<|endoftext|>` contains `|`, which is regex alternation. Unescaped, that pattern
        // would match the empty string at every position and encode() would produce garbage.
        let mut tok = trained();
        let base = tok.vocab_size() as u32;
        tok.register_special_tokens(&[("a|b".to_string(), base), ("c.d".to_string(), base + 1)])
            .unwrap();

        // Literal matches only: "a|b" is the token, "a" and "b" are not, and "." is not
        // "any character".
        assert_eq!(tok.encode("a|b", AllowedSpecial::All).unwrap(), vec![base]);
        assert_eq!(tok.encode("c.d", AllowedSpecial::All).unwrap(), vec![base + 1]);
        assert_eq!(tok.encode("cXd", AllowedSpecial::All).unwrap(), tok.encode_ordinary("cXd"));
    }

    #[test]
    fn the_longest_matching_special_wins() {
        // Regex alternation is first-match, not longest-match. If the short token were
        // tried first, "<|endoftext|>" would match "<|end|>"... no — it would match the
        // shorter PREFIX token and leave the rest as text. Sorting by length prevents it.
        let mut tok = trained();
        let base = tok.vocab_size() as u32;
        tok.register_special_tokens(&[
            ("<|end|>".to_string(), base),
            ("<|end|>x".to_string(), base + 1),
        ])
        .unwrap();

        assert_eq!(tok.encode("<|end|>x", AllowedSpecial::All).unwrap(), vec![base + 1]);
        assert_eq!(tok.encode("<|end|>", AllowedSpecial::All).unwrap(), vec![base]);
    }

    #[test]
    fn a_special_is_never_chopped_up_by_the_pre_tokenizer() {
        // Without cutting specials out first, the split regex would break "<|endoftext|>"
        // into "<|", "endoftext", "|>" and encode the pieces as ordinary vocabulary.
        let tok = with_specials();
        assert!(tok.split("<|endoftext|>").len() > 1, "the split pattern does chop it up");
        assert_eq!(tok.encode("<|endoftext|>", AllowedSpecial::All).unwrap().len(), 1);
    }

    #[test]
    fn registering_an_id_that_would_shadow_a_learned_token_is_rejected() {
        let mut tok = trained();
        // Relative, not a hardcoded number: this corpus runs out of pairs before reaching
        // the requested vocab_size, so the last valid id is not the one you would guess.
        let last_learned = tok.vocab_size() as u32 - 1;
        let e = tok
            .register_special_tokens(&[("<|x|>".to_string(), last_learned)])
            .unwrap_err();
        assert!(e.contains("shadow"), "unhelpful message: {e}");

        // ...and the very next id up is fine, which is where the boundary actually is.
        assert!(tok
            .register_special_tokens(&[("<|x|>".to_string(), last_learned + 1)])
            .is_ok());
    }

    #[test]
    fn invalid_special_registrations_are_rejected() {
        let mut tok = trained();
        let base = tok.vocab_size() as u32;

        // A newline would corrupt the one-special-per-line `.model` format.
        assert!(tok
            .register_special_tokens(&[("a\nb".to_string(), base)])
            .unwrap_err()
            .contains("line break"));
        // An empty alternation branch matches everywhere.
        assert!(tok
            .register_special_tokens(&[(String::new(), base)])
            .unwrap_err()
            .contains("empty"));
        // Two specials on one id makes decode ambiguous.
        assert!(tok
            .register_special_tokens(&[("<|a|>".to_string(), base), ("<|b|>".to_string(), base)])
            .unwrap_err()
            .contains("share id"));
    }

    #[test]
    fn decoding_an_unknown_id_is_an_error_not_a_panic() {
        // Before step 8 this indexed `vocab` directly and panicked. Ids can now arrive from
        // a pipe, so it has to be an ordinary error.
        let tok = with_specials();
        let e = tok.decode(&[999_999]).unwrap_err();
        assert!(e.contains("unknown token id"), "unhelpful message: {e}");
        // The gap between the vocabulary and the first special is invalid too.
        assert!(tok.decode(&[tok.vocab_size() as u32 + 50]).is_err());
    }

    #[test]
    fn special_tokens_are_listed_lowest_id_first() {
        // The order is not cosmetic: `save` writes the specials in this order, so without the
        // sort `.model` files would come out in HashMap order and stop being reproducible.
        // Added at 12.8b, when mutating the sort away was caught only by luck — 5 runs in 12
        // — because every other test registers two or three specials, and two entries come
        // out of a HashMap already sorted half the time.
        //
        // Twenty, registered in a scrambled order (7 is coprime to 20, so `i * 7 % 20` visits
        // every offset once). A random order passes by chance about once in 20! ≈ 2.4e18.
        let mut tok = trained();
        let base = tok.vocab_size() as u32;
        let specials: Vec<(String, u32)> =
            (0..20u32).map(|i| (format!("<|s{i}|>"), base + (i * 7) % 20)).collect();
        tok.register_special_tokens(&specials).unwrap();

        let ids: Vec<u32> = tok.special_tokens().iter().map(|&(_, id)| id).collect();
        assert_eq!(ids, (base..base + 20).collect::<Vec<_>>());
    }

    #[test]
    fn specials_survive_save_and_load() {
        let tok = with_specials();
        let prefix = temp_prefix("specials");
        tok.save(&prefix).unwrap();
        let mut model = prefix.as_os_str().to_owned();
        model.push(".model");
        let loaded = Tokenizer::load(Path::new(&model)).unwrap();

        assert_eq!(loaded.special_tokens(), tok.special_tokens());
        let text = "a<|endoftext|>b<|fim_prefix|>c";
        assert_eq!(
            loaded.encode(text, AllowedSpecial::All).unwrap(),
            tok.encode(text, AllowedSpecial::All).unwrap()
        );
        assert_eq!(loaded.decode(&loaded.encode(text, AllowedSpecial::All).unwrap()).unwrap(), text);
    }

    #[test]
    fn the_model_file_records_the_specials() {
        let tok = with_specials();
        let prefix = temp_prefix("specials_shape");
        tok.save(&prefix).unwrap();
        let mut model = prefix.as_os_str().to_owned();
        model.push(".model");
        let text = fs::read_to_string(Path::new(&model)).unwrap();
        let lines: Vec<&str> = text.lines().collect();

        let base = tok.vocab_size();
        assert_eq!(lines[3], "2");
        assert_eq!(lines[4], format!("<|endoftext|> {base}"));
        assert_eq!(lines[5], format!("<|fim_prefix|> {}", base + 1));
        // Two extra lines compared with a model that has none.
        assert_eq!(lines.len(), 4 + 2 + tok.model.merge_count());
    }

    #[test]
    fn a_model_claiming_a_special_that_shadows_the_vocabulary_is_rejected() {
        // The registration check doubles as file validation.
        let text = format!("naive_bbpe v1\n{GPT4_SPLIT_PATTERN}\nnone\n1\n<|x|> 3\n97 98\n");
        let e = load_from_text("shadowing_special", &text).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
        assert!(e.to_string().contains("invalid special token"), "{e}");
    }

    #[test]
    fn a_tokenizer_with_no_specials_behaves_exactly_as_before() {
        // The regression guard for the whole step: nothing above changes when the feature
        // is unused. `encode` with any policy must equal `encode_ordinary`.
        let tok = trained();
        assert!(tok.special_tokens().is_empty());
        for s in ROUND_TRIP_CASES {
            let ordinary = tok.encode_ordinary(s);
            for policy in [AllowedSpecial::All, AllowedSpecial::None, AllowedSpecial::NoneRaise] {
                assert_eq!(tok.encode(s, policy).unwrap(), ordinary, "{policy:?} differs on {s:?}");
            }
        }
    }

    #[test]
    fn a_missing_file_reports_not_found_not_invalid_data() {
        // A file that isn't there is a different problem from a file that is corrupt, and
        // the error kinds should say so.
        let e = Tokenizer::load(Path::new("no/such/model.model")).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::NotFound);
    }

    // -- property tests ---------------------------------------------------------------
    //
    // Everything above this line is an EXAMPLE test: a hand-picked input paired with a
    // hand-written expected answer. That only ever checks the inputs someone thought of.
    //
    // The tests below state a RULE that has to hold for *every* input, and `proptest`
    // generates hundreds of inputs trying to break it. When one does break it, proptest
    // shrinks the failure down to the smallest input that still fails — so the report is
    // `"a\u{301}"` rather than the 200-character string that happened to contain it.
    //
    // A failure also gets written to `proptest-regressions/`, which is committed: a case
    // found once is replayed on every run from then on.
    //
    // These live here rather than in `tests/` because T1 needs the private `normalizer`
    // field — there is deliberately no public accessor for it — and only a unit test in
    // this file can see it.
    mod props {
        use super::*;
        use proptest::prelude::*;
        use std::sync::OnceLock;
        use std::sync::atomic::{AtomicU64, Ordering};

        // -- generators ---------------------------------------------------------------

        /// The alphabet these properties draw from.
        ///
        /// `any::<char>()` and `any::<String>()` are overwhelmingly ASCII. They would run
        /// every case below without once producing a combining mark or a 4-byte character
        /// — which are precisely the inputs that can break a *byte*-level tokenizer, since
        /// a merge is free to join half of a multi-byte character. So the weights are set
        /// by hand, and every arm is here for a specific reason.
        fn interesting_char() -> impl Strategy<Value = char> {
            prop_oneof![
                // Ordinary text. Whitespace gets its own arm because the split pattern
                // treats it in three separate branches: `\s*[\r\n]`, `\s+(?!\S)`, `\s+`.
                10 => prop_oneof![
                    9 => proptest::char::range(' ', '~'),
                    1 => prop::sample::select(vec![' ', '\t', '\n', '\r']),
                ],
                // 2-byte UTF-8.
                3 => proptest::char::range('\u{00a1}', '\u{024f}'),
                // Category M — the class `\p{L}` does *not* cover. This is the subject of
                // step 15 and the reason a letters-only pattern cuts Indic words apart.
                3 => proptest::char::range('\u{0300}', '\u{036f}'),
                // 3-byte, and full of category-M matras and viramas.
                3 => proptest::char::range('\u{0900}', '\u{097f}'),
                2 => proptest::char::range('\u{0b80}', '\u{0bff}'),
                // 3-byte with no word spaces at all.
                2 => proptest::char::range('\u{4e00}', '\u{4fff}'),
                // Category Cf: invisible, and `render_token` deliberately does not escape
                // these — it escapes Cc only.
                1 => prop::sample::select(vec!['\u{200c}', '\u{200d}']),
                // 4-byte UTF-8: the longest encoding, so the most ways for a merge to
                // land halfway through a character.
                1 => proptest::char::range('\u{1f300}', '\u{1f5ff}'),
            ]
        }

        fn interesting_text() -> impl Strategy<Value = String> {
            prop::collection::vec(interesting_char(), 0..200)
                .prop_map(|chars| chars.into_iter().collect())
        }

        /// Token ids for the `decode`-never-panics property. Deliberately *not* only valid
        /// ids: the point is what happens on the ones a caller should not have sent.
        fn arbitrary_ids() -> impl Strategy<Value = Vec<u32>> {
            let id = prop_oneof![
                // Mostly in range. The shared tokenizer asks for 320 but ends at 288:
                // `corpus()` is 900 bytes, so `train` runs out of pairs and stops early —
                // which is why nothing here hard-codes a vocabulary size as a bound.
                6 => 0u32..288,
                // Straddles both boundaries that matter: 256, where learned tokens start,
                // and the end of the vocabulary, wherever training actually stopped.
                3 => 250u32..400,
                // The value most likely to index out of bounds.
                1 => Just(u32::MAX),
            ];
            prop::collection::vec(id, 0..64)
        }

        // -- shared fixtures ----------------------------------------------------------
        //
        // Training and regex compilation are the expensive parts, and a property runs its
        // body hundreds of times. Both are done once and shared.

        fn trained_with(normalizer: Normalizer) -> Tokenizer {
            let mut tok = Tokenizer::with_config(GPT4_SPLIT_PATTERN, normalizer);
            tok.train(&corpus(), 320, false);
            tok
        }

        fn cached(normalizer: Normalizer) -> &'static Tokenizer {
            static NONE: OnceLock<Tokenizer> = OnceLock::new();
            static NFC: OnceLock<Tokenizer> = OnceLock::new();
            static NFD: OnceLock<Tokenizer> = OnceLock::new();
            let cell = match normalizer {
                Normalizer::None => &NONE,
                Normalizer::Nfc => &NFC,
                Normalizer::Nfd => &NFD,
            };
            cell.get_or_init(|| trained_with(normalizer))
        }

        /// Untrained: `split` depends only on the pattern, so training would be wasted.
        fn splitter(pattern: &'static str) -> &'static Tokenizer {
            static GPT4: OnceLock<Tokenizer> = OnceLock::new();
            static GPT2: OnceLock<Tokenizer> = OnceLock::new();
            let cell = if pattern == GPT4_SPLIT_PATTERN { &GPT4 } else { &GPT2 };
            cell.get_or_init(|| Tokenizer::with_config(pattern, Normalizer::None))
        }

        // -- T1: round trip -----------------------------------------------------------

        proptest! {
            /// The core contract, on generated input.
            ///
            /// Stated over `normalize(s)` and NOT over `s`, deliberately. `encode_ordinary`
            /// normalizes internally before splitting, so under NFC the input `"e" + U+0301`
            /// legitimately comes back as `"é"` — different bytes from what went in.
            /// Normalization is encode-side only and therefore inherently lossy; there is an
            /// example test above asserting exactly that. Writing this property over `s`
            /// would contradict it.
            #[test]
            fn t1_decode_of_encode_returns_the_normalized_input(s in interesting_text()) {
                for normalizer in [Normalizer::None, Normalizer::Nfc, Normalizer::Nfd] {
                    let tok = cached(normalizer);
                    let ids = tok.encode_ordinary(&s);
                    let back = tok.decode(&ids).expect("decoding our own ids cannot fail");
                    prop_assert_eq!(
                        back,
                        normalizer.normalize(&s).into_owned(),
                        "round trip broke under {:?}",
                        normalizer
                    );
                }
            }
        }

        // -- T2: the pre-tokenizer loses nothing ---------------------------------------

        proptest! {
            /// `split` is `find_iter().map(as_str).collect()` — it keeps the *matches* and
            /// silently drops anything between them. So a pattern that fails to match some
            /// character does not error: it deletes it, and the text is gone before BPE
            /// ever sees it.
            ///
            /// Both built-in patterns are total today, which is the only reason this holds.
            /// Nothing pinned that until now, and `load` will accept an arbitrary pattern
            /// out of a `.model` file with no totality check at all.
            #[test]
            fn t2_concatenating_the_split_pieces_rebuilds_the_input(s in interesting_text()) {
                for pattern in [GPT4_SPLIT_PATTERN, GPT2_SPLIT_PATTERN] {
                    let pieces = splitter(pattern).split(&s);
                    prop_assert_eq!(pieces.concat(), s.clone(), "pattern dropped input");
                }
            }
        }

        // -- T3: ids, determinism, and not panicking -----------------------------------

        proptest! {
            /// Two rules at once, because they share an encode.
            ///
            /// Determinism is not free: `HashMap` iteration order is randomized in Rust, so
            /// before the `(count, Reverse(pair))` tie-break was introduced the winning pair
            /// on a tie differed between runs of the same binary.
            #[test]
            fn t3_ids_are_in_range_and_encoding_is_deterministic(s in interesting_text()) {
                let tok = cached(Normalizer::None);
                let first = tok.encode_ordinary(&s);
                let second = tok.encode_ordinary(&s);
                prop_assert_eq!(&first, &second, "encode is not deterministic");
                for &id in &first {
                    prop_assert!(
                        (id as usize) < tok.vocab_size(),
                        "id {} >= vocab_size {}",
                        id,
                        tok.vocab_size()
                    );
                }
            }
        }

        proptest! {
            /// `decode` takes ids from anywhere — a file, a network, a model mismatch — so
            /// an out-of-range id has to be an `Err`, never an index panic. This is the
            /// exact class of bug fixed in step 8, when special-token ids started living
            /// above `vocab.len()` and the old bounds check stopped being sufficient.
            ///
            /// Either answer is acceptable here. Only a panic is a failure.
            #[test]
            fn t3_decode_never_panics_on_arbitrary_ids(ids in arbitrary_ids()) {
                let _ = cached(Normalizer::None).decode(&ids);
            }
        }

        // -- T4: save/load/save is a fixpoint ------------------------------------------

        /// A unique path prefix under the system temp directory, cleaned up on drop.
        ///
        /// There is no `tempfile` dev-dependency and this does not add one: the project's
        /// dependency list is deliberately short, and the whole need here is two file names
        /// and a guaranteed cleanup. `Drop` runs even when a property fails mid-case, which
        /// matters when proptest is about to re-run the body dozens of times while shrinking.
        struct TempPrefix(std::path::PathBuf);

        impl TempPrefix {
            fn new() -> Self {
                static COUNTER: AtomicU64 = AtomicU64::new(0);
                let mut path = std::env::temp_dir();
                path.push(format!(
                    "naive_bbpe_prop_{}_{}",
                    std::process::id(),
                    COUNTER.fetch_add(1, Ordering::Relaxed)
                ));
                TempPrefix(path)
            }

            /// `save` appends rather than replacing an extension, so the file it writes for
            /// prefix `p` is literally `p` + `".model"`. Mirror that here.
            fn with_ext(&self, ext: &str) -> std::path::PathBuf {
                let mut p = self.0.as_os_str().to_owned();
                p.push(ext);
                std::path::PathBuf::from(p)
            }
        }

        impl Drop for TempPrefix {
            fn drop(&mut self) {
                let _ = fs::remove_file(self.with_ext(".model"));
                let _ = fs::remove_file(self.with_ext(".vocab"));
            }
        }

        proptest! {
            // Every case trains a tokenizer, which is far more work than the other
            // properties do, so this one runs fewer cases. Slow tests get skipped, and a
            // skipped test checks nothing.
            #![proptest_config(ProptestConfig::with_cases(32))]

            /// Saving a model, loading it back and saving again must produce the same bytes.
            ///
            /// This is the property that `.model` not storing the vocabulary depends on:
            /// `load` rebuilds `vocab` by replaying the merge lines through the same
            /// concatenation loop `train` runs. If replay disagreed with training in any
            /// way, the second `.vocab` would differ from the first.
            ///
            /// A special token is registered so the `num_special` header and its lines are
            /// covered too, not just the merge lines.
            #[test]
            fn t4_save_load_save_is_byte_identical(s in interesting_text()) {
                // Generated text alone can be empty or have no repeats, which would learn
                // no merges and make the property vacuous. Prepending the fixed corpus
                // guarantees there is always something to learn while still varying with `s`.
                let mut tok = Tokenizer::new();
                tok.train(&format!("{}{}", corpus(), s), 300, false);
                let special_id = tok.vocab_size() as u32;
                tok.register_special_tokens(&[("<|endoftext|>".to_string(), special_id)])
                    .expect("id is exactly vocab_size, which is the lowest legal one");

                let first = TempPrefix::new();
                let second = TempPrefix::new();
                tok.save(&first.0).expect("writing to the temp dir");
                let reloaded =
                    Tokenizer::load(&first.with_ext(".model")).expect("reading back our own file");
                reloaded.save(&second.0).expect("writing to the temp dir");

                // Compared as bytes, and reported as a length rather than a diff: printing
                // two 14 KB files on failure would bury the shrunk input that explains it.
                let (a, b) = (
                    fs::read(first.with_ext(".model")).unwrap(),
                    fs::read(second.with_ext(".model")).unwrap(),
                );
                prop_assert!(a == b, ".model differs after reload ({} vs {} bytes)", a.len(), b.len());

                let (a, b) = (
                    fs::read(first.with_ext(".vocab")).unwrap(),
                    fs::read(second.with_ext(".vocab")).unwrap(),
                );
                prop_assert!(a == b, ".vocab differs after reload ({} vs {} bytes)", a.len(), b.len());
            }
        }
    }
}
