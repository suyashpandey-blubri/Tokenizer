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

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use fancy_regex::Regex;

mod text;

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

// ---------------------------------------------------------------------------
// Pure helpers — no tokenizer state, so they stay free functions
// ---------------------------------------------------------------------------

/// The byte alphabet: one token per possible byte value, minted before training starts.
/// This is *the* reason byte-level BPE has no out-of-vocabulary case — every input byte
/// already has an id, so there is nothing to fall back to.
///
/// It is a named constant because the number is load-bearing in four places across two
/// functions written independently of each other: `with_config` and `load` both build the
/// initial table, and `train` and `load` both mint ids as `BYTE_ALPHABET + i`. `load`
/// carries its own copy of arithmetic `train` also performs — the two must agree forever,
/// and until now nothing in the code said so.
const BYTE_ALPHABET: u32 = 256;

/// The untrained vocabulary: `vocab[i] == [i]` for every byte value.
///
/// Extracted because `with_config` and `load` held two verbatim copies of it. `load` is the
/// one that matters: it must reconstruct *exactly* the state `train` started from, or
/// replaying the merge list rebuilds a different table than the one that was saved.
fn initial_vocab() -> Vec<Vec<u8>> {
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
fn get_stats_into(counts: &mut HashMap<(u32, u32), u64>, ids: &[u32], weight: u64) {
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
fn merge(ids: &[u32], pair: (u32, u32), idx: u32) -> Vec<u32> {
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

/// The pairs inside one chunk, with how many times each occurs.
///
/// Note this counts *overlapping* occurrences: `[a,a,a]` yields `(a,a) -> 2`, even though
/// `merge` will only replace one of them. That is deliberate and matches what the
/// from-scratch loop counted, so the two agree.
fn pairs_of(ids: &[u32]) -> HashMap<(u32, u32), i64> {
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
struct PairIndex {
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
    fn build(chunks: &[Vec<u32>], counts: &[u64]) -> Self {
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
    fn pop_best(&mut self) -> Option<(u32, u32)> {
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

    fn chunks_containing(&self, pair: (u32, u32)) -> Vec<usize> {
        self.where_.get(&pair).map(|s| s.iter().copied().collect()).unwrap_or_default()
    }

    /// Record that chunk `index` changed from `old` to `new`.
    fn apply_chunk_change(&mut self, index: usize, old: &[u32], new: &[u32], weight: u64) {
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
    fn assert_consistent(&self, chunks: &[Vec<u32>], counts: &[u64], after: &str) {
        let fresh = PairIndex::build(chunks, counts);
        assert_eq!(self.stats, fresh.stats, "stats drifted after {after}");
        assert_eq!(self.where_, fresh.where_, "where_ drifted after {after}");
    }
}

/// Render a token's byte string for display. Two things can go wrong, and both are handled
/// here rather than at each call site.
///
/// **The bytes need not be valid UTF-8.** A merge can join the first half of a multi-byte
/// character to something else — merge 262 in the demo corpus was `(226, 128)`, two bytes
/// from the middle of a UTF-8 sequence. Those become the replacement character U+FFFD.
/// Seeing "\u{fffd}" in a vocab file is expected, not a bug, and it is the reason `vocab`
/// is `Vec<Vec<u8>>` and not `Vec<String>` — a `String` could not hold that token at all.
///
/// **The bytes may contain control characters.** A token containing a newline would split
/// one `.vocab` entry across two lines and quietly corrupt the file's one-entry-per-line
/// shape, so control characters are escaped as `\uXXXX`.
///
/// This is minbpe's `render_token` / `replace_control_characters`, with one deliberate
/// narrowing: minbpe escapes every character in Unicode category C, which needs
/// `unicodedata`. `char::is_control` covers category Cc only — C0 and C1, which is every
/// character that can actually break the file format. Format characters like U+200D ZERO
/// WIDTH JOINER (category Cf) are left as-is here and would be escaped by minbpe. Since
/// `.vocab` is never parsed, the difference is cosmetic.
fn render_token(bytes: &[u8]) -> String {
    let mut out = String::new();
    for ch in String::from_utf8_lossy(bytes).chars() {
        if ch.is_control() {
            out.push_str(&format!("\\u{:04x}", ch as u32));
        } else {
            out.push(ch);
        }
    }
    out
}

/// Line 1 of every `.model` file. A file that does not start with this is not ours, and
/// saying so beats failing later with a confusing parse error.
const MODEL_HEADER: &str = "naive_bbpe v1";

/// Every failure in `load` is a malformed file rather than a bug, so they all become
/// `InvalidData` errors carrying an explanation.
fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

// ---------------------------------------------------------------------------
// The tokenizer
// ---------------------------------------------------------------------------

/// The two tables BPE is made of. `train` writes both; `encode` reads `merges` and
/// `decode` reads `vocab`. Bundling them is the only reason this type exists.
pub struct Tokenizer {
    /// (left_id, right_id) -> new_id. New ids are handed out 256, 257, ... in merge
    /// order, so the id doubles as the merge rank — no separate ranks table needed.
    merges: HashMap<(u32, u32), u32>,
    /// vocab[id] = the byte string that id expands to. `Vec<u8>` is Python's `bytes`.
    /// Ids are dense 0..vocab_size, so a Vec indexed by id beats a HashMap.
    vocab: Vec<Vec<u8>>,
    /// Applied to the text before encoding. Defaults to `None`, matching GPT-4.
    normalizer: Normalizer,
    /// The pre-tokenizer, compiled once at construction rather than per call.
    pattern: PreTokenizer,

    /// Special tokens: `"<|endoftext|>" -> 100000`. These are NOT learned by training and
    /// are NOT in `vocab`. Their ids sit *above* `vocab.len()`, which is what stops one
    /// from shadowing a token BPE actually learned.
    special_tokens: HashMap<String, u32>,
    /// The reverse direction, for `decode`. Kept as a second map rather than searched
    /// linearly because `decode` looks up every single id it is given.
    inverse_special_tokens: HashMap<u32, String>,
    /// An alternation of the escaped special tokens, compiled once. `None` when there are
    /// no specials, which is also the signal for `encode` to skip the whole stage.
    special_pattern: Option<Regex>,
}

/// What `encode` should do when a special token appears literally in the input text.
/// These are minbpe's three `allowed_special` policies.
///
/// The distinction matters because a special token is just *text* until someone decides it
/// is not. If a user's document genuinely contains the characters `<|endoftext|>`, treating
/// it as a document boundary silently corrupts their data — and treating it as text when it
/// was meant as a boundary silently corrupts the training run. There is no safe default, so
/// the caller has to say.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AllowedSpecial {
    /// Recognise every registered special token. What you want when you built the file.
    All,
    /// Recognise none of them; `<|endoftext|>` encodes as those 13 ordinary characters.
    None,
    /// Recognise none of them, but fail loudly if one appears. minbpe's default, and the
    /// only option that cannot silently do the wrong thing.
    NoneRaise,
}

/// Deriving `Debug` would print every one of `vocab`'s entries — 2000 of them after a
/// modest training run, and 50000 after a real one. This prints the shape instead, which
/// is what anyone looking at a `Tokenizer` in an error message actually wants.
impl fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab.len())
            .field("merges", &self.merges.len())
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
        self.vocab.len()
    }

    pub fn with_config(pattern: &str, normalizer: Normalizer) -> Self {
        Tokenizer {
            merges: HashMap::new(),
            vocab: initial_vocab(),
            normalizer,
            pattern: PreTokenizer::new(pattern).expect("split pattern failed to compile"),
            special_tokens: HashMap::new(),
            inverse_special_tokens: HashMap::new(),
            special_pattern: None,
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
        let floor = self.vocab.len() as u32;
        let mut seen_ids: HashMap<u32, &str> = HashMap::new();

        for (token, id) in specials {
            if token.is_empty() {
                return Err("a special token cannot be empty".to_string());
            }
            if token.contains('\n') || token.contains('\r') {
                return Err(format!(
                    "special token {token:?} contains a line break, which the .model format \
                     cannot represent"
                ));
            }
            if *id < floor {
                return Err(format!(
                    "special token {token:?} has id {id}, which would shadow a learned token \
                     (ids below {floor} are already taken by the vocabulary)"
                ));
            }
            if let Some(other) = seen_ids.insert(*id, token) {
                return Err(format!("special tokens {other:?} and {token:?} share id {id}"));
            }
        }

        self.special_tokens = specials.iter().cloned().collect();
        self.inverse_special_tokens =
            specials.iter().map(|(t, i)| (*i, t.clone())).collect();
        self.special_pattern = Self::compile_special_pattern(specials);
        Ok(())
    }

    /// Build the alternation used to find special tokens in text.
    ///
    /// Two details that are easy to get wrong:
    ///
    /// - **Escaping.** A special token is a literal string, but it is going into a regex.
    ///   `<|endoftext|>` contains `|`, which is regex alternation — unescaped, that pattern
    ///   would match the empty string everywhere. `fancy_regex::escape` handles it, so no
    ///   new dependency is needed.
    /// - **Longest first.** Regex alternation is first-match, not longest-match. With
    ///   `<|end|>` listed before `<|endoftext|>`, the text `<|endoftext|>` would match the
    ///   short one and leave `oftext|>` behind as ordinary text. Sorting by descending
    ///   length makes the longer token win.
    fn compile_special_pattern(specials: &[(String, u32)]) -> Option<Regex> {
        if specials.is_empty() {
            return None;
        }
        let mut tokens: Vec<&String> = specials.iter().map(|(t, _)| t).collect();
        tokens.sort_by_key(|t| std::cmp::Reverse(t.len()));
        let alternation = tokens
            .iter()
            .map(|t| fancy_regex::escape(t).into_owned())
            .collect::<Vec<_>>()
            .join("|");
        // Cannot fail: every branch is an escaped literal.
        Some(Regex::new(&alternation).expect("escaped literals always compile"))
    }

    /// The registered special tokens, lowest id first.
    pub fn special_tokens(&self) -> Vec<(String, u32)> {
        let mut v: Vec<(String, u32)> =
            self.special_tokens.iter().map(|(t, &i)| (t.clone(), i)).collect();
        v.sort_by_key(|(_, i)| *i);
        v
    }

    /// The pre-tokenizer stage: cut the text into chunks that BPE is not allowed to
    /// merge across.
    pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        self.pattern.split(text)
    }

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

            let idx = BYTE_ALPHABET + i; // this is our new token

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

            self.merges.insert((p0, p1), idx);

            // Build this token's byte string immediately, while we still have the pair
            // in hand. Because we append as we mint, ids stay dense and in order by
            // construction — no separate vocab-building pass, and nothing to sort.
            let mut bytes = self.vocab[p0 as usize].clone();
            bytes.extend_from_slice(&self.vocab[p1 as usize]);
            self.vocab.push(bytes);

            if verbose && (i == 0 || i + 1 == new_tokens || (i + 1) % report_every == 0) {
                println!(
                    "  merge {:>6}/{}  {:>5} + {:<5} -> {:<5} {:>12}  [{:.1}s]",
                    i + 1,
                    new_tokens,
                    p0,
                    p1,
                    idx,
                    format!("{:?}", render_token(&self.vocab[idx as usize])),
                    started.elapsed().as_secs_f64(),
                );
            }
        }
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
        let Some(pattern) = self.special_pattern.as_ref() else {
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
            ids.push(self.special_tokens[m.as_str()]);
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

        // Running this loop till merging is possible.
        while tokens.len() >= 2 {
            // Weight 1: encoding looks at one piece of text, so there is nothing to
            // deduplicate and the counts are only used as a set of "which pairs exist".
            let mut stats: HashMap<(u32, u32), u64> = HashMap::new();
            get_stats_into(&mut stats, &tokens, 1);

            // Python: `min(stats, key=lambda p: merges.get(p, float('inf')))`, then
            // `if pair not in merges: break`. Here we filter to the pairs that are in
            // merges and take the smallest id — ids are handed out in merge order, so
            // the lowest id is the earliest-learned merge. `None` is the `break` case.
            let best = stats
                .keys()
                .filter_map(|&pair| self.merges.get(&pair).map(|&idx| (idx, pair)))
                .min();

            match best {
                Some((idx, pair)) => tokens = merge(&tokens, pair, idx),
                None => break, // we have come to a point where no more merges can be made
            }
        }
        tokens
    }

    // -- persistence ------------------------------------------------------------

    /// Write `<prefix>.model` and `<prefix>.vocab`.
    ///
    /// This is where `dump_tables` went. It was print-only scaffolding that showed the two
    /// tables on stdout; the same information is now a pair of files that outlive the
    /// process, which is the entire point of the step.
    ///
    /// The two files have completely different jobs:
    ///
    /// - **`.model`** is the real artifact. `load` reads it and nothing else.
    /// - **`.vocab`** is for humans and is *never read back*. It exists so you can look at
    ///   what training produced without writing a program. minbpe does the same.
    pub fn save(&self, prefix: &Path) -> io::Result<()> {
        // Append rather than `Path::with_extension`, which would REPLACE an existing
        // extension: `with_extension` on "models/fw.v2" gives "models/fw.model", losing
        // the ".v2". Appending gives "models/fw.v2.model", which is what anyone typing
        // that prefix meant.
        let mut model_path = prefix.as_os_str().to_owned();
        model_path.push(".model");
        let mut vocab_path = prefix.as_os_str().to_owned();
        vocab_path.push(".vocab");

        self.write_model(&mut BufWriter::new(fs::File::create(&model_path)?))?;
        self.write_vocab(&mut BufWriter::new(fs::File::create(&vocab_path)?))?;
        Ok(())
    }

    /// The machine-readable half. Format, version 1:
    ///
    /// ```text
    /// naive_bbpe v1          <- header, so a wrong file fails loudly
    /// <pattern>              <- the pre-tokenizer regex, verbatim
    /// <normalizer>           <- none | nfc | nfd
    /// <num_special>          <- always 0 until step 8, but the field exists now so
    ///                           adding special tokens will not change the format version
    /// <token> <id>           x num_special
    /// <p0> <p1>              x num_merges, in learned order
    /// ```
    ///
    /// **Vocab is deliberately not stored.** Merges are the ground truth; vocab is derived
    /// from them by concatenation, so writing it down would be storing the same information
    /// twice and inviting the two copies to disagree. `load` rebuilds it by replaying the
    /// merge lines — the exact loop `train` already runs.
    ///
    /// The new id is implicit too: it is `256 + line_index`, because ids are handed out in
    /// merge order. That is the same fact that lets `encode` use "lowest id" to mean
    /// "earliest merge".
    fn write_model(&self, out: &mut impl Write) -> io::Result<()> {
        writeln!(out, "{MODEL_HEADER}")?;
        writeln!(out, "{}", self.pattern.as_str())?;
        writeln!(out, "{}", self.normalizer.name())?;
        let specials = self.special_tokens();
        writeln!(out, "{}", specials.len())?;
        for (token, id) in specials {
            writeln!(out, "{token} {id}")?;
        }

        for (_, (p0, p1)) in self.merges_in_learned_order() {
            writeln!(out, "{p0} {p1}")?;
        }
        Ok(())
    }

    /// The human-readable half. Never parsed, so the format is free to be whatever is
    /// easiest to read. Control characters are escaped by [`render_token`], without which
    /// a token containing a newline would silently split one entry across two lines.
    fn write_vocab(&self, out: &mut impl Write) -> io::Result<()> {
        writeln!(out, "# naive_bbpe vocabulary — for reading, never loaded back")?;
        writeln!(out, "# pattern:    {}", self.pattern.as_str())?;
        writeln!(out, "# normalizer: {}", self.normalizer.name())?;
        writeln!(
            out,
            "# {} entries: 256 byte tokens + {} learned",
            self.vocab.len(),
            self.merges.len()
        )?;

        // id -> the pair it was built from, for the learned half.
        let parents: HashMap<u32, (u32, u32)> =
            self.merges.iter().map(|(&pair, &idx)| (idx, pair)).collect();

        for (idx, bytes) in self.vocab.iter().enumerate() {
            let rendered = render_token(bytes);
            match parents.get(&(idx as u32)) {
                // A learned token: show what it was built from, so the file reads as a
                // history of the merges rather than a flat list.
                Some(&(p0, p1)) => writeln!(
                    out,
                    "{idx}\t[{rendered}]\t= [{}] + [{}]",
                    render_token(&self.vocab[p0 as usize]),
                    render_token(&self.vocab[p1 as usize]),
                )?,
                // One of the 256 byte tokens.
                None => writeln!(out, "{idx}\t[{rendered}]")?,
            }
        }
        Ok(())
    }

    /// `merges` is a HashMap and has no order of its own. Sorting by the minted id
    /// recovers the order training created them in.
    fn merges_in_learned_order(&self) -> Vec<(u32, (u32, u32))> {
        let mut by_id: Vec<(u32, (u32, u32))> =
            self.merges.iter().map(|(&pair, &idx)| (idx, pair)).collect();
        by_id.sort_unstable();
        by_id
    }

    /// Read a `<prefix>.model` written by [`save`](Tokenizer::save).
    ///
    /// Every failure here is a malformed *file*, not a bug, so they come back as
    /// `io::ErrorKind::InvalidData` rather than a panic — a corrupt model should print a
    /// message, not abort the process.
    pub fn load(path: &Path) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let mut lines = text.lines();

        let mut next = |what: &str| -> io::Result<&str> {
            lines.next().ok_or_else(|| invalid(format!("file ended before {what}")))
        };

        let header = next("the header")?;
        if header != MODEL_HEADER {
            return Err(invalid(format!(
                "not a naive_bbpe model: expected {MODEL_HEADER:?} on line 1, found {header:?}"
            )));
        }

        let pattern = next("the split pattern")?.to_string();
        let normalizer_name = next("the normalizer")?;
        let normalizer = Normalizer::from_name(normalizer_name)
            .ok_or_else(|| invalid(format!("unknown normalizer: {normalizer_name:?}")))?;

        let raw = next("the special-token count")?;
        let num_special: usize = raw
            .trim()
            .parse()
            .map_err(|_| invalid(format!("special-token count is not a number: {raw:?}")))?;
        let mut special_tokens = Vec::with_capacity(num_special);
        for _ in 0..num_special {
            let line = next("a special token")?;
            let (token, id) = line
                .rsplit_once(' ')
                .ok_or_else(|| invalid(format!("malformed special token line: {line:?}")))?;
            let id: u32 = id
                .parse()
                .map_err(|_| invalid(format!("special token id is not a number: {id:?}")))?;
            special_tokens.push((token.to_string(), id));
        }
        // `with_config` would panic on a pattern that does not compile. A corrupt file
        // must not be able to do that, so compile it here where the error is catchable.
        let compiled = PreTokenizer::new(&pattern)
            .map_err(|e| invalid(format!("split pattern does not compile: {e}")))?;

        let mut tok = Tokenizer {
            merges: HashMap::new(),
            vocab: initial_vocab(),
            normalizer,
            pattern: compiled,
            // Registered further down, once the merges have been replayed:
            // `register_special_tokens` validates ids against `vocab.len()`, which is not
            // final until then.
            special_tokens: HashMap::new(),
            inverse_special_tokens: HashMap::new(),
            special_pattern: None,
        };

        // Replay the merges in file order, rebuilding vocab as we go — exactly what
        // `train` does as it learns them.
        for (i, line) in lines.enumerate() {
            let line_no = i + 5 + num_special;
            let (a, b) = line
                .split_once(' ')
                .ok_or_else(|| invalid(format!("line {line_no}: expected \"<p0> <p1>\", found {line:?}")))?;
            let parse = |s: &str| -> io::Result<u32> {
                s.trim()
                    .parse::<u32>()
                    .map_err(|_| invalid(format!("line {line_no}: not a token id: {s:?}")))
            };
            let (p0, p1) = (parse(a)?, parse(b)?);

            // A merge can only refer to tokens that already exist. Without this check a
            // corrupt file would index out of bounds and panic.
            let known = tok.vocab.len() as u32;
            if p0 >= known || p1 >= known {
                return Err(invalid(format!(
                    "line {line_no}: merge ({p0}, {p1}) refers to a token that does not exist yet \
                     (only 0..{known} are defined at that point)"
                )));
            }

            let idx = BYTE_ALPHABET + i as u32;
            tok.merges.insert((p0, p1), idx);
            let mut bytes = tok.vocab[p0 as usize].clone();
            bytes.extend_from_slice(&tok.vocab[p1 as usize]);
            tok.vocab.push(bytes);
        }

        // Now that vocab is complete, the "id must not shadow a learned token" check is
        // meaningful. Doing it here also validates the file: a model claiming a special at
        // id 300 in a 2000-token vocabulary is corrupt, and this is what says so.
        tok.register_special_tokens(&special_tokens)
            .map_err(|e| invalid(format!("invalid special token in model file: {e}")))?;

        Ok(tok)
    }

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
            if let Some(bytes) = self.vocab.get(idx as usize) {
                tokens.extend_from_slice(bytes);
            } else if let Some(special) = self.inverse_special_tokens.get(&idx) {
                tokens.extend_from_slice(special.as_bytes());
            } else {
                return Err(format!(
                    "unknown token id {idx}: this model has {} vocabulary entries and {} \
                     special tokens",
                    self.vocab.len(),
                    self.special_tokens.len()
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
            !tok.merges.contains_key(&(101, 32)),
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
        assert_eq!(a.merges, b.merges);
        assert_eq!(a.vocab, b.vocab);
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

        assert_eq!(once.merges, ten_times.merges);
        assert_eq!(once.vocab, ten_times.vocab);
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
        assert_eq!(a.merges, b.merges, "learned tables depend on chunk order");
    }

    #[test]
    fn training_to_vocab_size_256_learns_nothing() {
        let mut tok = Tokenizer::new();
        tok.train(&corpus(), 256, false);
        assert!(tok.merges.is_empty());
        assert_eq!(tok.vocab.len(), 256);
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
        assert_eq!(tok.merges.len(), 2, "should have run out of pairs after 2 merges");
        assert_eq!(tok.vocab.len(), 258);
    }

    #[test]
    fn new_token_ids_are_dense_and_start_at_256() {
        // Ids are handed out 256, 257, ... as merges are learned, which is what lets the
        // id double as the merge rank and lets `vocab` be a Vec instead of a HashMap.
        let tok = trained();
        assert_eq!(tok.vocab.len(), 256 + tok.merges.len());

        let mut ids: Vec<u32> = tok.merges.values().copied().collect();
        ids.sort_unstable();
        let expected: Vec<u32> = (256..(256 + tok.merges.len() as u32)).collect();
        assert_eq!(ids, expected, "ids must be exactly 256..256+n with no gaps");
    }

    #[test]
    fn vocab_entries_are_concatenations_of_their_parents() {
        // vocab[idx] == vocab[p0] ++ vocab[p1], for every merge. This is the invariant
        // `decode` relies on, and the reason vocab never needs to be stored on disk —
        // it can always be rebuilt by replaying the merges.
        let tok = trained();
        for (&(p0, p1), &idx) in &tok.merges {
            let mut expected = tok.vocab[p0 as usize].clone();
            expected.extend_from_slice(&tok.vocab[p1 as usize]);
            assert_eq!(tok.vocab[idx as usize], expected, "vocab[{idx}] is not p0 ++ p1");
        }
    }

    #[test]
    fn every_byte_has_a_token_before_training() {
        // Why byte-level BPE has no out-of-vocabulary case.
        let tok = Tokenizer::new();
        assert_eq!(tok.vocab.len(), 256);
        for i in 0..256usize {
            assert_eq!(tok.vocab[i], vec![i as u8]);
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
                !tok.merges.contains_key(&(w[0], w[1])),
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
        assert_eq!(tok.merges, loaded.merges);
        assert_eq!(tok.vocab, loaded.vocab);
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
        assert_eq!(lines.len(), 4 + tok.merges.len());

        // Merge lines are in learned order, so line 4 is the first merge ever made.
        let first = tok.merges.iter().find(|&(_, &idx)| idx == 256).unwrap().0;
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
    fn render_token_escapes_control_characters() {
        // Without escaping, a token containing a newline would split one `.vocab` entry
        // across two lines and quietly corrupt the file's shape.
        assert_eq!(render_token(b"a\nb"), "a\\u000ab");
        assert_eq!(render_token(b"\t"), "\\u0009");
        assert_eq!(render_token(&[226, 128]), "\u{fffd}"); // invalid UTF-8, not a control
        assert_eq!(render_token(b"hi"), "hi");
    }

    #[test]
    fn the_vocab_file_stays_one_line_per_token_even_with_newline_tokens() {
        // The escaping above, proven end to end. This corpus is nothing but blank lines,
        // so BPE is forced to learn a token made of newlines — exactly the case that
        // would break a naive writer.
        let mut tok = Tokenizer::new();
        tok.train(&"a\n\n\nb\n\n\n".repeat(30), 300, false);
        assert!(
            tok.vocab.iter().any(|v| v.len() > 1 && v.iter().all(|&b| b == b'\n')),
            "expected the corpus to produce a newline-only token"
        );

        let prefix = temp_prefix("newline_tokens");
        tok.save(&prefix).unwrap();
        let mut vocab_path = prefix.as_os_str().to_owned();
        vocab_path.push(".vocab");
        let text = fs::read_to_string(Path::new(&vocab_path)).unwrap();

        // 4 header comment lines + one line per token, and not one more.
        assert_eq!(text.lines().count(), 4 + tok.vocab.len());
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
        assert_eq!(lines.len(), 4 + 2 + tok.merges.len());
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
