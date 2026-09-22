//! Pipeline stage 2: cut the text into chunks that BPE is not allowed to merge across.
//!
//! THE PRE-TOKENIZER IS A CEILING, NOT A TENDENCY
//! -----------------------------------------------
//! BPE can only merge *inside* one chunk, and every chunk yields at least one output
//! token, so `output tokens >= chunk count`, always. No vocabulary size and no amount of
//! corpus escapes that bound. Which is why step 15.2 — `\p{L}++` to `[\p{L}\p{M}]++` —
//! is the single highest-leverage change on the roadmap: Devanagari and Tamil vowel signs
//! are Unicode category *Mark*, not *Letter*, so today's pattern cuts every Indic word at
//! every vowel sign and no downstream work can undo it.

use fancy_regex::Regex;

// These need `fancy-regex`, not the std `regex` crate: `\s+(?!\S)` is a negative
// lookahead and `\p{L}++` / `?+` are possessive quantifiers, neither of which a
// finite-automata engine can do.
//
// Note a space attaches to the FOLLOWING word (` ?\p{L}+`), so "world!" splits as
// [" world", "!"] and a pair like ("e", " ") can never occur inside one chunk.
pub const GPT4_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

// Kept for comparison: simpler, and it doesn't chop digits into groups of 3.
pub const GPT2_SPLIT_PATTERN: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// A compiled pre-tokenizer.
///
/// WHY A STRUCT AND NOT A FREE FUNCTION TAKING `&Regex`
/// -----------------------------------------------------
/// A free function would have worked for everything the code does today. The struct was
/// chosen deliberately, to have the seam already cut when step 15 arrives: 15.2 changes
/// the pattern, and 15.3's parked question may need the splitter to carry state (akshara
/// tables) that a bare `Regex` cannot hold. A seam that exists is cheap to re-point; one
/// that does not has to be cut later, under pressure, against a working baseline.
///
/// It is `pub(crate)`. The public API is the five items `main.rs` and `tests/api.rs`
/// import, and this is not one of them — a caller has no reason to build a pre-tokenizer
/// on its own. Promoting it later is additive; un-publishing it would not be.
///
/// Measured at 12.1: this costs nothing. Only `Tokenizer::split` changed call shape at
/// all — `as_str()` covers `Debug`, `write_model`, `write_vocab` and both save/load tests
/// unmodified, because `Regex` was already being used through exactly the two operations
/// a pre-tokenizer *is*.
pub(crate) struct PreTokenizer {
    pattern: Regex,
}

impl PreTokenizer {
    /// Fallible, because there are two callers with opposite needs and they must share a
    /// constructor: `with_config` takes a pattern from our own source and may panic on a
    /// bad one, while `load` takes it from a *file* and must not. Returning
    /// `fancy_regex::Error` unchanged keeps `load`'s error message byte-identical.
    pub(crate) fn new(pattern: &str) -> Result<Self, fancy_regex::Error> {
        Ok(PreTokenizer {
            pattern: Regex::new(pattern)?,
        })
    }

    /// The pattern source text, which is what `.model` stores and what `Debug` prints.
    /// A `Regex` cannot be compared or serialized; its source string can be both.
    pub(crate) fn as_str(&self) -> &str {
        self.pattern.as_str()
    }

    /// Cut `text` into chunks.
    ///
    /// **This silently drops anything between matches.** `concat(split(s)) == s` holds
    /// only because both built-in patterns are total — pinned by `split_is_lossless` and
    /// the T2 property from 11.1. `load` accepts an arbitrary pattern from a file with
    /// **no totality check**, so a lossy pattern loads fine and silently corrupts encode.
    /// A known wart, not yet fixed.
    pub(crate) fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        self.pattern
            .find_iter(text)
            .map(|m| m.expect("regex match failed").as_str())
            .collect()
    }
}
