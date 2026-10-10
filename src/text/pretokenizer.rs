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

use std::io::{self, Read};

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

    /// Read `reader` to the end and hand it to `f` in blocks of roughly `block_bytes`, each
    /// cut at a [`last_safe_cut`] — so splitting the blocks one by one gives exactly the
    /// chunks that splitting the whole input at once would. This is what lets training read
    /// a corpus far larger than memory.
    ///
    /// Refuses any pattern but the two built-in ones, because the cut rule is a claim about
    /// those two patterns and has been checked for nothing else.
    ///
    /// If a full buffer contains no safe cut, it grows by another block rather than cutting
    /// unsafely. That is correct but unbounded: an input with no safe cut at all (no
    /// newlines, or `\r\n` line endings throughout) is read whole, as before.
    pub(crate) fn for_each_block<R: Read>(
        &self,
        mut reader: R,
        block_bytes: usize,
        mut f: impl FnMut(&str) -> io::Result<()>,
    ) -> io::Result<()> {
        if self.as_str() != GPT4_SPLIT_PATTERN && self.as_str() != GPT2_SPLIT_PATTERN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "reading input in blocks is only proven safe for the gpt4 and gpt2 patterns",
            ));
        }
        let mut buf: Vec<u8> = Vec::new();
        // Bytes already handed to `f`, so a UTF-8 error can say where in the INPUT it is.
        let mut consumed: u64 = 0;
        loop {
            // Top up by one block. `take(n).read_to_end` stops at n bytes or at the end of
            // the input, so reading fewer than n means the end was reached.
            let read = (&mut reader).take(block_bytes as u64).read_to_end(&mut buf)?;
            let at_end = read < block_bytes;
            let cut = if at_end {
                buf.len()
            } else {
                match last_safe_cut(&buf) {
                    Some(cut) => cut,
                    None => continue, // no safe cut yet: grow the buffer
                }
            };
            // A cut is always just after a `\n`, which is never part of a multi-byte
            // character, so a valid file always gives valid blocks.
            let block = std::str::from_utf8(&buf[..cut]).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("not valid UTF-8 (at byte {})", consumed + e.valid_up_to() as u64),
                )
            })?;
            if !block.is_empty() {
                f(block)?;
            }
            consumed += cut as u64;
            buf.drain(..cut);
            if at_end {
                return Ok(());
            }
        }
    }
}

/// The default for [`PreTokenizer::for_each_block`]: big enough that per-block overhead
/// vanishes, small next to what the chunk tally itself holds.
pub(crate) const BLOCK_BYTES: usize = 64 << 20;

/// The last position in `bytes` where the input may be cut without changing what either
/// built-in pattern produces: just after a `\n` whose neighbours on BOTH sides are visible
/// ASCII (`!` to `~` — no space, no newline). `None` if there is no such position.
///
/// WHY BOTH SIDES
/// --------------
/// The obvious rule — "after a `\n` that is followed by a non-space" — is safe for GPT-4
/// but NOT for GPT-2. GPT-2's `\s+(?!\S)` looks at the character after a run of spaces, and
/// at the end of a block there is none: `"a \nb"` splits as `" "`, `"\n"` whole, but a block
/// ending in `"a \n"` splits as `" \n"`. Measured (prod.1) on `book.txt`, `rust.txt` and
/// 20 MB of fineweb with every cut applied at once: that rule changed the GPT-2 chunks on
/// all three; this one left both patterns identical on all three.
///
/// A visible ASCII character after the cut also keeps NFC/NFD per block equal to NFC/NFD
/// of the whole: it is never the second half of a composition, and the `\n` before the cut
/// combines with nothing.
pub(crate) fn last_safe_cut(bytes: &[u8]) -> Option<usize> {
    (1..bytes.len().saturating_sub(1))
        .rev()
        .find(|&i| {
            bytes[i] == b'\n' && bytes[i - 1].is_ascii_graphic() && bytes[i + 1].is_ascii_graphic()
        })
        .map(|i| i + 1)
}
