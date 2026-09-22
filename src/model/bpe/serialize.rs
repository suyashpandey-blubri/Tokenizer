//! `.model` and `.vocab`: writing them, and reading `.model` back.
//!
//! WHY THIS SITS UNDER `model/bpe/` AND NOT AT THE CRATE ROOT
//! -----------------------------------------------------------
//! The objection was that the `.model` header records the *pattern* and the *normalizer*,
//! which are `text/` concerns. True, and it does not decide the question. The format's core
//! rule is **"the new id is `BYTE_ALPHABET + line_index`"** — a statement about how BPE
//! mints ids, and meaningless to an algorithm that does not merge pairs. A Unigram model
//! (step 17) could not use this format at all; it would need its own. So the format belongs
//! to the model family.
//!
//! THE FORMAT
//! ----------
//! ```text
//! naive_bbpe v1
//! <pattern regex, verbatim>
//! <normalizer>          none|nfc|nfd
//! <num_special>
//! <token> <id>          x num_special
//! <p0> <p1>             x num_merges      (new id IMPLICIT = BYTE_ALPHABET + line index)
//! ```
//!
//! **The vocabulary is deliberately not stored.** Merges are ground truth; `load` replays
//! them by concatenation, running the exact loop `train` runs as it learns them. That is
//! also what makes the merge list directly comparable to rustbpe's `get_mergeable_ranks()`
//! — the differential test at 11.5 is a plain list `==` with no reimplementation.
//!
//! `.vocab` IS NEVER READ BACK, AND IS CURRENTLY LOSSY
//! ---------------------------------------------------
//! `render_token` runs `from_utf8_lossy`, so every token that is half a multi-byte
//! character collapses to U+FFFD: measured, **590 of 50,004** entries in `fineweb50k.vocab`
//! and 113 of 807 non-ASCII tokens. Never parse it. Step 13 replaces the lossy decode with
//! hex escaping (`\xNN`) so the file becomes an inspection tool rather than a trap — not
//! done here, because it would change `.vocab` bytes and break the equivalence gate this
//! refactor depends on.
//!
//! Like `trainer.rs`, the `impl Tokenizer` block moved here with **no visibility change**:
//! private fields are visible in descendants of the module that declares them.

use std::collections::HashMap;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::model::bpe::model::{BYTE_ALPHABET, initial_vocab};
use crate::text::pretokenizer::PreTokenizer;
use crate::{Normalizer, Tokenizer};

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
pub(crate) fn render_token(bytes: &[u8]) -> String {
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
pub(crate) const MODEL_HEADER: &str = "naive_bbpe v1";

/// Every failure in `load` is a malformed file rather than a bug, so they all become
/// `InvalidData` errors carrying an explanation.
pub(crate) fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

impl Tokenizer {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    // The ONE serialization test that is a unit test of a free function. The other 13
    // persistence and malformed-model tests stay in `lib.rs`: they drive `tok.save()` and
    // `Tokenizer::load()`, so they are claims about the public type, and their helpers
    // `temp_prefix` / `load_from_text` are shared with the special-tokens section, which
    // is not moving.
    #[test]
    fn render_token_escapes_control_characters() {
        // Without escaping, a token containing a newline would split one `.vocab` entry
        // across two lines and quietly corrupt the file's shape.
        assert_eq!(render_token(b"a\nb"), "a\\u000ab");
        assert_eq!(render_token(b"\t"), "\\u0009");
        assert_eq!(render_token(&[226, 128]), "\u{fffd}"); // invalid UTF-8, not a control
        assert_eq!(render_token(b"hi"), "hi");
    }
}
