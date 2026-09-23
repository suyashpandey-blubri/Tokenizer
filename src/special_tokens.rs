//! Special tokens: reserved ids for literal markers like `<|endoftext|>`.
//!
//! WHY THIS IS AT THE TOP LEVEL, NOT IN `text/`
//! ---------------------------------------------
//! `text/`'s rule is that nothing in it knows what a token id is (`text/mod.rs`). Special
//! tokens break that three ways: they *are* a text-to-id table; they wrap the text pipeline
//! rather than sitting in it (`encode` finds them in the raw text before `encode_ordinary`
//! normalizes anything); and training never uses them. Nor are they BPE-specific, so not
//! `model/` either. They belong to the tokenizer as a whole.
//!
//! 12.8a moved the two pieces that needed no redesign. The three fields that must agree
//! with each other — token to id, id to token, and the regex built from those strings —
//! become one type here at 12.8b.

use fancy_regex::Regex;

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
pub(crate) fn compile_special_pattern(specials: &[(String, u32)]) -> Option<Regex> {
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
