//! Pipeline stage 1: rewrite the text before anything else looks at it.
//!
//! Moved here verbatim from `lib.rs` at 12.1. Step 15.1 (B1, Indic normalization) is the
//! next thing to touch this file.

use std::borrow::Cow;

use unicode_normalization::UnicodeNormalization;

/// The first pipeline stage. minbpe has no normalizer at all, and GPT-2/GPT-4 do no
/// normalization, so `None` is the faithful default — this exists to demonstrate the
/// stage. Note that anything other than `None` is LOSSY: it rewrites the text before
/// encoding sees it, so `decode(encode(x)) == x` no longer holds. That is inherent, and
/// is exactly why GPT-4 skips this stage.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Normalizer {
    None,
    Nfc,
    Nfd,
}

impl Normalizer {
    /// `Cow` = "clone on write": `None` hands back a borrow of the input and allocates
    /// nothing, while Nfc/Nfd have to build a new String. Returning `String` would force
    /// a full copy of the corpus even in the common no-op case.
    pub fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        match self {
            Normalizer::None => Cow::Borrowed(text),
            Normalizer::Nfc => Cow::Owned(text.nfc().collect()),
            Normalizer::Nfd => Cow::Owned(text.nfd().collect()),
        }
    }

    /// The spelling used in a `.model` file and accepted by `--normalizer`.
    pub fn name(&self) -> &'static str {
        match self {
            Normalizer::None => "none",
            Normalizer::Nfc => "nfc",
            Normalizer::Nfd => "nfd",
        }
    }

    /// Inverse of [`name`](Normalizer::name). The normalizer must survive save/load: it
    /// rewrites text before encoding, so loading a model without it would silently encode
    /// differently from the tokenizer that was trained.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "none" => Some(Normalizer::None),
            "nfc" => Some(Normalizer::Nfc),
            "nfd" => Some(Normalizer::Nfd),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both of these are pure `Normalizer` tests — they never build a `Tokenizer`, so they
    /// move here with the code. The two *other* normalizer-related tests
    /// (`normalization_breaks_the_round_trip_and_that_is_correct`,
    /// `the_default_normalizer_is_none`) stay in `lib.rs`: they assert things about a
    /// `Tokenizer` that happens to have a normalizer, which is a different claim.
    #[test]
    fn normalizer_none_borrows_instead_of_allocating() {
        // The reason the return type is Cow: the common case copies nothing.
        assert!(matches!(Normalizer::None.normalize("abc"), Cow::Borrowed(_)));
        assert!(matches!(Normalizer::Nfc.normalize("abc"), Cow::Owned(_)));
    }

    #[test]
    fn nfc_collapses_and_nfd_expands() {
        let decomposed = "e\u{0301}"; // 'e' + combining acute, 3 bytes
        let precomposed = "\u{00e9}"; // 'é' as one code point,  2 bytes
        assert_eq!(Normalizer::Nfc.normalize(decomposed), precomposed);
        assert_eq!(Normalizer::Nfd.normalize(precomposed), decomposed);
        // ...and each leaves its own spelling alone.
        assert_eq!(Normalizer::Nfc.normalize(precomposed), precomposed);
        assert_eq!(Normalizer::Nfd.normalize(decomposed), decomposed);
    }
}
