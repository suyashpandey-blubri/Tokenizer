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
//! 12.8a moved the two pieces that needed no redesign. 12.8b made the three fields that
//! must agree with each other — token to id, id to token, and the regex built from those
//! strings — one type, [`SpecialTokens`].

use std::collections::HashMap;

use fancy_regex::Regex;

/// The registered special tokens: both directions of the table, plus the regex that finds
/// them in text.
///
/// WHY THIS IS A TYPE
/// ------------------
/// The three fields were `Tokenizer`'s until 12.8b, and they are not independent: `by_id`
/// is meant to be the reverse of `by_token`, and `pattern` is compiled from exactly those
/// strings. Nothing enforced that. They were only ever *assigned* together, in one place,
/// and `load` built three empty ones by hand and filled them in later. This is the shape
/// 12.3a fixed for `merges` + `vocab`: the fields are private here, and the only way to get
/// a `SpecialTokens` is [`none`](Self::none) or [`new`](Self::new), which builds all three
/// at once.
///
/// `new` takes the id floor as an argument rather than reading it from the model, because
/// this type knows nothing about the model. That is also what will make the step-13 fix for
/// retraining (re-validate the specials against the new vocab size) one call.
///
/// A KNOWN HOLE, PRESERVED ON PURPOSE
/// -----------------------------------
/// `new` rejects two specials sharing an **id**, but not two sharing a **string**. Given
/// `<a>=300` and `<a>=301`, `by_token` keeps only 301 while `by_id` keeps both, so the two
/// directions disagree — and since `.model` is written from `by_token`, a save and reload
/// loses 300. That predates step 12, and fixing it (rejecting the repeat) changes behaviour,
/// so it is a step-13 item. 12.8b reproduces it exactly, and the gate checks that it does.
pub(crate) struct SpecialTokens {
    /// `"<|endoftext|>" -> 100000`. These are NOT learned by training and are NOT in
    /// `vocab`. Their ids sit *above* `vocab.len()`, which is what stops one from shadowing
    /// a token BPE actually learned.
    by_token: HashMap<String, u32>,
    /// The reverse direction, for `decode`. Kept as a second map rather than searched
    /// linearly because `decode` looks up every single id it is given.
    by_id: HashMap<u32, String>,
    /// An alternation of the escaped special tokens, compiled once. `None` when there are
    /// no specials, which is also the signal for `encode` to skip the whole stage.
    pattern: Option<Regex>,
}

impl SpecialTokens {
    /// No special tokens — what every tokenizer starts with.
    pub(crate) fn none() -> Self {
        SpecialTokens {
            by_token: HashMap::new(),
            by_id: HashMap::new(),
            pattern: None,
        }
    }

    /// Validate `specials` against `floor` (the vocabulary size) and build all three fields.
    /// The four rules and why each exists are on
    /// [`Tokenizer::register_special_tokens`](crate::Tokenizer::register_special_tokens),
    /// which is the public way in.
    pub(crate) fn new(specials: &[(String, u32)], floor: u32) -> Result<Self, String> {
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

        Ok(SpecialTokens {
            by_token: specials.iter().cloned().collect(),
            by_id: specials.iter().map(|(t, i)| (*i, t.clone())).collect(),
            pattern: compile_special_pattern(specials),
        })
    }

    /// The regex that finds any registered special in text, or `None` if there are none.
    pub(crate) fn pattern(&self) -> Option<&Regex> {
        self.pattern.as_ref()
    }

    /// The id of a special the pattern just matched. Indexes rather than returning an
    /// `Option`, exactly as `encode` did before 12.8b: `pattern` is built from these very
    /// strings, so a match that is not a key cannot happen.
    pub(crate) fn id(&self, token: &str) -> u32 {
        self.by_token[token]
    }

    /// The text of special id `id`, if it is one.
    pub(crate) fn token(&self, id: u32) -> Option<&str> {
        self.by_id.get(&id).map(String::as_str)
    }

    /// How many specials are registered, counted from `by_token` — which is what `decode`'s
    /// error message has always reported.
    pub(crate) fn len(&self) -> usize {
        self.by_token.len()
    }

    /// The registered special tokens, lowest id first.
    pub(crate) fn list(&self) -> Vec<(String, u32)> {
        let mut v: Vec<(String, u32)> =
            self.by_token.iter().map(|(t, &i)| (t.clone(), i)).collect();
        v.sort_by_key(|(_, i)| *i);
        v
    }
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
