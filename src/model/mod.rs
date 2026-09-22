//! The model side: everything that works on ids rather than on characters.
//!
//! The boundary between this and `text/` is one line in `train` —
//! `chunk.bytes().map(|b| b as u32)`. Upstream of it, text; downstream, `Vec<u32>`.
//! Nothing in here knows what a character is.
//!
//! `bpe` is a directory rather than a file because step 17 may add a second model family
//! (Unigram), and the `.model` format is BPE's own: "the new id is `BYTE_ALPHABET + line
//! index`" is meaningless to an algorithm that does not merge pairs. That is also why
//! serialization lives under `bpe/` rather than at the crate root.

pub(crate) mod bpe;
