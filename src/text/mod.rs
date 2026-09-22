//! The text stages: everything that happens to a string *before* it becomes ids.
//!
//! WHY THESE THREE SHARE A DIRECTORY
//! ----------------------------------
//! Not by analogy to any other library — by where the types actually change. Every item in
//! here takes `&str` and returns text or *positions in text*; nothing in here knows what a
//! token id is:
//!
//! | | in | out |
//! |---|---|---|
//! | [`normalizer::Normalizer::normalize`] | `&str` | `Cow<str>` |
//! | [`pretokenizer::PreTokenizer::split`] | `&str` | `Vec<&str>` |
//! | `boundary` (12.4) | `&str` | positions |
//!
//! Everything under `model/` works on `Vec<u32>` and never sees a character. The
//! conversion happens at exactly one line in `train` — `chunk.bytes().map(|b| b as u32)`.
//! Upstream of it is `text/`; downstream is `model/`. The directory is that cut, not a
//! category someone invented.
//!
//! It is also a *sequence*: `train` normalizes, then splits, both before any id exists.
//!
//! The concrete payoff is step 15, where **all five** sub-steps land in this directory —
//! 15.1 normalization, 15.2 the `\p{M}` pattern fix, 15.3/15.4/15.5 boundaries — while
//! `model/` holds still. The ablation runs that step expects vary something in here and
//! nothing else.
//!
//! NOT one directory per component, the way a library with many implementations of each
//! would do it. That layout earns its keep when `normalizers/` holds a dozen normalizers.
//! Ours is one enum with three variants and 48 lines; a directory to hold it would be the
//! same empty ceremony that downgraded `vocab/initializer.rs` to a constant at 12.0.

pub mod normalizer;
pub mod pretokenizer;
