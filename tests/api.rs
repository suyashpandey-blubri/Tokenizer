//! Integration tests: the public API, seen from outside.
//!
//! Every `.rs` file directly inside `tests/` is compiled as its own **separate crate**, and
//! links against `naive_bbpe` the way any other project would. That has one consequence
//! that shapes everything below: **only `pub` items are visible here.** There is no
//! `merge`, no `get_stats_into`, no `encode_chunk`, and no `tok.merges` / `tok.vocab`.
//!
//! That restriction is the point. The unit tests in `src/lib.rs` sit in the same file as
//! the code and can see everything, so they can never notice a missing `pub`. This file
//! **fails to compile** if the public surface is incomplete — which makes "did we export
//! enough to be usable?" a compile-time question rather than a matter of opinion.
//!
//! So these test *behaviour reachable through the API*, not internals.

use naive_bbpe::{
    AllowedSpecial, GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN, Normalizer, Tokenizer,
};

/// Small and repetitive, same as the unit tests use. Kept separate rather than shared,
/// because sharing it would mean exporting a test fixture from the library.
fn corpus() -> String {
    "the quick brown fox jumps over the lazy dog. ".repeat(20)
}

fn trained() -> Tokenizer {
    let mut tok = Tokenizer::new();
    tok.train(&corpus(), 320, false);
    tok
}

/// The 29 cases, exercised through the public API only. The same literal the unit tests
/// in `src/lib.rs` use — `include!` pastes it into both crates at compile time, which is
/// how two separate crates share a fixture without the library exporting one.
const ROUND_TRIP_CASES: &[&str] = include!("data/round_trip_cases.rs");

#[test]
fn the_documented_workflow_works_end_to_end() {
    // Construct, train, encode, decode — the four calls a user actually makes. If this
    // compiles and passes, the library is usable by someone who has never seen its source.
    let mut tok = Tokenizer::new();
    tok.train(&corpus(), 320, false);

    let text = corpus();
    let ids = tok.encode_ordinary(&text);

    assert_eq!(tok.decode(&ids).unwrap(), text);
    assert!(ids.len() < text.len(), "no compression was achieved");
}

#[test]
fn round_trip_holds_through_the_public_api() {
    let tok = trained();
    for s in ROUND_TRIP_CASES {
        assert_eq!(*s, tok.decode(&tok.encode_ordinary(s)).unwrap(), "round trip failed for {s:?}");
    }
}

#[test]
fn both_split_patterns_are_reachable_and_usable() {
    // The GPT-2 pattern was `#[allow(dead_code)]` until this step, because nothing called
    // it. Making it `pub` is what retires that attribute — and this is the call site that
    // justifies the claim. The visible difference is digits: GPT-4 chops them into groups
    // of three, GPT-2 does not.
    let gpt4 = Tokenizer::with_config(GPT4_SPLIT_PATTERN, Normalizer::None);
    let gpt2 = Tokenizer::with_config(GPT2_SPLIT_PATTERN, Normalizer::None);

    assert_eq!(gpt4.split("2026"), vec!["202", "6"]);
    assert_eq!(gpt2.split("2026"), vec!["2026"]);
}

#[test]
fn every_normalizer_variant_is_constructible_and_trainable() {
    // Proves the enum and all three variants are exported, not just the type name.
    for normalizer in [Normalizer::None, Normalizer::Nfc, Normalizer::Nfd] {
        let mut tok = Tokenizer::with_config(GPT4_SPLIT_PATTERN, normalizer);
        tok.train(&corpus(), 300, false);
        assert_eq!(tok.decode(&tok.encode_ordinary("Hello world")).unwrap(), "Hello world");
    }
}

#[test]
fn the_normalizer_stage_can_be_used_on_its_own() {
    // `normalize` is public, so a caller can run the pipeline's first stage by itself —
    // useful for understanding what the tokenizer will actually see.
    assert_eq!(Normalizer::Nfc.normalize("e\u{0301}"), "\u{00e9}");
    assert_eq!(Normalizer::Nfd.normalize("\u{00e9}"), "e\u{0301}");
    assert_eq!(Normalizer::None.normalize("e\u{0301}"), "e\u{0301}");
}

#[test]
fn an_untrained_tokenizer_is_still_a_working_tokenizer() {
    // No merges learned, so encode is just the bytes — but there is no out-of-vocabulary
    // case even so, because all 256 byte tokens exist from construction.
    let tok = Tokenizer::new();
    assert_eq!(tok.encode_ordinary("hi"), vec![104, 105]);
    for s in ROUND_TRIP_CASES {
        assert_eq!(*s, tok.decode(&tok.encode_ordinary(s)).unwrap());
    }
}

#[test]
fn training_the_same_corpus_twice_gives_the_same_tokenizer() {
    // The unit test for this compares `merges` and `vocab` directly. From out here those
    // fields are invisible, so determinism has to be observed through behaviour instead:
    // two independently trained tokenizers must encode identically.
    let (a, b) = (trained(), trained());
    for s in ROUND_TRIP_CASES {
        assert_eq!(a.encode_ordinary(s), b.encode_ordinary(s), "training was not deterministic for {s:?}");
    }
    assert_eq!(a.encode_ordinary(&corpus()), b.encode_ordinary(&corpus()));
}

#[test]
fn a_tokenizer_survives_a_trip_through_the_filesystem() {
    // The workflow this whole step exists for: train once, save, and later load it back
    // in a different process without retraining. The unit test for this compares `merges`
    // and `vocab` directly; from out here only behaviour is visible, which is arguably the
    // stronger claim — the tables are a means, identical encoding is the end.
    let dir = std::env::temp_dir().join("naive_bbpe_api_tests");
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = dir.join("api_roundtrip");

    let original = trained();
    original.save(&prefix).unwrap();

    let loaded = Tokenizer::load(&prefix.with_extension("model")).unwrap();
    assert_eq!(original.vocab_size(), loaded.vocab_size());
    for s in ROUND_TRIP_CASES {
        assert_eq!(original.encode_ordinary(s), loaded.encode_ordinary(s), "encode differs for {s:?}");
        assert_eq!(*s, loaded.decode(&loaded.encode_ordinary(s)).unwrap(), "round trip failed for {s:?}");
    }

    // `.vocab` is written for humans and never read back, so its only contract is that it
    // exists and describes the same number of tokens.
    let vocab = std::fs::read_to_string(prefix.with_extension("vocab")).unwrap();
    assert_eq!(vocab.lines().count(), 4 + loaded.vocab_size());
}

#[test]
fn loading_a_corrupt_model_is_an_error_rather_than_a_panic() {
    let dir = std::env::temp_dir().join("naive_bbpe_api_tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("api_corrupt.model");
    std::fs::write(&path, "this is not a model file\n").unwrap();

    let e = Tokenizer::load(&path).unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn special_tokens_are_usable_through_the_public_api() {
    // Proves `AllowedSpecial`, `register_special_tokens`, `special_tokens` and the
    // two-argument `encode` are all actually exported and work together.
    let mut tok = trained();
    let eot = tok.vocab_size() as u32;
    tok.register_special_tokens(&[("<|endoftext|>".to_string(), eot)]).unwrap();

    assert_eq!(tok.special_tokens(), vec![("<|endoftext|>".to_string(), eot)]);

    let text = "doc one<|endoftext|>doc two";
    let ids = tok.encode(text, AllowedSpecial::All).unwrap();
    assert!(ids.contains(&eot));
    assert_eq!(tok.decode(&ids).unwrap(), text);

    // The other two policies, from outside.
    assert_eq!(tok.encode(text, AllowedSpecial::None).unwrap(), tok.encode_ordinary(text));
    assert!(tok.encode(text, AllowedSpecial::NoneRaise).is_err());
}

#[test]
fn special_tokens_survive_the_filesystem_too() {
    let dir = std::env::temp_dir().join("naive_bbpe_api_tests");
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = dir.join("api_specials");

    let mut tok = trained();
    let eot = tok.vocab_size() as u32;
    tok.register_special_tokens(&[("<|endoftext|>".to_string(), eot)]).unwrap();
    tok.save(&prefix).unwrap();

    let loaded = Tokenizer::load(&prefix.with_extension("model")).unwrap();
    assert_eq!(loaded.special_tokens(), tok.special_tokens());

    let text = "a<|endoftext|>b";
    assert_eq!(
        loaded.encode(text, AllowedSpecial::All).unwrap(),
        tok.encode(text, AllowedSpecial::All).unwrap()
    );
    assert_eq!(loaded.decode(&loaded.encode(text, AllowedSpecial::All).unwrap()).unwrap(), text);
}
