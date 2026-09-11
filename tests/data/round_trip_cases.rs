// Inputs that must survive encode -> decode unchanged.
//
// NOT a Rust file in the normal sense: it is a bare array literal with no items, no
// `fn`, no `const`. Your editor will flag it as broken Rust, and that is expected.
// It is pasted textually into whoever `include!`s it:
//
//     src/lib.rs    const ROUND_TRIP_CASES: &[&str] = include!("../tests/data/round_trip_cases.rs");
//     tests/api.rs  const ROUND_TRIP_CASES: &[&str] = include!("data/round_trip_cases.rs");
//
// Those are two separate crates — the library and the integration-test crate — so the
// alternative to this was either duplicating 29 string literals or making the list a
// `pub const`, i.e. shipping test fixtures as part of the library's public API.
//
// It lives in `tests/data/` and not `tests/` because Cargo compiles every `.rs` file
// placed DIRECTLY in `tests/` as its own integration-test crate. Subdirectories are
// ignored, which is also why the `tests/common/mod.rs` convention exists.
&[
    // the originals
    "Hello world",
    "aaabdaaabac",
    "Ｕｎｉｃｏｄｅ! 😄",
    "é",
    "",
    // scripts the training text barely covers — these still round-trip because
    // byte-level BPE has no out-of-vocabulary case; they just compress badly
    "שלום עולם",                    // Hebrew, right-to-left
    "नमस्ते दुनिया",                    // Devanagari, with combining vowel signs
    "안녕하세요 세계",                 // Hangul, precomposed syllables
    "こんにちは、世界！",              // Japanese + fullwidth punctuation
    "Ωμέγα ΣΊΣΥΦΟΣ",               // Greek, mixed case
    // one grapheme, several code points
    "👨‍👩‍👧‍👦",                          // ZWJ family sequence — 4 emoji + 3 joiners
    "🇮🇳🇯🇵",                          // regional-indicator flag pairs
    "e\u{0301}\u{0327}",             // "e" + two stacked combining marks
    "Z\u{0351}\u{036b}a\u{0343}l\u{0489}g\u{0317}o", // Zalgo: marks piled on ASCII
    // whitespace and line endings, which the split pattern has dedicated arms for
    "line1\r\nline2\n\nline3",
    "   leading and trailing   ",
    "\ttab\tseparated\tvalues\t",
    "\u{feff}text after a byte-order mark",
    "\u{00a0}non-breaking\u{2009}thin\u{3000}ideographic spaces",
    // contractions and digits — the two fiddliest arms of the GPT-4 pattern
    "It's the 'best' — isn't it? I'd've said so.",
    "IT'S SHOUTING BUT THE PATTERN IS CASE-INSENSITIVE",
    "3.14159 2026-09-05 1_000_000 0xDEADBEEF 1e-9",
    "1234567890 12345678901234567890",
    // code, which is mostly punctuation runs
    "let x: Vec<u32> = (0..256).map(|i| i as u8).collect();",
    "a==b&&c!=d||e<=f",
    // adversarial shapes
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", // long run — merges compound on themselves
    " ",
    "\u{0}\u{1}\u{7f}",                 // control bytes
    "🐈",                               // a single 4-byte character alone
]
