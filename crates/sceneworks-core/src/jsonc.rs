//! Minimal JSONC (`.jsonc`) comment stripping for the manifest readers shared by
//! the rust-api and the rust-worker (sc-4279 / F-MLXW-15). Both crates read the
//! same `.jsonc` model/LoRA manifests, so the stripper lives here once rather
//! than byte-identically in each.

/// Strip `//` line and `/* */` block comments from a JSONC string, leaving the
/// JSON otherwise byte-for-byte (newlines inside line comments are preserved so
/// downstream error spans keep their line numbers). String literals — including
/// escaped quotes — are passed through untouched so a `//` inside a string is not
/// mistaken for a comment.
pub fn strip_jsonc_comments(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;
    while let Some(character) = chars.next() {
        if in_string {
            output.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }
        if character == '"' {
            in_string = true;
            output.push(character);
            continue;
        }
        if character == '/' && chars.peek() == Some(&'/') {
            chars.next();
            for next in chars.by_ref() {
                if next == '\r' || next == '\n' {
                    output.push(next);
                    break;
                }
            }
            continue;
        }
        if character == '/' && chars.peek() == Some(&'*') {
            chars.next();
            let mut previous = '\0';
            for next in chars.by_ref() {
                if previous == '*' && next == '/' {
                    break;
                }
                previous = next;
            }
            continue;
        }
        output.push(character);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::strip_jsonc_comments;

    #[test]
    fn strips_line_and_block_comments_but_preserves_strings() {
        let input = r#"{
            // a line comment
            "url": "https://example.com", /* trailing block */
            "note": "a // b /* c */ d"
        }"#;
        let stripped = strip_jsonc_comments(input);
        // Comments gone, but the string literal (incl. its // and /* */) survives.
        assert!(!stripped.contains("a line comment"));
        assert!(!stripped.contains("trailing block"));
        assert!(stripped.contains(r#""note": "a // b /* c */ d""#));
        assert!(stripped.contains(r#""url": "https://example.com""#));
        // Still valid JSON after stripping.
        let parsed: serde_json::Value = serde_json::from_str(&stripped).expect("valid json");
        assert_eq!(parsed["url"], "https://example.com");
    }
}
