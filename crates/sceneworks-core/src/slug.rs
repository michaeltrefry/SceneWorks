pub fn slugify(value: &str, fallback: &str, max_length: Option<usize>) -> String {
    let mut slug = String::new();
    let mut previous_dash = false;
    for character in value.trim().chars() {
        if character.is_ascii_alphanumeric() {
            slug.push(character.to_ascii_lowercase());
            previous_dash = false;
        } else if !previous_dash && !slug.is_empty() {
            slug.push('-');
            previous_dash = true;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        slug = fallback.to_owned();
    }
    if let Some(max_length) = max_length {
        slug.truncate(max_length);
    }
    slug
}
