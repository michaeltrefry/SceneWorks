use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::Path;

use url::{Host, Url};

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum LoraUrlError {
    Invalid,
    UnsupportedScheme,
    MissingFilename,
    UnsafeFilename,
    BlockedHost,
}

impl LoraUrlError {
    pub fn message(&self) -> &'static str {
        match self {
            Self::Invalid => "LoRA sourceUrl must be a valid URL",
            Self::UnsupportedScheme => "LoRA sourceUrl must use http or https",
            Self::MissingFilename => "LoRA sourceUrl must include a filename",
            Self::UnsafeFilename => {
                "LoRA sourceUrl filename must use letters, numbers, dots, dashes, or underscores"
            }
            Self::BlockedHost => "LoRA sourceUrl host is not allowed",
        }
    }
}

pub fn parse_lora_source_url(source_url: &str) -> Result<Url, LoraUrlError> {
    parse_lora_source_url_with_private(source_url, false)
}

pub fn parse_lora_source_url_with_private(
    source_url: &str,
    allow_private_hosts: bool,
) -> Result<Url, LoraUrlError> {
    let url = Url::parse(source_url).map_err(|_| LoraUrlError::Invalid)?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(LoraUrlError::UnsupportedScheme);
    }
    if !allow_private_hosts {
        validate_lora_url_host(&url)?;
    }
    lora_source_url_file_name(source_url)?;
    Ok(url)
}

pub fn lora_source_url_file_name(source_url: &str) -> Result<String, LoraUrlError> {
    let url = Url::parse(source_url).map_err(|_| LoraUrlError::Invalid)?;
    let file_name = url
        .path_segments()
        .and_then(Iterator::last)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or(LoraUrlError::MissingFilename)?;
    if !safe_lora_filename(file_name) {
        return Err(LoraUrlError::UnsafeFilename);
    }
    Ok(file_name.to_owned())
}

pub fn lora_source_url_file_stem(source_url: &str) -> Result<String, LoraUrlError> {
    let file_name = lora_source_url_file_name(source_url)?;
    Path::new(&file_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .map(str::to_owned)
        .filter(|value| !value.trim().is_empty())
        .ok_or(LoraUrlError::MissingFilename)
}

pub fn safe_lora_filename(file_name: &str) -> bool {
    let trimmed = file_name.trim();
    !trimmed.is_empty()
        && trimmed.len() <= 160
        && trimmed.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_')
        })
}

pub fn validate_lora_url_host(url: &Url) -> Result<(), LoraUrlError> {
    let Some(host) = url.host() else {
        return Err(LoraUrlError::BlockedHost);
    };
    match host {
        Host::Ipv4(address) => validate_public_ip(IpAddr::V4(address)),
        Host::Ipv6(address) => validate_public_ip(IpAddr::V6(address)),
        Host::Domain(domain) => {
            let domain = domain.trim_end_matches('.').to_ascii_lowercase();
            if domain == "localhost" || domain.ends_with(".localhost") || domain.ends_with(".local")
            {
                return Err(LoraUrlError::BlockedHost);
            }
            Ok(())
        }
    }
}

pub fn validate_public_ip(address: IpAddr) -> Result<(), LoraUrlError> {
    let blocked = match address {
        IpAddr::V4(address) => blocked_ipv4(address),
        IpAddr::V6(address) => blocked_ipv6(address),
    };
    if blocked {
        Err(LoraUrlError::BlockedHost)
    } else {
        Ok(())
    }
}

fn blocked_ipv4(address: Ipv4Addr) -> bool {
    address.is_private()
        || address.is_loopback()
        || address.is_link_local()
        || address.is_broadcast()
        || address.is_documentation()
        || address.octets()[0] == 0
        || address.octets()[0] >= 224
}

fn blocked_ipv6(address: Ipv6Addr) -> bool {
    address.is_loopback()
        || address.is_unspecified()
        || address.is_unique_local()
        || address.is_unicast_link_local()
        || (address.segments()[0] == 0x2001 && address.segments()[1] == 0x0db8)
        || address.is_multicast()
}

#[cfg(test)]
mod tests {
    use super::{lora_source_url_file_name, parse_lora_source_url, LoraUrlError};

    #[test]
    fn lora_source_urls_validate_scheme_host_and_filename() {
        assert_eq!(
            lora_source_url_file_name("https://example.com/models/style.safetensors").unwrap(),
            "style.safetensors"
        );
        assert_eq!(
            parse_lora_source_url("file:///tmp/style.safetensors").unwrap_err(),
            LoraUrlError::UnsupportedScheme
        );
        assert_eq!(
            parse_lora_source_url("https://example.com/").unwrap_err(),
            LoraUrlError::MissingFilename
        );
        assert_eq!(
            parse_lora_source_url("http://127.0.0.1/style.safetensors").unwrap_err(),
            LoraUrlError::BlockedHost
        );
        assert_eq!(
            parse_lora_source_url("https://example.com/style.safetensors%00.txt").unwrap_err(),
            LoraUrlError::UnsafeFilename
        );
    }
}
