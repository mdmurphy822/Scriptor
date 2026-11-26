//! The blocking HTTP client constructor for requests to the recognition
//! engine.
//!
//! Redirects are never followed. Loopback targets bypass any proxy; every
//! other target honours the `ALL_PROXY`, `HTTP_PROXY`, `HTTPS_PROXY` and
//! `NO_PROXY` environment variables. Certificate verification is never
//! relaxed. Requests to an `https` target trust the operating system's
//! certificate store in addition to the built-in roots.

use std::time::Duration;

use reqwest::Url;
use reqwest::blocking::Client;
use reqwest::redirect::Policy;

/// How requests to one target are routed with respect to proxies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProxyRule {
    /// The target is on this computer; no proxy is used.
    Bypass,
    /// The environment proxy variables decide.
    Environment,
}

/// Timeouts applied to a client; `None` leaves that limit unset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HttpProfile {
    pub timeout: Option<Duration>,
    pub connect_timeout: Option<Duration>,
}

impl HttpProfile {
    /// A profile limiting the whole request to `timeout`.
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            timeout: Some(timeout),
            connect_timeout: None,
        }
    }
}

/// Whether `url` names `localhost` or a loopback address (127.0.0.0/8, `::1`).
/// An unparsable URL or one without a host is not loopback.
pub fn is_loopback_target(url: &str) -> bool {
    let Ok(parsed) = Url::parse(url.trim()) else {
        return false;
    };
    let Some(host) = parsed.host_str() else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(address) = host.parse::<std::net::IpAddr>() {
        return address.is_loopback();
    }
    let name = host.trim_end_matches('.').to_ascii_lowercase();
    name == "localhost" || name.ends_with(".localhost")
}

/// The proxy rule for requests to `url`.
pub fn proxy_rule(url: &str) -> ProxyRule {
    if is_loopback_target(url) {
        ProxyRule::Bypass
    } else {
        ProxyRule::Environment
    }
}

/// Why a client could not be built.
#[derive(Debug)]
pub enum ClientError {
    /// The client could not be built.
    Build(reqwest::Error),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Build(error) => write!(formatter, "the HTTP client cannot be built: {error}"),
        }
    }
}

impl std::error::Error for ClientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Build(error) => Some(error),
        }
    }
}

/// Builds a blocking client for requests to `target` under `profile`.
pub fn http_client_for(target: &str, profile: HttpProfile) -> Result<Client, ClientError> {
    client_builder(target, profile)
        .build()
        .map_err(ClientError::Build)
}

/// The additional root certificates for requests to `target`: the operating
/// system's for an `https` target.
fn trusted_roots(target: &str) -> Vec<reqwest::Certificate> {
    if is_https_target(target) {
        os_root_certificates().to_vec()
    } else {
        Vec::new()
    }
}

/// The operating system's trusted root certificates, read once per process;
/// empty when the store cannot be read.
fn os_root_certificates() -> &'static [reqwest::Certificate] {
    static ROOTS: std::sync::OnceLock<Vec<reqwest::Certificate>> = std::sync::OnceLock::new();
    ROOTS.get_or_init(|| {
        rustls_native_certs::load_native_certs()
            .certs
            .iter()
            .filter_map(|der| reqwest::Certificate::from_der(der.as_ref()).ok())
            .collect()
    })
}

/// Whether `url` is an `https` URL.
fn is_https_target(url: &str) -> bool {
    Url::parse(url.trim()).is_ok_and(|parsed| parsed.scheme() == "https")
}

fn client_builder(target: &str, profile: HttpProfile) -> reqwest::blocking::ClientBuilder {
    let mut builder = Client::builder()
        .redirect(Policy::none())
        .timeout(profile.timeout);
    for certificate in trusted_roots(target) {
        builder = builder.add_root_certificate(certificate);
    }
    if let Some(connect) = profile.connect_timeout {
        builder = builder.connect_timeout(connect);
    }
    if proxy_rule(target) == ProxyRule::Bypass {
        builder = builder.no_proxy();
    }
    builder
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_hosts_bypass_the_proxy() {
        for url in [
            "http://127.0.0.1:8000/v1",
            "http://127.9.9.9/v1",
            "http://[::1]:8000/",
            "http://localhost:8000/",
            "http://LOCALHOST./",
        ] {
            assert_eq!(proxy_rule(url), ProxyRule::Bypass, "{url}");
        }
    }

    #[test]
    fn remote_and_unparsable_targets_use_the_environment_proxy() {
        for url in [
            "https://api.example.com/v1",
            "http://10.0.0.5:8000/",
            "http://localhost.example.com/",
            "not a url",
        ] {
            assert_eq!(proxy_rule(url), ProxyRule::Environment, "{url}");
        }
    }

    #[test]
    fn clients_build_for_both_rules_without_following_redirects() {
        let profile = HttpProfile::with_timeout(Duration::from_secs(1));
        assert!(http_client_for("http://127.0.0.1:1/", profile).is_ok());
        assert!(http_client_for("https://api.example.com/", profile).is_ok());
    }
}
