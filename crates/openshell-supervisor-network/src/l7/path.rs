// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP request-target canonicalization for L7 policy enforcement.
//!
//! The L7 REST proxy evaluates OPA rules against the request path and
//! forwards the raw request line to the upstream server. If the path the
//! policy sees is not the path the upstream dispatches on, any path-based
//! allow rule can be bypassed with non-canonical encodings (`..`, `%2e%2e`,
//! `//`, `;params`). This module resolves that divergence by producing one
//! canonical segment structure that is both the input to policy evaluation
//! and the bytes written onto the wire. The policy and wire forms differ only
//! in reserved characters the client percent-encoded (such as `%3A`): policy
//! sees them decoded, and the wire keeps them encoded so a client signature
//! over the path (for example AWS SigV4) still verifies upstream.
//!
//! Behavior for v1:
//! - Percent-decode unreserved path bytes; preserve the rest as uppercase
//!   `%HH` on the wire. Reserved sub-delimiters, `:`, and `@` are decoded
//!   for policy evaluation only.
//! - Resolve `.` and `..` segments per RFC 3986 Section 5.2.4. `..` that
//!   would escape the root is rejected rather than silently clamped to
//!   `/` — non-canonical input is almost always adversarial.
//! - Collapse repeated slashes.
//! - Reject control bytes (`0x00..=0x1F`, `0x7F`), fragments in the
//!   request-target, raw non-ASCII bytes, and paths that cannot be parsed
//!   as origin-form.
//! - Strip trailing `;params` from each segment by default (Tomcat-class
//!   `;jsessionid` ACL-bypass mitigation). Stripping happens *before*
//!   dot-segment resolution so that `..;` cannot evade the traversal guard
//!   and then revert to `..` on the way out.
//! - Reject any `.`/`..` that survives to the end of canonicalization. The
//!   policy engine trusts this function to have removed them.
//! - Reject `%2F` (encoded slash) inside a segment by default. Operators
//!   can opt in per-endpoint for APIs that rely on encoded slashes in
//!   slugs.

use thiserror::Error;

/// Reasons a request-target can be rejected at the canonicalization boundary.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CanonicalizeError {
    #[error("request-target contains a null or control byte")]
    NullOrControlByte,
    #[error("request-target contains an invalid percent-encoded sequence")]
    InvalidPercentEncoding,
    #[error("request-target contains an encoded '/' (%2F) which is not allowed on this endpoint")]
    EncodedSlashNotAllowed,
    #[error("request-target contains a fragment")]
    FragmentInRequestTarget,
    #[error("request-target contains raw non-ASCII bytes; non-ASCII must be percent-encoded")]
    NonAscii,
    #[error("request-target's `..` segment would escape the path root")]
    TraversalAboveRoot,
    #[error("request-target still contains a `.`/`..` segment after canonicalization")]
    ResidualDotSegment,
    #[error("request-target exceeds the configured maximum length")]
    PathTooLong,
    #[error("request-target is not a valid origin-form path")]
    MalformedTarget,
}

/// Options controlling canonicalization strictness.
///
/// Produced by the endpoint configuration. Defaults are intentionally strict:
/// operators opt in to looser behavior per-endpoint when the upstream API
/// requires it.
#[derive(Debug, Clone, Copy)]
pub struct CanonicalizeOptions {
    /// When `true`, `%2F` inside a segment is preserved (re-emitted as
    /// `%2F` on the wire) rather than rejected. Defaults to `false`.
    pub allow_encoded_slash: bool,
    /// When `true`, RFC 3986 path parameters (`;param`) are stripped from
    /// each segment before policy evaluation and before forwarding.
    /// Defaults to `true`: path parameters are an ambiguity surface
    /// historically used to bypass ACLs and are not part of any policy
    /// we author.
    pub strip_path_parameters: bool,
}

impl Default for CanonicalizeOptions {
    fn default() -> Self {
        Self {
            allow_encoded_slash: false,
            strip_path_parameters: true,
        }
    }
}

/// Result of a successful canonicalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalPath {
    /// The canonical path policy evaluates. Always starts with `/`. Contains
    /// no `.`/`..` segments, no doubled slashes, and no `;params` (when
    /// stripping is enabled).
    pub path: String,
    /// The canonical path to forward upstream. It has the same segments as
    /// `path` but keeps any reserved character the client percent-encoded
    /// (such as `%3A` or `%40`) encoded, so the upstream receives the bytes
    /// the client sent (and possibly signed) wherever no structural
    /// canonicalization was needed.
    pub wire_path: String,
    /// `true` if `wire_path` differs from the input. Callers use this to
    /// decide whether to rewrite the outbound request line.
    pub rewritten: bool,
}

/// Maximum accepted length of an origin-form path (bytes).
pub(crate) const MAX_PATH_LEN: usize = 4 * 1024;

/// A decoded path byte. The low 8 bits hold the byte; [`PERCENT_ENCODED`]
/// marks a reserved byte the client percent-encoded. Marked bytes never
/// compare equal to a literal `/`, `.`, or `;`, so an encoded slash cannot
/// split a segment and is kept inside it.
type PathUnit = u16;

/// Flag on a [`PathUnit`] for a reserved byte that arrived percent-encoded
/// and is re-emitted percent-encoded on the wire.
const PERCENT_ENCODED: PathUnit = 0x100;
const SLASH: PathUnit = 0x2F;
const DOT: PathUnit = 0x2E;
const SEMICOLON: PathUnit = 0x3B;
/// A `%2F` inside a segment (only produced when `allow_encoded_slash`).
const ENCODED_SLASH: PathUnit = PERCENT_ENCODED | SLASH;

/// Canonicalize an HTTP request-target's path component.
///
/// Accepts origin-form (`"/a/b?q=1"`) or absolute-form (`"http://h/a/b"`)
/// targets. Asterisk-form (`"*"`, used only for `OPTIONS *`) is rejected
/// because the L7 enforcement pipeline does not handle it.
///
/// Returns the canonical path plus the original query suffix (byte-for-byte
/// as supplied by the client). Query-parameter parsing is left to the
/// caller — this function only operates on the path component.
pub fn canonicalize_request_target(
    target: &str,
    opts: &CanonicalizeOptions,
) -> Result<(CanonicalPath, Option<String>), CanonicalizeError> {
    // 1. Reject control bytes and raw non-ASCII outright. These tests also
    //    cover CR/LF which are never legal in a single-line request-target.
    for &b in target.as_bytes() {
        if b == 0 || b == b'\n' || b == b'\r' || b == b'\t' || b == 0x7F {
            return Err(CanonicalizeError::NullOrControlByte);
        }
        if b < 0x20 {
            return Err(CanonicalizeError::NullOrControlByte);
        }
        if b >= 0x80 {
            return Err(CanonicalizeError::NonAscii);
        }
    }

    // 2. Reject fragments — forbidden in request-target per RFC 7230.
    if target.contains('#') {
        return Err(CanonicalizeError::FragmentInRequestTarget);
    }

    // 3. Split off query at the first `?`. Query is preserved verbatim.
    let (path_part, query_part) = match target.split_once('?') {
        Some((p, q)) => (p, Some(q.to_string())),
        None => (target, None),
    };

    // 4. Handle absolute-form by stripping the URI authority. Origin-form
    // targets may legitimately embed `://` in a path segment, so only a URI
    // with a scheme is absolute-form.
    let absolute_form_uri = path_part
        .parse::<http::Uri>()
        .ok()
        .filter(|uri| uri.scheme().is_some());
    let raw_path = absolute_form_uri
        .as_ref()
        .map(http::Uri::path)
        .filter(|path| !path.is_empty())
        .unwrap_or_else(|| {
            if absolute_form_uri.is_some() {
                "/"
            } else {
                path_part
            }
        });

    // 5. Empty is equivalent to "/".
    let raw_path = if raw_path.is_empty() { "/" } else { raw_path };

    // 6. Must begin with '/' (origin-form).
    if !raw_path.starts_with('/') {
        return Err(CanonicalizeError::MalformedTarget);
    }

    // 7. Length bound.
    if raw_path.len() > MAX_PATH_LEN {
        return Err(CanonicalizeError::PathTooLong);
    }

    // 8. Percent-decode the path into units. A percent-encoded reserved
    //    byte, including `%2F`, keeps a marker so that subsequent `/`
    //    splitting cannot confuse it with a real path separator and so the
    //    wire form can re-emit it encoded.
    let decoded = percent_decode(raw_path.as_bytes(), opts.allow_encoded_slash)?;

    // 9. Split on literal `/`, then strip `;params` *before* resolving
    //    dot-segments. Stripping afterwards would let a `..;` segment slip
    //    past the traversal guard (it is not byte-equal to `..`) and then
    //    revert to a bare `..` during reconstruction.
    let segments = split_path_segments(&decoded);
    let segments: Vec<&[PathUnit]> = if opts.strip_path_parameters {
        segments.into_iter().map(strip_path_parameters).collect()
    } else {
        segments
    };
    let resolved = resolve_dot_segments(segments)?;

    // 9b. Defense in depth: no `.`/`..` may survive canonicalization. The
    //     policy engine trusts this function to have removed them, so a
    //     residual dot-segment is always a bug or an attack. This also
    //     catches a `..` hidden behind a `%2F` on endpoints that opted into
    //     `allow_encoded_slash`, where the encoded slash keeps the
    //     dot-segment inside its segment and out of `resolve_dot_segments`.
    for seg in &resolved {
        for part in seg.split(|&unit| unit == ENCODED_SLASH) {
            if is_dot_segment(part) {
                return Err(CanonicalizeError::ResidualDotSegment);
            }
        }
    }

    // 10. Reconstruct the policy and wire forms from the same segments.
    let (path, wire_path) = build_canonical_paths(&resolved, *opts);

    let rewritten = wire_path != raw_path;
    Ok((
        CanonicalPath {
            path,
            wire_path,
            rewritten,
        },
        query_part,
    ))
}

/// Report whether a canonical policy path carries a `%2F` that survived
/// canonicalization.
///
/// Callers that canonicalize before they know which endpoint config applies
/// use this to re-check the result against the config that actually matched.
///
/// The test is exact: [`build_canonical_paths`] emits the literal `%2F` in the
/// policy path only for an encoded slash, and percent-encodes any other `%`
/// byte as `%25`, so no `%2F` substring can reach it by another route.
#[must_use]
pub fn canonical_path_has_encoded_slash(canonical_path: &str) -> bool {
    canonical_path.contains("%2F")
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Reserved bytes that are emitted literally in a canonical path but keep
/// their percent-encoding on the wire when the client sent them encoded.
///
/// RFC 3986 Section 6.2.2.2 permits decoding only unreserved characters
/// during normalization; decoding a reserved character can change how the
/// upstream interprets the path and invalidates any signature over the path
/// bytes (for example AWS SigV4). `;` is not listed: an encoded `%3B` still
/// decodes to a path-parameter delimiter so parameter stripping keeps
/// closing the `..;` traversal bypass. `/` is handled separately.
fn keeps_percent_encoding(byte: u8) -> bool {
    matches!(
        byte,
        b'!' | b'$' | b'&' | b'\'' | b'(' | b')' | b'*' | b'+' | b',' | b'=' | b':' | b'@'
    )
}

fn percent_decode(
    raw: &[u8],
    allow_encoded_slash: bool,
) -> Result<Vec<PathUnit>, CanonicalizeError> {
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        let b = raw[i];
        if b == b'%' {
            if i + 2 >= raw.len() {
                return Err(CanonicalizeError::InvalidPercentEncoding);
            }
            let decoded = match (decode_hex(raw[i + 1]), decode_hex(raw[i + 2])) {
                (Some(hi), Some(lo)) => (hi << 4) | lo,
                _ => return Err(CanonicalizeError::InvalidPercentEncoding),
            };
            if decoded == b'/' {
                if !allow_encoded_slash {
                    return Err(CanonicalizeError::EncodedSlashNotAllowed);
                }
                out.push(ENCODED_SLASH);
            } else if decoded == 0 || decoded == 0x7F || (decoded < 0x20 && decoded != b'\t') {
                return Err(CanonicalizeError::NullOrControlByte);
            } else if decoded == b'\n' || decoded == b'\r' || decoded == b'\t' {
                // %-encoded CR/LF/TAB are still control bytes; reject.
                return Err(CanonicalizeError::NullOrControlByte);
            } else if keeps_percent_encoding(decoded) {
                out.push(PERCENT_ENCODED | PathUnit::from(decoded));
            } else {
                out.push(PathUnit::from(decoded));
            }
            i += 3;
        } else {
            out.push(PathUnit::from(b));
            i += 1;
        }
    }
    Ok(out)
}

/// Drop a trailing `;params` suffix from a single path segment.
fn strip_path_parameters(seg: &[PathUnit]) -> &[PathUnit] {
    seg.iter()
        .position(|&unit| unit == SEMICOLON)
        .map_or(seg, |pos| &seg[..pos])
}

fn split_path_segments(decoded: &[PathUnit]) -> Vec<&[PathUnit]> {
    // decoded is guaranteed to start with `/`. Skip the leading `/` and
    // split on subsequent `/` bytes. An encoded slash never matches, so it
    // stays inside its segment.
    decoded[1..].split(|&unit| unit == SLASH).collect()
}

fn is_dot(seg: &[PathUnit]) -> bool {
    matches!(seg, [DOT])
}

fn is_dot_dot(seg: &[PathUnit]) -> bool {
    matches!(seg, [DOT, DOT])
}

fn is_dot_segment(seg: &[PathUnit]) -> bool {
    is_dot(seg) || is_dot_dot(seg)
}

fn resolve_dot_segments(
    segments: Vec<&[PathUnit]>,
) -> Result<Vec<Vec<PathUnit>>, CanonicalizeError> {
    let mut stack: Vec<Vec<PathUnit>> = Vec::with_capacity(segments.len());
    let last = segments.len().saturating_sub(1);
    for (idx, seg) in segments.into_iter().enumerate() {
        if is_dot_dot(seg) {
            if stack.pop().is_none() {
                return Err(CanonicalizeError::TraversalAboveRoot);
            }
            if idx == last {
                // A trailing `..` leaves a "directory" (trailing slash).
                stack.push(Vec::new());
            }
            continue;
        }
        if is_dot(seg) {
            if idx == last {
                stack.push(Vec::new());
            }
            continue;
        }
        if seg.is_empty() && idx != last {
            // Collapse repeated slashes except at the very end, where an
            // empty trailing segment encodes a trailing `/`.
            continue;
        }
        stack.push(seg.to_vec());
    }
    Ok(stack)
}

/// Rebuild the policy path and the wire path from the same resolved
/// segments.
///
/// Both strip `;params` per segment if requested and percent-encode any byte
/// outside the pchar set. They differ only for reserved bytes the client
/// percent-encoded: the policy path decodes them (except `%2F`), while the
/// wire path keeps them encoded.
fn build_canonical_paths(
    segments: &[Vec<PathUnit>],
    opts: CanonicalizeOptions,
) -> (String, String) {
    let mut policy = String::from("/");
    let mut wire = String::from("/");
    for (idx, seg) in segments.iter().enumerate() {
        if idx > 0 {
            policy.push('/');
            wire.push('/');
        }
        let trimmed: &[PathUnit] = if opts.strip_path_parameters {
            strip_path_parameters(seg)
        } else {
            seg
        };
        for &unit in trimmed {
            let [byte, flags] = unit.to_le_bytes();
            if flags != 0 {
                push_percent_encoded(&mut wire, byte);
                if byte == b'/' {
                    push_percent_encoded(&mut policy, byte);
                } else {
                    policy.push(char::from(byte));
                }
            } else if is_pchar_unreserved(byte) {
                policy.push(char::from(byte));
                wire.push(char::from(byte));
            } else {
                push_percent_encoded(&mut policy, byte);
                push_percent_encoded(&mut wire, byte);
            }
        }
    }
    (policy, wire)
}

fn push_percent_encoded(out: &mut String, byte: u8) {
    out.push('%');
    out.push(upper_hex_nibble(byte >> 4));
    out.push(upper_hex_nibble(byte & 0x0F));
}

fn is_pchar_unreserved(b: u8) -> bool {
    // RFC 3986 pchar without the percent-encoded slot — i.e. bytes we emit
    // literally. Unreserved plus RFC 3986 sub-delims plus `:` and `@`.
    b.is_ascii_alphanumeric()
        || matches!(
            b,
            b'-' | b'.'
                | b'_'
                | b'~'
                | b'!'
                | b'$'
                | b'&'
                | b'\''
                | b'('
                | b')'
                | b'*'
                | b'+'
                | b','
                | b';'
                | b'='
                | b':'
                | b'@'
        )
}

fn decode_hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn upper_hex_nibble(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'A' + (n - 10)) as char,
        _ => unreachable!("nibble out of range"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn canon(input: &str) -> Result<String, CanonicalizeError> {
        let opts = CanonicalizeOptions::default();
        canonicalize_request_target(input, &opts).map(|(p, _)| p.path)
    }

    fn canon_with(input: &str, opts: CanonicalizeOptions) -> Result<String, CanonicalizeError> {
        canonicalize_request_target(input, &opts).map(|(p, _)| p.path)
    }

    #[test]
    fn literal_dot_segments_resolve() {
        assert_eq!(canon("/a/./b").unwrap(), "/a/b");
        assert_eq!(canon("/a/b/.").unwrap(), "/a/b/");
        assert_eq!(canon("/a/../b").unwrap(), "/b");
        assert_eq!(canon("/a/b/..").unwrap(), "/a/");
    }

    #[test]
    fn percent_encoded_dot_segments_resolve_the_same_way() {
        assert_eq!(canon("/public/%2e%2e/secret").unwrap(), "/secret");
        assert_eq!(canon("/public/%2E%2E/secret").unwrap(), "/secret");
        assert_eq!(canon("/public/%2e/secret").unwrap(), "/public/secret");
    }

    #[test]
    fn traversal_above_root_is_rejected() {
        assert_eq!(canon("/.."), Err(CanonicalizeError::TraversalAboveRoot));
        assert_eq!(
            canon("/a/../.."),
            Err(CanonicalizeError::TraversalAboveRoot)
        );
        assert_eq!(
            canon("/a/%2e%2e/%2e%2e"),
            Err(CanonicalizeError::TraversalAboveRoot)
        );
    }

    #[test]
    fn doubled_slashes_collapse() {
        assert_eq!(canon("//").unwrap(), "/");
        assert_eq!(canon("//public//../secret").unwrap(), "/secret");
        assert_eq!(canon("/public//secret").unwrap(), "/public/secret");
    }

    #[test]
    fn encoded_slash_rejected_by_default() {
        assert_eq!(
            canon("/a/%2f/b"),
            Err(CanonicalizeError::EncodedSlashNotAllowed)
        );
        assert_eq!(
            canon("/public/..%2fsecret"),
            Err(CanonicalizeError::EncodedSlashNotAllowed)
        );
    }

    #[test]
    fn encoded_slash_preserved_when_opted_in() {
        let opts = CanonicalizeOptions {
            allow_encoded_slash: true,
            ..CanonicalizeOptions::default()
        };
        assert_eq!(canon_with("/a/%2f/b", opts).unwrap(), "/a/%2F/b");
        assert_eq!(canon_with("/a/%2F/b", opts).unwrap(), "/a/%2F/b");
    }

    #[test]
    fn null_and_control_bytes_rejected() {
        assert_eq!(canon("/a%00b"), Err(CanonicalizeError::NullOrControlByte));
        assert_eq!(canon("/a%0Ab"), Err(CanonicalizeError::NullOrControlByte));
        assert_eq!(canon("/a%0Db"), Err(CanonicalizeError::NullOrControlByte));
        assert_eq!(canon("/a%7Fb"), Err(CanonicalizeError::NullOrControlByte));
        // Raw CR/LF/TAB in input should also fail. Build strings via
        // byte-level concatenation since the literals in the source are
        // otherwise flagged as control bytes in CI.
        let mut raw = String::from("/a");
        raw.push('\n');
        raw.push('b');
        assert_eq!(canon(&raw), Err(CanonicalizeError::NullOrControlByte));
    }

    #[test]
    fn fragment_rejected() {
        assert_eq!(
            canon("/a#b"),
            Err(CanonicalizeError::FragmentInRequestTarget)
        );
    }

    #[test]
    fn absolute_form_strips_authority() {
        assert_eq!(canon("http://host/a/../b").unwrap(), "/b");
        assert_eq!(canon("https://host").unwrap(), "/");
        assert_eq!(canon("http://host:443/foo").unwrap(), "/foo");
    }

    #[test]
    fn origin_form_with_embedded_url_is_not_stripped_as_absolute_form() {
        let (path, query) = canonicalize_request_target(
            "/fetch/http://example.test?next=http://other.test",
            &CanonicalizeOptions::default(),
        )
        .expect("origin-form target with embedded URLs must be accepted");
        // Repeated slashes are canonicalized everywhere, but the embedded
        // URL must remain part of the origin-form path rather than being
        // treated as a new absolute-form authority.
        assert_eq!(path.path, "/fetch/http:/example.test");
        assert_eq!(query.as_deref(), Some("next=http://other.test"));
    }

    #[test]
    fn legitimate_percent_encoded_bytes_round_trip() {
        assert_eq!(
            canon("/files/hello%20world.txt").unwrap(),
            "/files/hello%20world.txt"
        );
        assert_eq!(canon("/search/a%3Fb").unwrap(), "/search/a%3Fb");
        assert_eq!(canon("/users/%40alice").unwrap(), "/users/@alice");
    }

    #[test]
    fn percent_encoded_reserved_bytes_are_forwarded_as_sent() {
        // Signed requests (AWS SigV4 and similar schemes) sign the exact path
        // bytes the client sends. AWS REST-JSON APIs carry resource ARNs in a
        // path segment with `:` and `/` percent-encoded. Policy evaluation may
        // see the decoded form, but an already-canonical path must reach the
        // upstream unchanged or the upstream rejects the signature.
        let opts = CanonicalizeOptions {
            allow_encoded_slash: true,
            ..CanonicalizeOptions::default()
        };
        for target in [
            "/tags/arn%3Aaws%3Aeks%3Aeu-west-2%3A123456789012%3Acluster%2Fdemo",
            "/2015-03-31/functions/arn%3Aaws%3Alambda%3Aeu-west-2%3A123456789012%3Afunction%3Ademo/policy",
            "/users/%40alice",
            "/v1/a%2Bb%3Dc%2Cd%21%24%26%27%28%29%2A",
        ] {
            let (canonical, _) = canonicalize_request_target(target, &opts).unwrap();
            assert!(
                !canonical.rewritten,
                "{target} must be forwarded byte-for-byte, policy path {}",
                canonical.path
            );
        }
    }

    #[test]
    fn policy_path_decodes_percent_encoded_reserved_bytes() {
        // Policy rules keep matching the decoded form.
        let opts = CanonicalizeOptions {
            allow_encoded_slash: true,
            ..CanonicalizeOptions::default()
        };
        assert_eq!(
            canon_with(
                "/tags/arn%3Aaws%3Aeks%3Aeu-west-2%3A123456789012%3Acluster%2Fdemo",
                opts
            )
            .unwrap(),
            "/tags/arn:aws:eks:eu-west-2:123456789012:cluster%2Fdemo"
        );
        assert_eq!(canon("/v1/a%2Bb%3Dc").unwrap(), "/v1/a+b=c");
    }

    #[test]
    fn path_parameters_stripped_by_default() {
        assert_eq!(canon("/a;jsessionid=xyz/b").unwrap(), "/a/b");
        assert_eq!(canon("/public;x=1/../secret").unwrap(), "/secret");
    }

    #[test]
    fn path_parameters_preserved_when_disabled() {
        let opts = CanonicalizeOptions {
            strip_path_parameters: false,
            ..CanonicalizeOptions::default()
        };
        assert_eq!(
            canon_with("/a;jsessionid=xyz/b", opts).unwrap(),
            "/a;jsessionid=xyz/b"
        );
    }

    #[test]
    fn non_ascii_raw_byte_rejected() {
        let mut raw = String::from("/a");
        raw.push('é');
        assert_eq!(canon(&raw), Err(CanonicalizeError::NonAscii));
    }

    #[test]
    fn percent_encoded_non_ascii_bytes_round_trip() {
        // `é` in UTF-8 is 0xC3 0xA9. The proxy treats the path as opaque
        // bytes; percent-encoded high bytes pass through unchanged.
        assert_eq!(canon("/users/caf%C3%A9").unwrap(), "/users/caf%C3%A9");
    }

    #[test]
    fn empty_and_root_equivalent() {
        assert_eq!(canon("").unwrap(), "/");
        assert_eq!(canon("/").unwrap(), "/");
    }

    #[test]
    fn path_too_long_rejected() {
        let long = format!("/{}", "a".repeat(MAX_PATH_LEN));
        assert_eq!(canon(&long), Err(CanonicalizeError::PathTooLong));
    }

    #[test]
    fn mixed_case_percent_normalizes_to_uppercase() {
        // Request comes in with lowercase %c3 — after canonicalization we
        // emit %C3 so policy authors don't need to enumerate both cases.
        assert_eq!(canon("/a/caf%c3%a9").unwrap(), "/a/caf%C3%A9");
    }

    #[test]
    fn rewritten_flag_reflects_transformation() {
        let (canon, _) =
            canonicalize_request_target("/a", &CanonicalizeOptions::default()).unwrap();
        assert!(!canon.rewritten);
        let (canon, _) =
            canonicalize_request_target("/a/../b", &CanonicalizeOptions::default()).unwrap();
        assert!(canon.rewritten);
    }

    #[test]
    fn query_suffix_is_returned_separately() {
        let (canon, query) =
            canonicalize_request_target("/a?q=1&r=2", &CanonicalizeOptions::default()).unwrap();
        assert_eq!(canon.path, "/a");
        assert_eq!(query.as_deref(), Some("q=1&r=2"));
    }

    // ---------------------------------------------------------------------
    // Regression tests for the documented attack payloads. Every one of
    // these used to bypass a `/public/**` allow rule because the proxy and
    // the OPA policy never agreed with the upstream on what path was being
    // dispatched.
    // ---------------------------------------------------------------------

    #[test]
    fn regression_public_slash_dotdot_secret() {
        assert_eq!(canon("/public/../secret").unwrap(), "/secret");
    }

    #[test]
    fn regression_public_slash_percent_dotdot_secret() {
        assert_eq!(canon("/public/%2e%2e/secret").unwrap(), "/secret");
        assert_eq!(canon("/public/%2E%2E/secret").unwrap(), "/secret");
    }

    #[test]
    fn regression_percent_encoded_slash_in_dotdot_rejected() {
        assert_eq!(
            canon("/public/%2E%2E%2Fsecret"),
            Err(CanonicalizeError::EncodedSlashNotAllowed)
        );
    }

    #[test]
    fn regression_double_slash_prefix() {
        assert_eq!(canon("//public/../secret").unwrap(), "/secret");
    }

    #[test]
    fn regression_dot_slash_dotdot() {
        assert_eq!(canon("/public/./../secret").unwrap(), "/secret");
    }

    // ---------------------------------------------------------------------
    // A `;params` suffix on the dot segment itself must not let the segment
    // slip past the traversal guard. Stripping used to run after dot-segment
    // resolution, so `..;` was not byte-equal to `..` when the guard looked
    // at it, and reverted to a bare `..` during reconstruction — handing the
    // policy engine and the upstream a path that still escaped its prefix.
    // ---------------------------------------------------------------------

    #[test]
    fn regression_dotdot_with_path_parameter_is_resolved_not_smuggled() {
        assert_eq!(canon("/public/..;/secret").unwrap(), "/secret");
        assert_eq!(canon("/public/..;x/secret").unwrap(), "/secret");
        assert_eq!(
            canon("/public/..;jsessionid=xyz/secret").unwrap(),
            "/secret"
        );
        assert_eq!(canon("/public/.;/secret").unwrap(), "/public/secret");
        assert_eq!(canon("/public/.;x/secret").unwrap(), "/public/secret");
    }

    #[test]
    fn regression_percent_encoded_dotdot_with_path_parameter() {
        // `%2e%2e;` and `..%3B` both decode to `..;` before segmentation.
        assert_eq!(canon("/public/%2e%2e;/secret").unwrap(), "/secret");
        assert_eq!(canon("/public/..%3B/secret").unwrap(), "/secret");
        assert_eq!(canon("/public/..%3b/secret").unwrap(), "/secret");
    }

    #[test]
    fn regression_chained_dotdot_with_path_parameters_hits_root_guard() {
        assert_eq!(
            canon("/public/..;/..;/secret"),
            Err(CanonicalizeError::TraversalAboveRoot)
        );
        assert_eq!(
            canon("/api/v1/..;/..;/..;/admin/keys"),
            Err(CanonicalizeError::TraversalAboveRoot)
        );
        assert_eq!(canon("/..;"), Err(CanonicalizeError::TraversalAboveRoot));
    }

    #[test]
    fn regression_dotdot_behind_encoded_slash_is_rejected_when_opted_in() {
        // With `allow_encoded_slash`, an encoded `%2F` keeps the dot-segment
        // inside its segment, so `resolve_dot_segments` never sees it. The
        // residual guard is what closes this.
        let opts = CanonicalizeOptions {
            allow_encoded_slash: true,
            ..CanonicalizeOptions::default()
        };
        assert_eq!(
            canon_with("/public/..%2fsecret", opts),
            Err(CanonicalizeError::ResidualDotSegment)
        );
        assert_eq!(
            canon_with("/public/..%2f..%2fsecret", opts),
            Err(CanonicalizeError::ResidualDotSegment)
        );
        assert_eq!(
            canon_with("/public/.%2fsecret", opts),
            Err(CanonicalizeError::ResidualDotSegment)
        );
        // A legitimate encoded-slash slug still round-trips.
        assert_eq!(
            canon_with("/repos/group%2fproject/issues", opts).unwrap(),
            "/repos/group%2Fproject/issues"
        );
    }

    #[test]
    fn encoded_slash_detection_on_canonical_paths_is_exact() {
        let opts = CanonicalizeOptions {
            allow_encoded_slash: true,
            ..CanonicalizeOptions::default()
        };

        // A surviving encoded slash is detected.
        let slug = canon_with("/repos/group%2fproject/issues", opts).unwrap();
        assert_eq!(slug, "/repos/group%2Fproject/issues");
        assert!(canonical_path_has_encoded_slash(&slug));

        // Ordinary paths are not.
        assert!(!canonical_path_has_encoded_slash(
            &canon("/public/secret").unwrap()
        ));

        // A literal `%` in the input is re-emitted as `%25`, so it cannot
        // fabricate a `%2F` substring — including when the input spells out
        // `%252F`, which decodes to the three characters `%`, `2`, `F`.
        let escaped = canon("/a/%252F/b").unwrap();
        assert_eq!(escaped, "/a/%252F/b");
        assert!(!canonical_path_has_encoded_slash(&escaped));

        let percent = canon("/a/100%25/b").unwrap();
        assert_eq!(percent, "/a/100%25/b");
        assert!(!canonical_path_has_encoded_slash(&percent));
    }

    #[test]
    fn canonical_output_never_contains_dot_segments() {
        // The contract the policy engine relies on: whatever comes back is
        // free of `.`/`..`, so rego never has to defend against them.
        for target in [
            "/public/..;/secret",
            "/public/.;/secret",
            "/public/%2e%2e;/secret",
            "/a;jsessionid=xyz/b",
            "/public//../secret",
            "/a/b/..",
            "/a/b/.",
        ] {
            if let Ok(path) = canon(target) {
                assert!(
                    !path.split('/').any(|seg| seg == ".." || seg == "."),
                    "{target} canonicalized to {path}, which still has a dot-segment"
                );
            }
        }
    }
}
