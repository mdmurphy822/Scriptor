//! Routing policy for GPU JPEG decoding: JPEG header inspection and the
//! per-image choice between the nvJPEG device path and the libjpeg-turbo path.
//!
//! The device decoder itself lives in `scriptor-kernels`; this module is host-only.

use crate::error::ImageError;

/// Environment variable that turns the GPU JPEG path on when set to `1` or `true`.
pub const NVJPEG_ENV: &str = "SCRIPTOR_NVJPEG";

/// Chroma subsampling of a JPEG, from the component sampling factors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Subsampling {
    /// One component.
    Gray,
    /// 4:4:4, no chroma reduction.
    S444,
    /// 4:2:2, chroma halved horizontally.
    S422,
    /// 4:2:0, chroma halved in both directions.
    S420,
    /// 4:4:0, chroma halved vertically.
    S440,
    /// 4:1:1, chroma quartered horizontally.
    S411,
    /// 4:1:0, chroma quartered horizontally and halved vertically.
    S410,
    /// Any other sampling layout, or a component count other than 1 or 3.
    Other,
}

/// The frame parameters of a JPEG read from its start-of-frame marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JpegHeader {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of colour components.
    pub components: u8,
    /// Bits per sample.
    pub precision: u8,
    /// Whether the frame is progressive (SOF2).
    pub progressive: bool,
    /// Whether the entropy coding is arithmetic (SOF9 and above).
    pub arithmetic: bool,
    /// The chroma subsampling.
    pub subsampling: Subsampling,
}

/// Reads the start-of-frame parameters of a JPEG without decoding it.
pub fn parse_header(bytes: &[u8]) -> Result<JpegHeader, ImageError> {
    let bad = |detail: &str| ImageError::Decode {
        format: "jpeg",
        detail: detail.to_string(),
    };
    if !bytes.starts_with(&[0xFF, 0xD8]) {
        return Err(bad("missing SOI marker"));
    }
    let mut pos = 2usize;
    loop {
        while bytes.get(pos) == Some(&0xFF) && bytes.get(pos + 1) == Some(&0xFF) {
            pos += 1;
        }
        if bytes.get(pos) != Some(&0xFF) {
            return Err(bad("marker expected"));
        }
        let marker = *bytes.get(pos + 1).ok_or_else(|| bad("truncated marker"))?;
        pos += 2;
        if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        if marker == 0xD9 || marker == 0xDA {
            return Err(bad("no frame header before scan data"));
        }
        let len = match bytes.get(pos..pos + 2) {
            Some(l) => usize::from(u16::from_be_bytes([l[0], l[1]])),
            None => return Err(bad("truncated segment")),
        };
        if len < 2 {
            return Err(bad("invalid segment length"));
        }
        let is_sof = matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xC8 | 0xCC);
        if !is_sof {
            pos += len;
            continue;
        }
        let seg = bytes
            .get(pos + 2..pos + len)
            .ok_or_else(|| bad("truncated frame header"))?;
        if seg.len() < 6 {
            return Err(bad("short frame header"));
        }
        let precision = seg[0];
        let height = u32::from(u16::from_be_bytes([seg[1], seg[2]]));
        let width = u32::from(u16::from_be_bytes([seg[3], seg[4]]));
        let components = seg[5];
        if seg.len() < 6 + 3 * usize::from(components) {
            return Err(bad("short component table"));
        }
        if width == 0 || height == 0 {
            return Err(ImageError::InvalidDimensions { width, height });
        }
        let sampling: Vec<(u8, u8)> = (0..usize::from(components))
            .map(|c| (seg[7 + 3 * c] >> 4, seg[7 + 3 * c] & 0x0F))
            .collect();
        return Ok(JpegHeader {
            width,
            height,
            components,
            precision,
            progressive: matches!(marker, 0xC2 | 0xC6 | 0xCA | 0xCE),
            arithmetic: marker >= 0xC9,
            subsampling: classify(&sampling),
        });
    }
}

fn classify(sampling: &[(u8, u8)]) -> Subsampling {
    match sampling {
        [_] => Subsampling::Gray,
        [(yh, yv), (bh, bv), (rh, rv)] if (bh, bv) == (rh, rv) && *bh > 0 && *bv > 0 => {
            if yh % bh != 0 || yv % bv != 0 {
                return Subsampling::Other;
            }
            match (yh / bh, yv / bv) {
                (1, 1) => Subsampling::S444,
                (2, 1) => Subsampling::S422,
                (2, 2) => Subsampling::S420,
                (1, 2) => Subsampling::S440,
                (4, 1) => Subsampling::S411,
                (4, 2) => Subsampling::S410,
                _ => Subsampling::Other,
            }
        }
        _ => Subsampling::Other,
    }
}

/// Which decoder handles one image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// Decode on the device with nvJPEG.
    Gpu,
    /// Decode on the host with libjpeg-turbo, for the stated reason.
    Cpu(&'static str),
}

/// The runtime switch and the per-image eligibility rules for the GPU JPEG path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuJpegPolicy {
    /// Whether the GPU path may be used at all.
    pub enabled: bool,
    /// Subsampling modes that must stay on the CPU path.
    pub cpu_only: Vec<Subsampling>,
    /// Whether progressive frames stay on the CPU path.
    pub progressive_on_cpu: bool,
}

impl GpuJpegPolicy {
    /// The policy with the GPU path off; every image routes to the CPU.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            cpu_only: Vec::new(),
            progressive_on_cpu: false,
        }
    }

    /// The policy with the GPU path on: only modes recorded as pixel-identical to
    /// libjpeg-turbo route to the device, every other mode stays on the CPU.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            cpu_only: CPU_ONLY_SUBSAMPLING.to_vec(),
            progressive_on_cpu: PROGRESSIVE_ON_CPU,
        }
    }

    /// The policy with every structurally decodable image routed to the device,
    /// including modes that are not pixel-identical to libjpeg-turbo.
    pub fn unverified() -> Self {
        Self {
            enabled: true,
            cpu_only: Vec::new(),
            progressive_on_cpu: false,
        }
    }

    /// The policy selected by [`NVJPEG_ENV`]: `1` or `true` selects
    /// [`GpuJpegPolicy::enabled`], `unverified` selects [`GpuJpegPolicy::unverified`],
    /// anything else (or unset) leaves the GPU path off.
    pub fn from_env() -> Self {
        Self::from_value(std::env::var(NVJPEG_ENV).ok().as_deref())
    }

    /// The policy for one value of [`NVJPEG_ENV`].
    pub fn from_value(value: Option<&str>) -> Self {
        match value {
            Some("1") | Some("true") => Self::enabled(),
            Some("unverified") => Self::unverified(),
            _ => Self::disabled(),
        }
    }

    /// Chooses the decoder for an image with the given header.
    pub fn route(&self, header: &JpegHeader) -> Route {
        if !self.enabled {
            return Route::Cpu("gpu jpeg decode disabled");
        }
        if header.arithmetic {
            return Route::Cpu("arithmetic coding");
        }
        if header.precision != 8 {
            return Route::Cpu("non-8-bit precision");
        }
        if !matches!(header.components, 1 | 3) {
            return Route::Cpu("component count");
        }
        if header.progressive && self.progressive_on_cpu {
            return Route::Cpu("progressive frame not pixel-identical");
        }
        if self.cpu_only.contains(&header.subsampling) {
            return Route::Cpu("subsampling not pixel-identical");
        }
        Route::Gpu
    }
}

/// Subsampling modes whose nvJPEG output was measured to differ from libjpeg-turbo:
/// every mode, so the enabled policy routes nothing to the device.
pub const CPU_ONLY_SUBSAMPLING: &[Subsampling] = &[
    Subsampling::Gray,
    Subsampling::S444,
    Subsampling::S422,
    Subsampling::S420,
    Subsampling::S440,
    Subsampling::S411,
    Subsampling::S410,
    Subsampling::Other,
];

/// Whether progressive frames were measured to differ from libjpeg-turbo.
pub const PROGRESSIVE_ON_CPU: bool = true;

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/image/modes")
            .join(name);
        std::fs::read(path).expect("fixture")
    }

    #[test]
    fn headers_classify_every_fixture_mode() {
        let cases = [
            ("baseline_444.jpg", Subsampling::S444, false, 96, 64),
            ("baseline_422.jpg", Subsampling::S422, false, 96, 64),
            ("baseline_420.jpg", Subsampling::S420, false, 96, 64),
            ("progressive_444.jpg", Subsampling::S444, true, 96, 64),
            ("progressive_420.jpg", Subsampling::S420, true, 96, 64),
            ("gray.jpg", Subsampling::Gray, false, 96, 64),
            ("odd_420.jpg", Subsampling::S420, false, 67, 45),
            ("odd_gray.jpg", Subsampling::Gray, false, 67, 45),
            ("exif_orient6_420.jpg", Subsampling::S420, false, 96, 64),
        ];
        for (name, subsampling, progressive, width, height) in cases {
            let h = parse_header(&fixture(name)).expect(name);
            assert_eq!(h.subsampling, subsampling, "{name}");
            assert_eq!(h.progressive, progressive, "{name}");
            assert_eq!((h.width, h.height), (width, height), "{name}");
        }
    }

    #[test]
    fn malformed_headers_are_rejected() {
        assert!(parse_header(b"").is_err());
        assert!(parse_header(b"\x89PNG").is_err());
        assert!(parse_header(&[0xFF, 0xD8, 0xFF, 0xDA, 0, 2]).is_err());
        let full = fixture("baseline_420.jpg");
        assert!(parse_header(&full[..30]).is_err());
    }

    #[test]
    fn classify_covers_sampling_layouts() {
        assert_eq!(classify(&[(1, 2), (1, 1), (1, 1)]), Subsampling::S440);
        assert_eq!(classify(&[(4, 1), (1, 1), (1, 1)]), Subsampling::S411);
        assert_eq!(classify(&[(2, 2), (1, 1), (2, 1)]), Subsampling::Other);
        assert_eq!(classify(&[(1, 1); 4]), Subsampling::Other);
    }

    #[test]
    fn disabled_policy_routes_everything_to_cpu() {
        let h = parse_header(&fixture("baseline_420.jpg")).unwrap();
        assert!(matches!(GpuJpegPolicy::disabled().route(&h), Route::Cpu(_)));
    }

    #[test]
    fn environment_values_select_policies() {
        assert!(!GpuJpegPolicy::from_value(None).enabled);
        assert!(!GpuJpegPolicy::from_value(Some("0")).enabled);
        assert!(!GpuJpegPolicy::from_value(Some("yes please")).enabled);
        assert_eq!(
            GpuJpegPolicy::from_value(Some("1")),
            GpuJpegPolicy::enabled()
        );
        assert_eq!(
            GpuJpegPolicy::from_value(Some("true")),
            GpuJpegPolicy::enabled()
        );
        assert_eq!(
            GpuJpegPolicy::from_value(Some("unverified")),
            GpuJpegPolicy::unverified()
        );
    }

    #[test]
    fn enabled_policy_keeps_every_measured_mode_on_cpu() {
        for name in [
            "baseline_444.jpg",
            "baseline_422.jpg",
            "baseline_420.jpg",
            "gray.jpg",
            "progressive_420.jpg",
        ] {
            let h = parse_header(&fixture(name)).unwrap();
            assert!(
                matches!(GpuJpegPolicy::enabled().route(&h), Route::Cpu(_)),
                "{name}"
            );
            assert_eq!(GpuJpegPolicy::unverified().route(&h), Route::Gpu, "{name}");
        }
    }

    #[test]
    fn enabled_policy_applies_eligibility_rules() {
        let base = parse_header(&fixture("baseline_420.jpg")).unwrap();
        let mut policy = GpuJpegPolicy::unverified();
        policy.cpu_only = vec![Subsampling::S440];
        policy.progressive_on_cpu = true;
        assert_eq!(policy.route(&base), Route::Gpu);
        let mut h = base;
        h.subsampling = Subsampling::S440;
        assert!(matches!(policy.route(&h), Route::Cpu(_)));
        let mut h = base;
        h.progressive = true;
        assert!(matches!(policy.route(&h), Route::Cpu(_)));
        let mut h = base;
        h.arithmetic = true;
        assert!(matches!(policy.route(&h), Route::Cpu(_)));
        let mut h = base;
        h.precision = 12;
        assert!(matches!(policy.route(&h), Route::Cpu(_)));
    }
}
