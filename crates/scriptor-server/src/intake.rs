//! Byte-bounded request intake: every in-flight request holds a reservation sized by its
//! body and the decoded payload of its images until its response ends.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use scriptor_image::{
    GridThw, PATCH_ELEMS, SmartResizeConfig, data_url_prefix, image_header, smart_resize_with,
};

/// Share of host `MemAvailable` the intake of request bodies and decoded image payloads may hold.
pub const INTAKE_HOST_SHARE: f64 = 1.0 / 16.0;

/// Bytes of the decoded patch payload one image occupies at `max_pixels`
/// (`[patches, 1176]` bf16).
pub fn image_payload_bytes(max_pixels: u64) -> u64 {
    let patch_pixels = (scriptor_image::PATCH_SIZE * scriptor_image::PATCH_SIZE) as u64;
    max_pixels.div_ceil(patch_pixels) * PATCH_ELEMS as u64 * 2
}

/// Bytes of an image data URL decoded to read its header.
const HEADER_PREFIX_BYTES: usize = 64 << 10;

/// Bytes the preprocessor's `[patches, 1176]` bf16 payload occupies for an image of
/// `width` x `height` under `resize`; `max_pixels` payload when the size cannot be resized.
fn resized_payload_bytes(width: u32, height: u32, resize: &SmartResizeConfig) -> u64 {
    match smart_resize_with(height, width, resize) {
        Ok((h, w)) => {
            let patches = GridThw::for_image(h, w).patches() as u64;
            patches * PATCH_ELEMS as u64 * 2
        }
        Err(_) => image_payload_bytes(resize.max_pixels),
    }
}

/// Bytes the decoded payload of the image at `url` occupies: read from the header within
/// the first 64 KiB of a base64 data URL, never above the `max_pixels` payload, and
/// the `max_pixels` payload when the header cannot be read.
pub fn image_charge(url: &str, resize: &SmartResizeConfig) -> u64 {
    let worst = image_payload_bytes(resize.max_pixels);
    let Some(prefix) = data_url_prefix(url, HEADER_PREFIX_BYTES) else {
        return worst;
    };
    match image_header(&prefix) {
        Ok((_, width, height)) => resized_payload_bytes(width, height, resize).min(worst),
        Err(_) => worst,
    }
}

/// The derived intake budget and its inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntakeBudget {
    /// Host `MemAvailable` the budget derives from.
    pub mem_available: u64,
    /// Bytes the intake may hold: `INTAKE_HOST_SHARE` of `mem_available`.
    pub bytes: u64,
    /// Bytes one worst-case request reserves (largest body plus its decoded images).
    pub per_request: u64,
}

impl IntakeBudget {
    /// Derives the budget from `mem_available` and the worst-case request size.
    pub fn derive(mem_available: u64, per_request: u64) -> Self {
        let bytes = (mem_available as f64 * INTAKE_HOST_SHARE) as u64;
        Self {
            mem_available,
            bytes,
            per_request,
        }
    }

    /// One log line stating the derivation.
    pub fn describe(&self) -> String {
        format!(
            "intake budget {} B = {:.4} x MemAvailable {} B; worst-case request {} B",
            self.bytes, INTAKE_HOST_SHARE, self.mem_available, self.per_request
        )
    }
}

/// Why a reservation was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The request alone exceeds the whole budget.
    TooLarge {
        /// Bytes the request needs.
        cost: u64,
        /// The whole budget.
        budget: u64,
    },
    /// The budget is held by other requests; retry after the derived delay.
    Full {
        /// Bytes the request needs.
        cost: u64,
        /// Bytes currently free.
        free: u64,
        /// The whole budget.
        budget: u64,
        /// Seconds after which a slot is expected to free.
        retry_after: u64,
    },
}

#[derive(Default)]
struct Ledger {
    used: u64,
    in_flight: u64,
    completed: u64,
    held: Duration,
}

/// The byte ledger shared by all handlers.
#[derive(Clone)]
pub struct Intake {
    budget: u64,
    ledger: Arc<Mutex<Ledger>>,
}

impl Intake {
    /// Creates a ledger over `budget` bytes.
    pub fn new(budget: u64) -> Self {
        Self {
            budget,
            ledger: Arc::default(),
        }
    }

    /// Bytes the ledger allows in flight.
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// Bytes currently reserved.
    pub fn used(&self) -> u64 {
        self.lock().used
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Ledger> {
        self.ledger.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Reserves `cost` bytes. `slots` is the engine's concurrent stream count, used to
    /// derive `Retry-After` as the mean measured hold time divided by `slots`;
    /// `fallback_retry` applies before any request has completed.
    pub fn reserve(
        &self,
        cost: u64,
        slots: u64,
        fallback_retry: u64,
    ) -> Result<Reservation, Refusal> {
        if cost > self.budget {
            return Err(Refusal::TooLarge {
                cost,
                budget: self.budget,
            });
        }
        let mut ledger = self.lock();
        if ledger.used + cost > self.budget {
            let retry_after = if ledger.completed == 0 {
                fallback_retry
            } else {
                let mean = ledger.held.as_secs_f64() / ledger.completed as f64;
                (mean / slots.max(1) as f64).ceil() as u64
            };
            return Err(Refusal::Full {
                cost,
                free: self.budget - ledger.used,
                budget: self.budget,
                retry_after: retry_after.max(1),
            });
        }
        ledger.used += cost;
        ledger.in_flight += 1;
        Ok(Reservation {
            cost,
            since: Instant::now(),
            ledger: Arc::clone(&self.ledger),
        })
    }
}

/// Bytes held in the intake ledger until dropped.
pub struct Reservation {
    cost: u64,
    since: Instant,
    ledger: Arc<Mutex<Ledger>>,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut ledger = self.ledger.lock().unwrap_or_else(|e| e.into_inner());
        ledger.used = ledger.used.saturating_sub(self.cost);
        ledger.in_flight = ledger.in_flight.saturating_sub(1);
        ledger.completed += 1;
        ledger.held += self.since.elapsed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_derives_from_available_memory() {
        let b = IntakeBudget::derive(16 << 30, 128 << 20);
        assert_eq!(b.bytes, 1 << 30);
        assert_eq!(b.per_request, 128 << 20);
        assert!(b.describe().contains("worst-case request 134217728 B"));
    }

    #[test]
    fn reservation_is_released_on_drop() {
        let intake = Intake::new(100);
        let a = intake.reserve(60, 1, 2).unwrap();
        match intake.reserve(60, 1, 2) {
            Err(Refusal::Full {
                free, retry_after, ..
            }) => assert_eq!((free, retry_after), (40, 2)),
            _ => panic!("expected a full ledger"),
        }
        drop(a);
        assert_eq!(intake.used(), 0);
        assert!(intake.reserve(60, 1, 2).is_ok());
    }

    #[test]
    fn oversized_request_is_too_large() {
        let intake = Intake::new(100);
        assert!(matches!(
            intake.reserve(101, 1, 1),
            Err(Refusal::TooLarge {
                cost: 101,
                budget: 100
            })
        ));
    }

    #[test]
    fn retry_after_follows_measured_hold_time() {
        let intake = Intake::new(10);
        let a = intake.reserve(10, 2, 1).unwrap();
        std::thread::sleep(Duration::from_millis(30));
        drop(a);
        let _held = intake.reserve(10, 2, 1).unwrap();
        match intake.reserve(1, 2, 1) {
            Err(Refusal::Full { retry_after, .. }) => assert_eq!(retry_after, 1),
            _ => panic!("expected a full ledger"),
        }
    }

    fn png_url(width: u32, height: u32) -> String {
        let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13];
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[8, 2, 0, 0, 0]);
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        bytes.resize(bytes.len().div_ceil(3) * 3, 0);
        let b64: String = bytes
            .chunks(3)
            .flat_map(|c| {
                let n = u32::from(c[0]) << 16 | u32::from(c[1]) << 8 | u32::from(c[2]);
                (0..4).map(move |i| ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char)
            })
            .collect();
        format!("data:image/png;base64,{b64}")
    }

    #[test]
    fn small_image_is_charged_its_real_payload() {
        let resize = SmartResizeConfig::default();
        // 280 x 560 is already a multiple of 28: 20 x 40 patches of 1176 bf16 elements.
        let charge = image_charge(&png_url(280, 560), &resize);
        assert_eq!(charge, 20 * 40 * PATCH_ELEMS as u64 * 2);
        assert!(charge < image_payload_bytes(resize.max_pixels));
    }

    #[test]
    fn oversized_image_is_charged_at_most_the_max_pixels_payload() {
        let resize = SmartResizeConfig::default();
        let worst = image_payload_bytes(resize.max_pixels);
        let charge = image_charge(&png_url(8000, 8000), &resize);
        assert!(charge <= worst);
    }

    #[test]
    fn unreadable_header_is_charged_the_max_pixels_payload() {
        let resize = SmartResizeConfig::default();
        let worst = image_payload_bytes(resize.max_pixels);
        assert_eq!(image_charge("data:image/png;base64,AAAA", &resize), worst);
        assert_eq!(image_charge("data:image/png;base64,@@@@", &resize), worst);
    }

    #[test]
    fn many_small_reservations_fit_where_the_worst_case_charge_refuses() {
        let resize = SmartResizeConfig::default();
        let worst = image_payload_bytes(resize.max_pixels);
        let intake = Intake::new(worst * 2);
        let small = image_charge(&png_url(280, 280), &resize);
        let held: Vec<_> = (0..32)
            .map(|_| intake.reserve(small, 4, 1).expect("small image fits"))
            .collect();
        assert_eq!(held.len(), 32);
        let strict = Intake::new(worst * 2);
        let _a = strict.reserve(worst, 4, 1).unwrap();
        let _b = strict.reserve(worst, 4, 1).unwrap();
        assert!(matches!(
            strict.reserve(worst, 4, 1),
            Err(Refusal::Full { .. })
        ));
    }
}
