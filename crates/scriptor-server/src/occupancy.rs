//! Prometheus text for the decode-step slot occupancy counters.

use std::fmt::Write as _;

use scriptor_engine::{BucketOccupancy, SlotOccupancy};

/// Reads one per-bucket series value.
type BucketValue = fn(&BucketOccupancy) -> f64;

/// Appends the occupancy counters under `labels`: step and row totals, then per-bucket step
/// counts, the steps each bucket would serve under a gap-free slot layout, and device time.
pub fn render(out: &mut String, labels: &str, occ: &SlotOccupancy) {
    for (name, value) in [
        ("scriptor_decode_step_shapes_total", occ.steps),
        ("scriptor_decode_active_rows_total", occ.active_rows),
        ("scriptor_decode_span_rows_total", occ.span_rows),
        ("scriptor_decode_bucket_rows_total", occ.bucket_rows),
        ("scriptor_decode_compact_rows_total", occ.compact_rows),
        ("scriptor_decode_padded_rows_total", occ.padded_rows()),
        ("scriptor_decode_sparse_rows_total", occ.sparse_rows()),
        ("scriptor_decode_padded_steps_total", occ.padded_steps),
        ("scriptor_decode_sparse_steps_total", occ.sparse_steps),
    ] {
        let _ = writeln!(out, "# TYPE {name} counter");
        let _ = writeln!(out, "{name}{{{labels}}} {value}");
    }
    for (name, label, hist) in [
        ("scriptor_decode_span_steps_total", "span", &occ.span_steps),
        (
            "scriptor_decode_active_steps_total",
            "active",
            &occ.active_steps,
        ),
    ] {
        let _ = writeln!(out, "# TYPE {name} counter");
        for (value, steps) in hist.0.iter().enumerate() {
            let _ = writeln!(out, "{name}{{{labels},{label}=\"{value}\"}} {steps}");
        }
    }
    let entries = occ.bucket_entries();
    let series: [(&str, BucketValue); 4] = [
        ("scriptor_decode_bucket_steps_total", |b| b.steps as f64),
        ("scriptor_decode_compact_bucket_steps_total", |b| {
            b.compact_steps as f64
        }),
        ("scriptor_decode_bucket_timed_steps_total", |b| {
            b.timed_steps as f64
        }),
        ("scriptor_decode_bucket_device_seconds_total", |b| {
            b.timed_nanos as f64 / 1e9
        }),
    ];
    for (name, value) in series {
        let _ = writeln!(out, "# TYPE {name} counter");
        for b in &entries {
            let _ = writeln!(out, "{name}{{{labels},bucket=\"{}\"}} {}", b.rows, value(b));
        }
    }
}

#[cfg(test)]
mod tests {
    use scriptor_spec::{graph_buckets, step_shape};

    use super::*;

    #[test]
    fn occupancy_counters_render_with_bucket_labels() {
        let buckets = graph_buckets(32);
        let mut occ = SlotOccupancy::default();
        occ.record(&step_shape(&buckets, [31]).unwrap());
        occ.record(&step_shape(&buckets, 0..32).unwrap());
        occ.record_time(32, 2_500_000);
        let mut out = String::new();
        render(&mut out, "model_name=\"m\"", &occ);
        assert!(out.contains("scriptor_decode_step_shapes_total{model_name=\"m\"} 2\n"));
        assert!(out.contains("scriptor_decode_padded_rows_total{model_name=\"m\"} 31\n"));
        assert!(out.contains("scriptor_decode_sparse_rows_total{model_name=\"m\"} 31\n"));
        assert!(out.contains("scriptor_decode_bucket_rows_total{model_name=\"m\"} 64\n"));
        assert!(
            out.contains("scriptor_decode_bucket_steps_total{model_name=\"m\",bucket=\"32\"} 2\n")
        );
        assert!(out.contains(
            "scriptor_decode_compact_bucket_steps_total{model_name=\"m\",bucket=\"1\"} 1\n"
        ));
        assert!(out.contains("scriptor_decode_span_steps_total{model_name=\"m\",span=\"32\"} 2\n"));
        assert!(out.contains("scriptor_decode_span_steps_total{model_name=\"m\",span=\"5\"} 0\n"));
        assert!(
            out.contains("scriptor_decode_active_steps_total{model_name=\"m\",active=\"1\"} 1\n")
        );
        assert!(
            out.contains("scriptor_decode_active_steps_total{model_name=\"m\",active=\"32\"} 1\n")
        );
        assert_eq!(out.matches("scriptor_decode_span_steps_total{").count(), 33);
        assert!(out.contains(
            "scriptor_decode_bucket_device_seconds_total{model_name=\"m\",bucket=\"32\"} 0.0025\n"
        ));
    }
}
