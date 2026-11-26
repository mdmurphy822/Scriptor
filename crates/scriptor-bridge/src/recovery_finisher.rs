//! The page finisher that applies the text-layer, figure-text, blank-band
//! and form-field recoveries to each page's built regions, before region
//! ids, furniture page-number splitting and the title are settled.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use crate::wire::Region;
use image::RgbImage;

use crate::assemble::BuiltPage;
use crate::client::OcrClient;
use crate::crop::BBOX_EXTENT;
use crate::format::PageResult;
use crate::raster;
use crate::recover::{
    BandReadError, BandReader, Cover, FormScan, RecoverError, RecoveryCounts, RereadReader,
    blank_band, pdf_form_fields, pdf_metadata_title, read_band, recover_band_above_heading,
    recover_blank_band, recover_figure_text, recover_repetition, recover_short_recognition,
    reread_region, short_recognition_bands, uncovered_band_above_heading, with_form_fields,
};
use crate::render::{PageSource, RenderedPage};
use crate::schedule::{FinishError, PageFinisher};
use crate::text_layer_rules::{PageFinish, PageReader, finish_regions};

#[derive(Default)]
struct Totals {
    recovery: RecoveryCounts,
    page: PageFinish,
    figure_covered: usize,
}

/// Applies, per page and in order: text-layer finishing of every region,
/// figure-text recovery, blank-band and above-heading band recovery through
/// the seat, and form-field insertion. Counters from every page merge into
/// the run's diagnostics.
pub struct RecoveryFinisher<'a> {
    client: &'a OcrClient,
    model: String,
    source: Arc<Vec<u8>>,
    metadata_title: String,
    forms: FormScan,
    totals: Mutex<Totals>,
    band_reads: AtomicUsize,
    band_read_ns: AtomicU64,
}

impl<'a> RecoveryFinisher<'a> {
    /// A finisher for the document `source` (already opened as `pages`),
    /// reading bands through `client` with request model `model`. The form
    /// fields of the whole document are listed once here; a field whose
    /// page cannot be read for covering is treated as covered.
    pub fn new(
        client: &'a OcrClient,
        model: &str,
        source: Arc<Vec<u8>>,
        pages: &PageSource,
    ) -> Self {
        let dimensions: Option<Vec<(f64, f64)>> = (0..pages.page_count())
            .map(|index| pages.page_pixel_size(index).ok())
            .collect();
        let forms = match dimensions {
            Some(dimensions) => scan_forms(&source, &dimensions),
            None => FormScan::default(),
        };
        Self {
            client,
            model: model.to_string(),
            metadata_title: pdf_metadata_title(&source),
            source,
            forms,
            totals: Mutex::new(Totals::default()),
            band_reads: AtomicUsize::new(0),
            band_read_ns: AtomicU64::new(0),
        }
    }
}

/// The key of a band in a prefetch table: the bit patterns of its edges.
fn band_key(band: [f64; 4]) -> [u64; 4] {
    band.map(f64::to_bits)
}

/// Runs the blank-band, above-heading, runaway-repetition and
/// short-recognition recoveries over `regions`. Every band those recoveries read follows from `raw` alone, so
/// all of them are read concurrently through `reader` first, one thread per
/// distinct band; each recovery then takes its band's prefetched result, and
/// a band that was not prefetched is read through `reader` on demand. A
/// prefetched cancellation surfaces from the recovery that uses it.
fn recover_with_prefetch(
    regions: Vec<Region>,
    raw: &PageResult,
    page_number: i64,
    dimensions: (f64, f64),
    reader: BandReader<'_>,
    rereader: RereadReader<'_>,
    recovery: &mut RecoveryCounts,
) -> Result<Vec<Region>, RecoverError> {
    let mut bands: Vec<[f64; 4]> = Vec::new();
    let wanted = blank_band(&raw.regions)
        .into_iter()
        .chain(uncovered_band_above_heading(&raw.regions))
        .chain(short_recognition_bands(&raw.regions, dimensions));
    for band in wanted {
        if !bands.iter().any(|held| band_key(*held) == band_key(band)) {
            bands.push(band);
        }
    }
    let fetched: HashMap<[u64; 4], Result<String, BandReadError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = bands
            .iter()
            .map(|band| scope.spawn(move || (band_key(*band), reader(page_number, *band))))
            .collect();
        handles
            .into_iter()
            .map(|handle| match handle.join() {
                Ok(entry) => entry,
                Err(panic) => std::panic::resume_unwind(panic),
            })
            .collect()
    });
    let band_reader = |page: i64, band: [f64; 4]| match fetched.get(&band_key(band)) {
        Some(result) => result.clone(),
        None => reader(page, band),
    };
    recover_blank_band(
        regions,
        &raw.regions,
        page_number,
        dimensions,
        &band_reader,
        recovery,
    )
    .and_then(|regions| {
        recover_band_above_heading(
            regions,
            &raw.regions,
            page_number,
            dimensions,
            &band_reader,
            recovery,
        )
    })
    .and_then(|regions| {
        recover_repetition(
            regions,
            &raw.regions,
            page_number,
            dimensions,
            rereader,
            recovery,
        )
    })
    .and_then(|regions| {
        recover_short_recognition(
            regions,
            &raw.regions,
            page_number,
            dimensions,
            &band_reader,
            recovery,
        )
    })
}

/// The form fields of `source`, each judged covered or not on a reader that
/// leaves annotation appearances undrawn; a page without a reader is
/// `Cover::Unknown`.
fn scan_forms(source: &Arc<Vec<u8>>, dimensions: &[(f64, f64)]) -> FormScan {
    let pdf = raster::load_shared(Arc::clone(source)).ok();
    let stand_in = RenderedPage {
        image: RgbImage::new(1, 1),
        width_pt: 1.0,
        height_pt: 1.0,
        scale: 1.0,
        pixel_size: (1.0, 1.0),
    };
    let readers = RefCell::new(HashMap::new());
    let cover = |index: usize, rect: [f64; 4]| {
        let Some(pdf) = pdf.as_ref() else {
            return Cover::Unknown;
        };
        let mut readers = readers.borrow_mut();
        match readers
            .entry(index)
            .or_insert_with(|| PageReader::open(pdf, index, &stand_in, false))
        {
            Some(reader) if reader.user_rect_covered(rect) => Cover::Covered,
            Some(_) => Cover::Clear,
            None => Cover::Unknown,
        }
    };
    pdf_form_fields(source, dimensions, &cover)
}

impl PageFinisher for RecoveryFinisher<'_> {
    fn finish_built(
        &self,
        page_index: usize,
        rendered: &RenderedPage,
        raw: &PageResult,
        dimensions: (f64, f64),
        built: &mut BuiltPage,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<(), FinishError> {
        let page_number = page_index as i64 + 1;
        let mut recovery = RecoveryCounts::default();
        let mut page = PageFinish::default();
        let mut figure_covered = 0usize;
        let pdf = raster::load_shared(Arc::clone(&self.source)).ok();
        let reader = pdf
            .as_ref()
            .and_then(|pdf| PageReader::open(pdf, page_index, rendered, true));
        if let Some(reader) = &reader {
            page = finish_regions(reader, page_number, dimensions, &mut built.regions);
            built.refresh_title();
        }
        let mut regions = std::mem::take(&mut built.regions);
        if let Some(reader) = &reader {
            let mut layers: BTreeMap<usize, String> = BTreeMap::new();
            for (index, region) in regions.iter().enumerate() {
                let (Some(bbox), "figure") = (region.bbox, region.kind.as_str()) else {
                    continue;
                };
                let layer = reader.read([
                    bbox[0] / dimensions.0 * BBOX_EXTENT,
                    bbox[1] / dimensions.1 * BBOX_EXTENT,
                    bbox[2] / dimensions.0 * BBOX_EXTENT,
                    bbox[3] / dimensions.1 * BBOX_EXTENT,
                ]);
                figure_covered += layer.covered_chars;
                layers.insert(index, layer.text);
            }
            if !layers.is_empty() {
                regions = recover_figure_text(regions, &layers, &mut recovery);
            }
        }
        let live_reader = |_: i64, band: [f64; 4]| {
            let started = Instant::now();
            let result = read_band(self.client, &self.model, rendered, band, cancelled);
            let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            self.band_reads.fetch_add(1, Ordering::Relaxed);
            self.band_read_ns.fetch_add(nanos, Ordering::Relaxed);
            result
        };
        let live_rereader = |_: i64, band: [f64; 4], mode| {
            let started = Instant::now();
            let result = reread_region(self.client, &self.model, rendered, band, mode, cancelled);
            let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            self.band_reads.fetch_add(1, Ordering::Relaxed);
            self.band_read_ns.fetch_add(nanos, Ordering::Relaxed);
            result
        };
        let recovered = recover_with_prefetch(
            regions,
            raw,
            page_number,
            dimensions,
            &live_reader,
            &live_rereader,
            &mut recovery,
        );
        regions = match recovered {
            Ok(regions) => regions,
            Err(RecoverError::Cancelled) => return Err(FinishError::Cancelled),
            Err(RecoverError::Geometry(error)) => {
                return Err(FinishError::Failed(error.to_string()));
            }
        };
        if let Some(fields) = self.forms.fields.get(&page_number) {
            regions = with_form_fields(regions, fields, page_number);
        }
        built.regions = regions;
        let mut totals = self.totals.lock().unwrap_or_else(PoisonError::into_inner);
        totals.recovery.add(&recovery);
        totals.page.substituted += page.substituted;
        totals.page.covered_skipped += page.covered_skipped;
        totals.figure_covered += figure_covered;
        Ok(())
    }

    fn metadata_title(&self) -> Option<String> {
        Some(self.metadata_title.clone())
    }

    fn diagnostics(&self) -> Vec<(&'static str, usize)> {
        let totals = self.totals.lock().unwrap_or_else(PoisonError::into_inner);
        let mut entries = Vec::new();
        if totals.page.substituted > 0 {
            entries.push(("text_layer_substituted", totals.page.substituted));
        }
        let covered = totals.page.covered_skipped + totals.figure_covered;
        if covered > 0 {
            entries.push(("text_layer_covered_skipped", covered));
        }
        entries.extend(totals.recovery.entries());
        entries.extend(self.forms.entries());
        let reads = self.band_reads.load(Ordering::Relaxed);
        if reads > 0 {
            entries.push(("recovery_band_reads", reads));
            let millis = self.band_read_ns.load(Ordering::Relaxed) / 1_000_000;
            entries.push(("recovery_band_read_ms", millis as usize));
        }
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assemble::assemble;
    use crate::format::{RegionResult, RegionTask};
    use std::sync::Condvar;
    use std::time::Duration;

    const PAGE: (f64, f64) = (1000.0, 2000.0);

    fn text(content: &str, bbox: [f64; 4]) -> RegionResult {
        RegionResult {
            label: "text".to_string(),
            task: RegionTask::Text,
            bbox: Some(bbox),
            content: Some(content.to_string()),
            repetition_stopped: false,
        }
    }

    /// A page with two short recognitions and a blank text region between
    /// recognised ones.
    fn page() -> PageResult {
        let mut regions: Vec<RegionResult> = (0..5)
            .map(|index| {
                let top = 100.0 + 70.0 * f64::from(index);
                text(
                    &format!("Body {index} lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore"),
                    [100.0, top, 900.0, top + 60.0],
                )
            })
            .collect();
        for index in 0..2 {
            let top = 460.0 + 180.0 * f64::from(index);
            regions.push(text(
                &format!("Short {index} a.\nShort {index} b."),
                [100.0, top, 900.0, top + 160.0],
            ));
        }
        PageResult { regions }
    }

    fn no_reread(_: i64, _: [f64; 4], _: crate::recover::Reread) -> Result<String, BandReadError> {
        Err(BandReadError::Failed)
    }

    fn answer(band: [f64; 4]) -> String {
        format!("Shared line\nOnly at {}", band[1])
    }

    fn run(raw: &PageResult, reader: BandReader<'_>) -> (Vec<Region>, RecoveryCounts) {
        let regions = assemble(std::slice::from_ref(raw), &[PAGE], "")
            .expect("assembles")
            .regions;
        let mut counts = RecoveryCounts::default();
        let regions = recover_with_prefetch(regions, raw, 1, PAGE, reader, &no_reread, &mut counts)
            .expect("recovers");
        (regions, counts)
    }

    #[test]
    fn the_prefetch_reads_every_band_concurrently_with_the_sequential_result() {
        let raw = page();
        let wanted = short_recognition_bands(&raw.regions, PAGE).len();
        assert_eq!(wanted, 2);
        let state = (Mutex::new(0usize), Condvar::new());
        let overlapped = Mutex::new(Vec::new());
        let reads = AtomicUsize::new(0);
        let waiting = |_: i64, band: [f64; 4]| {
            reads.fetch_add(1, Ordering::SeqCst);
            let (count, wake) = &state;
            let mut in_flight = count.lock().unwrap();
            *in_flight += 1;
            wake.notify_all();
            let deadline = Instant::now() + Duration::from_secs(30);
            while *in_flight < wanted {
                let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                    break;
                };
                in_flight = wake.wait_timeout(in_flight, left).unwrap().0;
            }
            overlapped.lock().unwrap().push(*in_flight >= wanted);
            Ok(answer(band))
        };
        let (regions, counts) = run(&raw, &waiting);
        assert_eq!(*overlapped.lock().unwrap(), vec![true; wanted]);
        assert_eq!(reads.load(Ordering::SeqCst), wanted);
        let plain = |_: i64, band: [f64; 4]| Ok(answer(band));
        let (sequential, sequential_counts) = run(&raw, &plain);
        assert_eq!(regions, sequential);
        assert_eq!(counts, sequential_counts);
    }

    #[test]
    fn a_prefetched_cancellation_surfaces_from_the_recovery() {
        let raw = page();
        let regions = assemble(std::slice::from_ref(&raw), &[PAGE], "")
            .expect("assembles")
            .regions;
        let cancelled = |_: i64, _: [f64; 4]| Err(BandReadError::Cancelled);
        let mut counts = RecoveryCounts::default();
        let outcome =
            recover_with_prefetch(regions, &raw, 1, PAGE, &cancelled, &no_reread, &mut counts);
        assert_eq!(outcome, Err(RecoverError::Cancelled));
    }
}
