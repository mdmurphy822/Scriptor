use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use scriptor_weights::{
    EngineFile, EngineFileError, HeapStaging, LayerPrefetcher, SafetensorsDir, Transform, convert,
    fused_name, infer_layout,
};

fn tiny_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/model/tiny")
}

/// Copies the tiny checkpoint into a fresh directory so tests may modify it.
fn scratch_checkpoint() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("snapshots").join("rev0");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(
        tiny_dir().join("model.safetensors"),
        dir.join("model.safetensors"),
    )
    .unwrap();
    (tmp, dir)
}

fn convert_tiny(dir: &Path, out: &Path) -> EngineFile {
    convert(dir, out).unwrap();
    EngineFile::open(out).unwrap()
}

#[test]
fn round_trip_bytes_match_the_repack_plan() {
    let (tmp, dir) = scratch_checkpoint();
    let out = tmp.path().join("tiny.oeng");
    let report = convert(&dir, &out).unwrap();
    let engine = EngineFile::open(&out).unwrap();
    let src = SafetensorsDir::open(&dir).unwrap();
    let layout = infer_layout(&src).unwrap();
    assert_eq!((layout.text_layers, layout.vision_blocks), (2, 2));
    assert_eq!(engine.header().revision, "rev0");
    assert_eq!(report.tensors, engine.entries().len());
    assert_eq!(report.out_bytes, std::fs::metadata(&out).unwrap().len());

    let mut consumed = 0;
    for e in engine.entries() {
        let mut expected = Vec::new();
        let mut shapes = Vec::new();
        for s in &e.sources {
            let t = src.get(s).unwrap();
            expected.extend_from_slice(src.bytes(t));
            shapes.push(t.shape.clone());
            consumed += 1;
        }
        assert_eq!(engine.bytes(e), expected.as_slice(), "{}", e.name);
        assert_eq!(e.offset % 4096, 0, "{}", e.name);
        assert_eq!(e.len as usize, e.shape.iter().product::<usize>() * 2);
        match e.transform {
            Transform::Copy => assert_eq!(e.shape, shapes[0]),
            Transform::Concat => {
                assert!(e.sources.len() > 1);
                assert_eq!(e.shape[0], shapes.iter().map(|s| s[0]).sum::<usize>());
                assert_eq!(e.shape[1..], shapes[0][1..]);
            }
            Transform::Reshape => {
                assert_eq!(e.shape.len(), 2);
                assert_eq!(e.shape[0], shapes[0][0]);
            }
        }
    }
    assert_eq!(consumed, src.tensors().len());
}

#[test]
fn fused_and_flattened_layouts() {
    let (tmp, dir) = scratch_checkpoint();
    let engine = convert_tiny(&dir, &tmp.path().join("t.oeng"));
    let q = engine
        .entry(&fused_name(scriptor_weights::FusedParam::TextQkv {
            layer: 0,
        }))
        .unwrap();
    // 4 heads x 12 plus 2 x (2 kv heads x 12) rows over hidden 48.
    assert_eq!(q.shape, vec![48 + 24 + 24, 48]);
    let gu = engine
        .entry(&fused_name(scriptor_weights::FusedParam::VisionGateUp {
            block: 1,
            kind: scriptor_weights::Kind::Bias,
        }))
        .unwrap();
    assert_eq!(gu.shape, vec![128]);
    let pe = engine
        .entry("model.visual.patch_embed.proj.weight")
        .unwrap();
    assert_eq!(pe.shape, vec![32, 1176]);
    let ds = engine.entry("model.visual.downsample.weight").unwrap();
    assert_eq!(ds.shape, vec![48, 32 * 4]);
    assert!(
        engine
            .entry("model.language_model.layers.0.self_attn.q_proj.weight")
            .is_err()
    );
}

#[test]
fn groups_follow_file_order_and_cover_every_entry() {
    let (tmp, dir) = scratch_checkpoint();
    let engine = convert_tiny(&dir, &tmp.path().join("t.oeng"));
    let groups = engine.layer_groups();
    let labels: Vec<&str> = groups.iter().map(|g| g.label.as_str()).collect();
    assert_eq!(
        labels,
        [
            "vision.patch_embed",
            "vision.block.0",
            "vision.block.1",
            "vision.adapter",
            "text.embed",
            "text.layer.0",
            "text.layer.1",
            "text.head",
            "mtp"
        ]
    );
    let total: usize = groups.iter().map(|g| g.entries.len()).sum();
    assert_eq!(total, engine.entries().len());
}

#[test]
fn prefetcher_delivers_every_tensor_in_order() {
    let (tmp, dir) = scratch_checkpoint();
    let engine = Arc::new(convert_tiny(&dir, &tmp.path().join("t.oeng")));
    let loaded: Vec<_> = LayerPrefetcher::start(engine.clone(), Arc::new(HeapStaging), 3, 2)
        .collect::<Result<_, _>>()
        .unwrap();
    let mut seen = BTreeMap::new();
    for g in &loaded {
        for (meta, buf) in &g.tensors {
            assert_eq!(buf.as_slice(), engine.bytes(meta), "{}", meta.name);
            seen.insert(meta.name.clone(), ());
        }
    }
    assert_eq!(seen.len(), engine.entries().len());
    assert_eq!(loaded.len(), engine.layer_groups().len());
}

#[test]
fn verify_accepts_the_unchanged_source() {
    let (tmp, dir) = scratch_checkpoint();
    let engine = convert_tiny(&dir, &tmp.path().join("t.oeng"));
    engine.verify_source(&dir).unwrap();
    engine.verify_source_contents(&dir).unwrap();
}

#[test]
fn source_size_change_is_refused() {
    let (tmp, dir) = scratch_checkpoint();
    let engine = convert_tiny(&dir, &tmp.path().join("t.oeng"));
    let shard = dir.join("model.safetensors");
    let mut bytes = std::fs::read(&shard).unwrap();
    bytes.push(0);
    std::fs::write(&shard, bytes).unwrap();
    let err = engine.verify_source(&dir).unwrap_err();
    assert!(
        matches!(&err, EngineFileError::SourceMismatch(m) if m.iter().any(|l| l.contains("size")))
    );
}

#[test]
fn source_content_hash_mismatch_is_refused_even_with_same_size_and_mtime() {
    let (tmp, dir) = scratch_checkpoint();
    let engine = convert_tiny(&dir, &tmp.path().join("t.oeng"));
    let shard = dir.join("model.safetensors");
    let mtime = std::fs::metadata(&shard).unwrap().modified().unwrap();
    let mut bytes = std::fs::read(&shard).unwrap();
    *bytes.last_mut().unwrap() ^= 0xff;
    std::fs::write(&shard, bytes).unwrap();
    OpenOptions::new()
        .write(true)
        .open(&shard)
        .unwrap()
        .set_modified(mtime)
        .unwrap();
    engine.verify_source(&dir).unwrap();
    let err = engine.verify_source_contents(&dir).unwrap_err();
    assert!(
        matches!(&err, EngineFileError::SourceMismatch(m) if m.iter().any(|l| l.contains("content hash")))
    );
}

#[test]
fn source_mtime_change_is_refused() {
    let (tmp, dir) = scratch_checkpoint();
    let engine = convert_tiny(&dir, &tmp.path().join("t.oeng"));
    let shard = dir.join("model.safetensors");
    let mtime = std::fs::metadata(&shard).unwrap().modified().unwrap();
    OpenOptions::new()
        .write(true)
        .open(&shard)
        .unwrap()
        .set_modified(mtime + Duration::from_secs(5))
        .unwrap();
    let err = engine.verify_source(&dir).unwrap_err();
    assert!(
        matches!(&err, EngineFileError::SourceMismatch(m) if m.iter().any(|l| l.contains("mtime")))
    );
}

#[test]
fn added_and_missing_shards_are_refused() {
    let (tmp, dir) = scratch_checkpoint();
    let engine = convert_tiny(&dir, &tmp.path().join("t.oeng"));
    std::fs::copy(dir.join("model.safetensors"), dir.join("extra.safetensors")).unwrap();
    assert!(matches!(
        engine.verify_source(&dir),
        Err(EngineFileError::SourceMismatch(m)) if m.iter().any(|l| l.contains("added"))
    ));
    std::fs::remove_file(dir.join("model.safetensors")).unwrap();
    assert!(matches!(
        engine.verify_source(&dir),
        Err(EngineFileError::SourceMismatch(m)) if m.iter().any(|l| l.contains("missing"))
    ));
}

#[test]
fn truncated_file_is_refused() {
    let (tmp, dir) = scratch_checkpoint();
    let out = tmp.path().join("t.oeng");
    convert(&dir, &out).unwrap();
    let full = std::fs::metadata(&out).unwrap().len();
    for keep in [full - 1, full / 2, 4096, 40, 3] {
        let cut = tmp.path().join("cut.oeng");
        std::fs::write(&cut, &std::fs::read(&out).unwrap()[..keep as usize]).unwrap();
        assert!(
            matches!(
                EngineFile::open(&cut),
                Err(EngineFileError::Corrupt { .. } | EngineFileError::BadMagic { .. })
            ),
            "kept {keep} of {full}"
        );
    }
}

#[test]
fn header_tamper_bad_magic_and_version_are_refused() {
    let (tmp, dir) = scratch_checkpoint();
    let out = tmp.path().join("t.oeng");
    convert(&dir, &out).unwrap();
    let good = std::fs::read(&out).unwrap();
    let write = |name: &str, bytes: &[u8]| {
        let p = tmp.path().join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    };

    let mut tampered = good.clone();
    tampered[60] ^= 1;
    assert!(matches!(
        EngineFile::open(write("a", &tampered)),
        Err(EngineFileError::Corrupt { reason, .. }) if reason.contains("checksum")
    ));

    let mut magic = good.clone();
    magic[0] = b'X';
    assert!(matches!(
        EngineFile::open(write("b", &magic)),
        Err(EngineFileError::BadMagic { .. })
    ));

    let mut version = good;
    version[8] = 9;
    assert!(matches!(
        EngineFile::open(write("c", &version)),
        Err(EngineFileError::LayoutVersion { found: 9, .. })
    ));
}

#[test]
fn missing_or_unknown_tensors_refuse_conversion_and_leave_no_file() {
    let (tmp, dir) = scratch_checkpoint();
    // A shard that adds an unknown tensor name.
    let header = br#"{"mystery.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
    let mut extra = (header.len() as u64).to_le_bytes().to_vec();
    extra.extend_from_slice(header);
    extra.extend_from_slice(&[0, 0]);
    std::fs::write(dir.join("zz.safetensors"), extra).unwrap();
    let out = tmp.path().join("t.oeng");
    assert!(convert(&dir, &out).is_err());
    assert!(!out.exists());
    assert!(!tmp.path().join("t.oeng.partial").exists());

    std::fs::remove_file(dir.join("zz.safetensors")).unwrap();
    std::fs::remove_file(dir.join("model.safetensors")).unwrap();
    assert!(convert(&dir, &out).is_err());
}
