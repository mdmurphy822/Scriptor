use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use scriptor_image::{
    GridThw, MediaPolicy, PATCH_ELEMS, SmartResizeConfig, decode_jpeg, inner_offset, load_media,
    parse_data_url, patch_origin, patchify_bf16, preprocess, resize_bicubic_aa, smart_resize,
    smart_resize_with,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    file: String,
    width: u32,
    height: u32,
    conforming: bool,
    target_height: u32,
    target_width: u32,
    tokens: usize,
    pixels_b64: String,
    resized_b64: Option<String>,
}

fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/image")
}

fn goldens() -> Vec<Case> {
    let text = std::fs::read_to_string(fixture_dir().join("goldens.json")).unwrap();
    serde_json::from_str::<Golden>(&text).unwrap().cases
}

fn jpeg_of(case: &Case) -> Vec<u8> {
    std::fs::read(fixture_dir().join(&case.file)).unwrap()
}

#[test]
fn decoded_pixels_equal_pil() {
    for case in goldens() {
        let image = decode_jpeg(&jpeg_of(&case)).unwrap();
        assert_eq!((image.width, image.height), (case.width, case.height));
        let want = STANDARD.decode(&case.pixels_b64).unwrap();
        assert_eq!(image.data.len(), want.len(), "{}", case.name);
        let diff = image.data.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "{}: {diff} differing samples", case.name);
    }
}

#[test]
fn smart_resize_matches_goldens() {
    for case in goldens() {
        let target = smart_resize(case.height, case.width).unwrap();
        assert_eq!(
            target,
            (case.target_height, case.target_width),
            "{}",
            case.name
        );
        assert_eq!(
            (target.0 / 28) as usize * (target.1 / 28) as usize,
            case.tokens
        );
        assert_eq!(case.conforming, target == (case.height, case.width));
    }
}

#[test]
fn smart_resize_matches_the_semantik_vectors() {
    let config = SmartResizeConfig {
        min_pixels: 112 * 112,
        max_pixels: 28 * 28 * 2 * 6000,
        max_aspect_ratio: None,
        ..SmartResizeConfig::default()
    };
    let of = |h, w| smart_resize_with(h, w, &config).unwrap();
    assert_eq!(of(100, 100), (112, 112));
    assert_eq!(of(28, 28), (84, 84));
    assert_eq!(of(4000, 4000), (2156, 2156));
    assert_eq!(of(14, 14), (84, 84));
    assert_eq!(of(1000, 3), (1456, 28));
    assert_eq!(of(3000, 2000), (2632, 1764));
    assert_eq!(of(5, 2000), (28, 1596));
    assert_eq!(of(70, 700), (56, 700));
    assert_eq!(of(42, 700), (56, 700));
    for (w, h) in [(867, 6771), (6771, 867), (4096, 4096)] {
        let (th, tw) = of(h, w);
        assert!(u64::from(th) * u64::from(tw) / 784 < 6084);
    }
}

#[test]
fn smart_resize_defaults() {
    assert_eq!(smart_resize(28, 28).unwrap(), (84, 84));
    assert_eq!(smart_resize(100, 100).unwrap(), (112, 112));
    assert_eq!(smart_resize(10_000, 10_000).unwrap().0 % 28, 0);
    assert!(smart_resize(1000, 3).is_err());
    assert!(smart_resize(0, 10).is_err());
    let bad = SmartResizeConfig {
        num_frames: 1,
        ..SmartResizeConfig::default()
    };
    assert!(smart_resize_with(100, 100, &bad).is_err());
}

#[test]
fn resize_tracks_pil_bicubic() {
    for case in goldens().into_iter().filter(|c| !c.conforming) {
        let image = decode_jpeg(&jpeg_of(&case)).unwrap();
        let out = resize_bicubic_aa(&image, case.target_height, case.target_width).unwrap();
        let want = STANDARD.decode(case.resized_b64.as_ref().unwrap()).unwrap();
        assert_eq!(out.data.len(), want.len());
        let worst = out
            .data
            .iter()
            .zip(&want)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(worst <= 1, "{}: max diff {worst}", case.name);
    }
}

#[test]
fn identity_resize_returns_input() {
    let case = &goldens()[1];
    let image = decode_jpeg(&jpeg_of(case)).unwrap();
    let out = resize_bicubic_aa(&image, image.height, image.width).unwrap();
    assert_eq!(out, image);
}

#[test]
fn patch_order_matches_hand_index_map() {
    let grid = GridThw { t: 1, h: 4, w: 4 };
    let expected = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
    ];
    for (i, (pr, pc)) in expected.iter().enumerate() {
        assert_eq!(patch_origin(&grid, i), (pr * 14, pc * 14), "patch {i}");
    }
    assert_eq!(inner_offset(0, 0, 0, 0), 0);
    assert_eq!(inner_offset(0, 1, 0, 0), 196);
    assert_eq!(inner_offset(1, 0, 0, 0), 392);
    assert_eq!(inner_offset(2, 1, 13, 13), PATCH_ELEMS - 1);
}

#[test]
fn patchify_places_each_pixel_and_duplicates_frames() {
    let case = &goldens()[1];
    let image = decode_jpeg(&jpeg_of(case)).unwrap();
    let (grid, out) = patchify_bf16(&image);
    assert_eq!((grid.t, grid.h, grid.w), (1, 8, 10));
    assert_eq!(out.len(), 80 * PATCH_ELEMS);
    let tables = scriptor_image::bf16_tables();
    for index in [0usize, 1, 2, 3, 4, 17, 79] {
        let (r0, c0) = patch_origin(&grid, index);
        for (c, y, x) in [(0usize, 0u32, 0u32), (1, 5, 7), (2, 13, 13)] {
            let value = image.pixel(r0 + y, c0 + x)[c];
            let want = tables[c][value as usize];
            let base = index * PATCH_ELEMS;
            assert_eq!(out[base + inner_offset(c, 0, y as usize, x as usize)], want);
            assert_eq!(out[base + inner_offset(c, 1, y as usize, x as usize)], want);
        }
    }
}

#[test]
fn token_counts_match_grid() {
    for case in goldens() {
        let out = preprocess(&jpeg_of(&case), &SmartResizeConfig::default()).unwrap();
        assert_eq!(out.tokens, case.tokens, "{}", case.name);
        assert_eq!(out.pixels.len(), out.grid_thw.patches() * PATCH_ELEMS);
        assert_eq!(out.resized, !case.conforming);
    }
}

#[test]
fn data_url_and_local_policy() {
    let case = &goldens()[0];
    let bytes = jpeg_of(case);
    let url = format!("data:image/jpeg;base64,{}", STANDARD.encode(&bytes));
    assert_eq!(parse_data_url(&url).unwrap(), bytes);
    assert_eq!(load_media(&url, &MediaPolicy::default()).unwrap(), bytes);
    assert!(parse_data_url("data:image/gif;base64,AAAA").is_err());
    assert!(parse_data_url("data:image/png,AAAA").is_err());
    assert!(parse_data_url("data:image/png;base64,@@@").is_err());
    assert!(load_media("https://example.com/x.jpg", &MediaPolicy::default()).is_err());

    let path = fixture_dir().join(&case.file);
    let path = path.to_str().unwrap();
    assert!(load_media(path, &MediaPolicy::default()).is_err());
    let allowed = MediaPolicy {
        allowed_local_media_path: Some(fixture_dir()),
    };
    assert_eq!(load_media(path, &allowed).unwrap(), bytes);
    assert_eq!(
        load_media(&format!("file://{path}"), &allowed).unwrap(),
        bytes
    );
    let elsewhere = MediaPolicy {
        allowed_local_media_path: Some(fixture_dir().join("..").join("chat")),
    };
    assert!(load_media(path, &elsewhere).is_err());
}
