fn main() {
    println!("cargo:rerun-if-changed=csrc/jpeg_shim.c");
    if std::env::var_os("CARGO_FEATURE_LIBJPEG").is_some() {
        cc::Build::new()
            .file("csrc/jpeg_shim.c")
            .compile("scriptor_jpeg_shim");
        println!("cargo:rustc-link-lib=jpeg");
    }
}
