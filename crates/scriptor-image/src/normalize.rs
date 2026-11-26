use half::bf16;

/// CLIP per-channel mean.
pub const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
/// CLIP per-channel standard deviation.
pub const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Normalises one 8-bit sample of `channel` as `(x / 255 - mean) / std` in fp32.
pub fn normalize_f32(value: u8, channel: usize) -> f32 {
    (f32::from(value) / 255.0 - MEAN[channel]) / STD[channel]
}

/// Normalises one 8-bit sample and rounds the fp32 result to bf16.
pub fn normalize_bf16(value: u8, channel: usize) -> bf16 {
    bf16::from_f32(normalize_f32(value, channel))
}

/// Per-channel lookup tables from 8-bit samples to bf16 normalised values.
pub fn bf16_tables() -> [[bf16; 256]; 3] {
    let mut tables = [[bf16::ZERO; 256]; 3];
    for (channel, table) in tables.iter_mut().enumerate() {
        for (value, slot) in table.iter_mut().enumerate() {
            *slot = normalize_bf16(value as u8, channel);
        }
    }
    tables
}
