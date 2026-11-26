use crate::error::WeightsError;

/// Element types that appear in safetensors headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    Bool,
    U8,
    I8,
    F8E4M3,
    F8E5M2,
    I16,
    U16,
    F16,
    Bf16,
    I32,
    U32,
    F32,
    I64,
    U64,
    F64,
}

impl Dtype {
    /// Parses the dtype string used in safetensors headers.
    pub fn parse(s: &str) -> Result<Self, WeightsError> {
        Ok(match s {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E5M2" => Self::F8E5M2,
            "I16" => Self::I16,
            "U16" => Self::U16,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "I32" => Self::I32,
            "U32" => Self::U32,
            "F32" => Self::F32,
            "I64" => Self::I64,
            "U64" => Self::U64,
            "F64" => Self::F64,
            other => return Err(WeightsError::UnknownDtype(other.to_string())),
        })
    }

    /// The safetensors header spelling of this dtype.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::F32 => "F32",
            Self::I64 => "I64",
            Self::U64 => "U64",
            Self::F64 => "F64",
        }
    }

    /// Bytes per element.
    pub fn size_bytes(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}
