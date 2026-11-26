//! Module loading from embedded fatbin/cubin bytes.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;

use crate::error::{CudaError, Result};

/// A kernel entry point resolved from a loaded module.
pub type Kernel = CudaFunction;

/// A loaded fatbin/cubin image.
pub struct KernelModule {
    module: Arc<CudaModule>,
}

impl KernelModule {
    /// Loads a module image (fatbin, cubin or PTX text) into the context.
    pub fn load(ctx: &Arc<CudaContext>, image: &[u8]) -> Result<Self> {
        if image.is_empty() {
            return Err(CudaError::Module("empty module image".into()));
        }
        let mut bytes = image.to_vec();
        if bytes.last() != Some(&0) {
            bytes.push(0);
        }
        let module = ctx
            .load_module(Ptx::from_binary(bytes))
            .map_err(|e| CudaError::Module(e.to_string()))?;
        Ok(Self { module })
    }

    /// Resolves a kernel by its symbol name.
    pub fn kernel(&self, name: &str) -> Result<Kernel> {
        self.module
            .load_function(name)
            .map_err(|e| CudaError::Module(format!("{name}: {e}")))
    }
}
