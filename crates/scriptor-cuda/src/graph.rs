//! CUDA graph capture and replay with updatable kernel-node parameters.

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::Arc;

use cudarc::driver::sys;
use cudarc::driver::{CudaGraph, CudaStream};

use crate::error::{CudaError, Result};

fn check(code: sys::CUresult, what: &str) -> Result<()> {
    if code == sys::cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaError::Graph(format!("{what}: {code:?}")))
    }
}

/// Packed by-value kernel arguments, in launch order.
#[derive(Debug, Default, Clone)]
pub struct KernelArgs {
    values: Vec<Vec<u8>>,
}

impl KernelArgs {
    /// Creates an empty argument list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one by-value argument (scalar or device pointer).
    pub fn push<T: Copy>(mut self, value: T) -> Self {
        let mut bytes = vec![0u8; std::mem::size_of::<T>()];
        // SAFETY: `bytes` has exactly size_of::<T>() bytes and T is Copy.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&value as *const T).cast::<u8>(),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        self.values.push(bytes);
        self
    }

    /// Number of arguments.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the list has no arguments.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn pointers(&mut self) -> Vec<*mut c_void> {
        self.values
            .iter_mut()
            .map(|v| v.as_mut_ptr().cast::<c_void>())
            .collect()
    }
}

/// New launch configuration and arguments for one captured kernel node.
/// Fields left `None` keep the captured value.
#[derive(Debug, Default, Clone)]
pub struct KernelNodeUpdate {
    pub grid: Option<(u32, u32, u32)>,
    pub block: Option<(u32, u32, u32)>,
    pub shared_mem_bytes: Option<u32>,
    pub args: Option<KernelArgs>,
}

/// An instantiated graph captured from a stream, with its kernel nodes
/// indexed in capture-graph order.
pub struct CapturedGraph {
    graph: CudaGraph,
    kernel_nodes: Vec<sys::CUgraphNode>,
}

// SAFETY: the graph is only mutated through `&mut self`/serialised calls and
// node handles are plain driver identifiers owned by the graph.
unsafe impl Send for CapturedGraph {}

impl CapturedGraph {
    /// Runs `body` while capturing `stream`, then instantiates the graph.
    pub fn capture<F>(stream: &Arc<CudaStream>, body: F) -> Result<Self>
    where
        F: FnOnce(&Arc<CudaStream>) -> Result<()>,
    {
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let outcome = body(stream);
        let ended = stream.end_capture(sys::CUgraphInstantiate_flags(0));
        outcome?;
        let graph = ended?.ok_or_else(|| CudaError::Graph("capture produced no graph".into()))?;
        let kernel_nodes = kernel_nodes(graph.cu_graph())?;
        Ok(Self {
            graph,
            kernel_nodes,
        })
    }

    /// Number of kernel nodes in the captured graph.
    pub fn kernel_node_count(&self) -> usize {
        self.kernel_nodes.len()
    }

    /// The execution-context handle each kernel node was captured with, in node order; nodes
    /// captured from a green-context stream report that context.
    pub fn kernel_node_contexts(&self) -> Result<Vec<usize>> {
        self.kernel_nodes
            .iter()
            .map(|&node| {
                let mut params = MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::zeroed();
                // SAFETY: `node` belongs to the owned graph; params is writable.
                check(
                    unsafe { sys::cuGraphKernelNodeGetParams_v2(node, params.as_mut_ptr()) },
                    "cuGraphKernelNodeGetParams",
                )?;
                // SAFETY: initialised by the successful call above.
                Ok(unsafe { params.assume_init() }.ctx as usize)
            })
            .collect()
    }

    /// Replays the graph on the stream it was captured from.
    pub fn launch(&self) -> Result<()> {
        Ok(self.graph.launch()?)
    }

    /// Rewrites the launch parameters of kernel node `index` in the
    /// instantiated graph via `cuGraphExecKernelNodeSetParams`.
    ///
    /// # Safety
    /// `update.args`, when present, must match the kernel's parameter list in
    /// count, size and layout; no replay may be in flight.
    pub unsafe fn update_kernel_node(
        &self,
        index: usize,
        mut update: KernelNodeUpdate,
    ) -> Result<()> {
        let node = *self.kernel_nodes.get(index).ok_or_else(|| {
            CudaError::Graph(format!(
                "kernel node {index} out of range ({} nodes)",
                self.kernel_nodes.len()
            ))
        })?;
        let mut params = MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::zeroed();
        // SAFETY: `node` belongs to the owned graph; params is writable.
        check(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, params.as_mut_ptr()) },
            "cuGraphKernelNodeGetParams",
        )?;
        // SAFETY: initialised by the successful call above.
        let mut params = unsafe { params.assume_init() };
        if let Some((x, y, z)) = update.grid {
            params.gridDimX = x;
            params.gridDimY = y;
            params.gridDimZ = z;
        }
        if let Some((x, y, z)) = update.block {
            params.blockDimX = x;
            params.blockDimY = y;
            params.blockDimZ = z;
        }
        if let Some(bytes) = update.shared_mem_bytes {
            params.sharedMemBytes = bytes;
        }
        let mut pointers;
        if let Some(args) = update.args.as_mut() {
            pointers = args.pointers();
            params.kernelParams = pointers.as_mut_ptr();
            params.extra = std::ptr::null_mut();
        }
        // SAFETY: params describes a kernel node of this exec graph; the
        // driver copies the argument values during the call.
        check(
            unsafe {
                sys::cuGraphExecKernelNodeSetParams_v2(self.graph.cu_graph_exec(), node, &params)
            },
            "cuGraphExecKernelNodeSetParams",
        )
    }
}

fn kernel_nodes(graph: sys::CUgraph) -> Result<Vec<sys::CUgraphNode>> {
    let mut count: usize = 0;
    // SAFETY: a null node array requests only the count.
    check(
        unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut count) },
        "cuGraphGetNodes(count)",
    )?;
    let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); count];
    // SAFETY: `nodes` has room for `count` handles.
    check(
        unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count) },
        "cuGraphGetNodes",
    )?;
    nodes.truncate(count);
    let mut kernels = Vec::new();
    for node in nodes {
        let mut kind = MaybeUninit::<sys::CUgraphNodeType>::uninit();
        // SAFETY: `node` came from this graph.
        check(
            unsafe { sys::cuGraphNodeGetType(node, kind.as_mut_ptr()) },
            "cuGraphNodeGetType",
        )?;
        // SAFETY: initialised by the call above.
        if unsafe { kind.assume_init() } == sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            kernels.push(node);
        }
    }
    Ok(kernels)
}
