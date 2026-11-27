//! CUDA graph capture and replay with updatable kernel-node parameters.

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
