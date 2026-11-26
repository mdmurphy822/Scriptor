//! The memory plan's pair-order charge matches the workspace's largest order allocation.

use scriptor_kernels::attention::{PAIR_ORDER_MAX_PAIRS, pair_order_bytes};
use scriptor_spec::memory_plan::PAIR_ORDER_RESERVE_BYTES;

#[test]
fn plan_charge_equals_largest_order_allocation() {
    assert_eq!(
        PAIR_ORDER_RESERVE_BYTES,
        pair_order_bytes(PAIR_ORDER_MAX_PAIRS) as u64
    );
}
