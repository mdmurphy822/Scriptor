//! Process exit codes of `serve`, which a supervisor uses to decide whether to restart.

use scriptor_engine::boot::BootError;

/// Clean stop after a drain.
pub const CLEAN: i32 = 0;
/// Internal failure whose state is unknown (a panic or a lost thread); restart.
pub const INTERNAL: i32 = 70;
/// Transient or device failure (boot stage failure, sticky device fault); restart.
pub const TRANSIENT: i32 = 75;
/// Refused configuration (memory plan, unusable inputs); a restart would fail the same way.
pub const REFUSED: i32 = 78;

/// Exit code for a failed boot: refusals and unusable cache or checkpoint inputs are
/// [`REFUSED`], any stage failure is [`TRANSIENT`].
pub fn boot_exit_code(e: &BootError) -> i32 {
    match e {
        BootError::Refused(_) | BootError::Cache(_) => REFUSED,
        BootError::Stage { .. } => TRANSIENT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusals_are_not_restartable() {
        assert_eq!(boot_exit_code(&BootError::Refused("no fit".into())), 78);
        assert_eq!(boot_exit_code(&BootError::Cache("dir".into())), 78);
    }

    #[test]
    fn stage_failures_restart() {
        assert_eq!(boot_exit_code(&BootError::stage("kv_pool", "oom")), 75);
    }

    #[test]
    fn codes_are_distinct() {
        let mut all = [CLEAN, INTERNAL, TRANSIENT, REFUSED];
        all.sort_unstable();
        assert!(all.windows(2).all(|w| w[0] != w[1]));
    }
}
