use thiserror::Error;

#[derive(Error, Debug)]
pub enum HomotopyError {
    #[error("continuity violation at t={t}: gap={gap}")]
    ContinuityViolation { t: f64, gap: f64 },

    #[error("singular point encountered at parameter {param}")]
    SingularPoint { param: f64 },

    #[error("obstruction class {class} is nonzero — extension blocked")]
    ObstructionNonZero { class: usize },

    #[error("homotopy failed: {reason}")]
    HomotopyFailed { reason: String },

    #[error("covering lift failed for path — no unique lift exists")]
    LiftFailed,

    #[error("fundamental group computation error: {reason}")]
    GroupError { reason: String },

    #[error("CW structure required for Whitehead theorem")]
    NotCWComplex,

    #[error("numerical error: {detail}")]
    Numerical { detail: String },

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("empty policy space")]
    EmptySpace,
}

pub type Result<T> = std::result::Result<T, HomotopyError>;
