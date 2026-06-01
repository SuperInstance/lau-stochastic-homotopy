//! # Lau Stochastic Homotopy
//!
//! Connects stochastic processes to homotopy theory — continuous deformation of agent policies.
//!
//! A homotopy between two policies is a continuous path in policy space. If two policies
//! are homotopic, they can be deformed into each other without crossing a singularity.
//! This determines whether an agent can safely transition between behaviors.

pub mod error;
pub mod policy;
pub mod homotopy;
pub mod fundamental_group;
pub mod higher_homotopy;
pub mod equivalence;
pub mod stochastic;
pub mod lifts;
pub mod obstruction;
pub mod whitehead;
pub mod van_kampen;
pub mod application;

pub use error::HomotopyError;
pub use policy::Policy;
pub use homotopy::Homotopy;
pub use fundamental_group::FundamentalGroup;
pub use higher_homotopy::HigherHomotopyGroup;
pub use equivalence::HomotopyEquivalence;
pub use stochastic::StochasticHomotopy;
pub use lifts::CoveringSpace;
pub use obstruction::Obstruction;
pub use whitehead::WhiteheadTheorem;
pub use van_kampen::SeifertVanKampen;
pub use application::PolicyTransitionChecker;
