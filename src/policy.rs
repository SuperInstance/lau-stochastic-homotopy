//! Agent policies as points in a function space.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::error::{HomotopyError, Result};

/// A policy maps states to action probability distributions.
/// Represented as a parameter vector in a continuous policy space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Parameter vector defining the policy.
    #[serde(with = "vector_serde")]
    pub params: DVector<f64>,
    /// Optional label for the policy.
    pub label: Option<String>,
}

mod vector_serde {
    use nalgebra::DVector;
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &DVector<f64>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_some(&v.as_slice())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<DVector<f64>, D::Error> {
        let data: Vec<f64> = Vec::deserialize(d)?;
        Ok(DVector::from_vec(data))
    }
}

impl Policy {
    /// Create a new policy from a parameter vector.
    pub fn new(params: Vec<f64>) -> Self {
        Self {
            params: DVector::from_vec(params),
            label: None,
        }
    }

    /// Create a new policy with a label.
    pub fn labeled(params: Vec<f64>, label: impl Into<String>) -> Self {
        Self {
            params: DVector::from_vec(params),
            label: Some(label.into()),
        }
    }

    /// The zero policy (origin of policy space).
    pub fn zero(dim: usize) -> Self {
        Self {
            params: DVector::zeros(dim),
            label: None,
        }
    }

    /// Dimension of the policy parameter space.
    pub fn dim(&self) -> usize {
        self.params.nrows()
    }

    /// Evaluate the policy at a given state index.
    pub fn evaluate(&self, state_idx: usize) -> f64 {
        if state_idx < self.params.nrows() {
            self.params[state_idx]
        } else {
            0.0
        }
    }

    /// L2 distance to another policy.
    pub fn distance_to(&self, other: &Policy) -> f64 {
        if self.dim() != other.dim() {
            return f64::INFINITY;
        }
        (&self.params - &other.params).norm()
    }

    /// Linear interpolation between this policy and another.
    pub fn lerp(&self, other: &Policy, t: f64) -> Result<Policy> {
        if self.dim() != other.dim() {
            return Err(HomotopyError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }
        let params = &self.params * (1.0 - t) + &other.params * t;
        Ok(Policy {
            params,
            label: None,
        })
    }

    /// Normalize the policy parameters to unit length.
    pub fn normalize(&self) -> Policy {
        let norm = self.params.norm();
        if norm < 1e-12 {
            return self.clone();
        }
        Policy {
            params: &self.params / norm,
            label: self.label.clone(),
        }
    }

    /// Add Gaussian noise to the policy.
    pub fn add_noise(&self, sigma: f64, rng: &mut impl rand::Rng) -> Policy {
        use rand_distr::Normal;
        let dist = Normal::new(0.0, sigma).unwrap();
        let noise: Vec<f64> = (0..self.dim()).map(|_| rng.sample(dist)).collect();
        let noise_vec = DVector::from_vec(noise);
        Policy {
            params: &self.params + noise_vec,
            label: self.label.clone(),
        }
    }

    /// Check if this policy is approximately equal to another.
    pub fn approx_eq(&self, other: &Policy, tol: f64) -> bool {
        self.distance_to(other) < tol
    }
}

/// A parameterized path through policy space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyPath {
    /// Waypoints along the path.
    pub waypoints: Vec<Policy>,
}

impl PolicyPath {
    /// Create a path from waypoints.
    pub fn new(waypoints: Vec<Policy>) -> Result<Self> {
        if waypoints.is_empty() {
            return Err(HomotopyError::EmptySpace);
        }
        Ok(Self { waypoints })
    }

    /// A constant path at a single policy.
    pub fn constant(policy: Policy) -> Self {
        Self { waypoints: vec![policy] }
    }

    /// Evaluate the path at parameter t ∈ [0, 1] via linear interpolation.
    pub fn evaluate(&self, t: f64) -> Policy {
        if self.waypoints.len() == 1 {
            return self.waypoints[0].clone();
        }
        let t = t.clamp(0.0, 1.0);
        let n = self.waypoints.len() - 1;
        let idx = (t * n as f64).floor() as usize;
        let idx = idx.min(n - 1);
        let local_t = t * n as f64 - idx as f64;
        self.waypoints[idx].lerp(&self.waypoints[idx + 1], local_t).unwrap_or_else(|_| self.waypoints[idx].clone())
    }

    /// Length of the path (sum of segment lengths).
    pub fn length(&self) -> f64 {
        self.waypoints.windows(2).map(|w| w[0].distance_to(&w[1])).sum()
    }

    /// Is this path a loop (start == end)?
    pub fn is_loop(&self, tol: f64) -> bool {
        if self.waypoints.len() < 2 {
            return true;
        }
        self.waypoints.first().unwrap().approx_eq(self.waypoints.last().unwrap(), tol)
    }

    /// Concatenate two paths.
    pub fn concatenate(&self, other: &PolicyPath) -> Result<PolicyPath> {
        let mut waypoints = self.waypoints.clone();
        waypoints.extend(other.waypoints.iter().cloned());
        PolicyPath::new(waypoints)
    }

    /// Reverse the path.
    pub fn reverse(&self) -> PolicyPath {
        PolicyPath {
            waypoints: self.waypoints.iter().cloned().rev().collect(),
        }
    }
}
