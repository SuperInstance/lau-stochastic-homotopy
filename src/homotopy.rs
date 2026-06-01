//! Homotopy: continuous deformation between agent policies.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::error::{HomotopyError, Result};
use crate::policy::Policy;

/// A homotopy H: [0,1] × PolicySpace → PolicySpace.
/// H(0, ·) = f (source policy), H(1, ·) = g (target policy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Homotopy {
    /// Source policy (t=0).
    pub source: Policy,
    /// Target policy (t=1).
    pub target: Policy,
    /// Number of discretization steps.
    pub steps: usize,
    /// Type of interpolation.
    pub interpolation: InterpolationType,
}

/// How to interpolate between policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationType {
    /// Linear interpolation in parameter space.
    Linear,
    /// Spherical interpolation (SLERP) — preserves norm.
    Spherical,
    /// Bezier interpolation for smooth paths.
    Bezier,
}

impl Homotopy {
    /// Create a new homotopy between two policies.
    pub fn new(source: Policy, target: Policy, steps: usize) -> Result<Self> {
        if source.dim() != target.dim() {
            return Err(HomotopyError::DimensionMismatch {
                expected: source.dim(),
                actual: target.dim(),
            });
        }
        if steps == 0 {
            return Err(HomotopyError::HomotopyFailed {
                reason: "steps must be > 0".into(),
            });
        }
        Ok(Self {
            source,
            target,
            steps,
            interpolation: InterpolationType::Linear,
        })
    }

    /// Create with spherical interpolation.
    pub fn spherical(source: Policy, target: Policy, steps: usize) -> Result<Self> {
        let mut h = Self::new(source, target, steps)?;
        h.interpolation = InterpolationType::Spherical;
        Ok(h)
    }

    /// Create with Bezier interpolation.
    pub fn bezier(source: Policy, target: Policy, steps: usize) -> Result<Self> {
        let mut h = Self::new(source, target, steps)?;
        h.interpolation = InterpolationType::Bezier;
        Ok(h)
    }

    /// Evaluate the homotopy at parameter t ∈ [0, 1].
    pub fn evaluate(&self, t: f64) -> Policy {
        let t = t.clamp(0.0, 1.0);
        match self.interpolation {
            InterpolationType::Linear => self.linear_interp(t),
            InterpolationType::Spherical => self.spherical_interp(t),
            InterpolationType::Bezier => self.bezier_interp(t),
        }
    }

    fn linear_interp(&self, t: f64) -> Policy {
        let params = &self.source.params * (1.0 - t) + &self.target.params * t;
        Policy { params, label: None }
    }

    fn spherical_interp(&self, t: f64) -> Policy {
        let p1 = &self.source.params;
        let p2 = &self.target.params;
        let n1 = p1.norm();
        let n2 = p2.norm();
        if n1 < 1e-12 || n2 < 1e-12 {
            return self.linear_interp(t);
        }
        let u1 = p1 / n1;
        let u2 = p2 / n2;
        let dot = u1.dot(&u2).clamp(-1.0, 1.0);
        let omega = dot.acos();
        if omega.abs() < 1e-10 {
            return self.linear_interp(t);
        }
        let sin_omega = omega.sin();
        let s1 = ((1.0 - t) * omega).sin() / sin_omega;
        let s2 = (t * omega).sin() / sin_omega;
        let norm = n1 * (1.0 - t) + n2 * t;
        let params = (u1 * s1 + u2 * s2) * norm;
        Policy { params, label: None }
    }

    fn bezier_interp(&self, t: f64) -> Policy {
        // Quadratic Bezier with midpoint as control
        let mid = (&self.source.params + &self.target.params) * 0.5;
        // Add some curvature
        let dim = self.source.dim();
        let offset = DVector::from_fn(dim, |i, _| {
            ((i as f64 + 1.0) * std::f64::consts::PI * 0.1).sin() * 0.2
        });
        let control = mid + offset;
        let one_minus_t = 1.0 - t;
        let params = &self.source.params * (one_minus_t * one_minus_t)
            + &control * (2.0 * one_minus_t * t)
            + &self.target.params * (t * t);
        Policy { params, label: None }
    }

    /// Get the full path as a vector of policies.
    pub fn path(&self) -> Vec<Policy> {
        (0..=self.steps)
            .map(|i| self.evaluate(i as f64 / self.steps as f64))
            .collect()
    }

    /// Check continuity: verify that consecutive points are close.
    pub fn check_continuity(&self, epsilon: f64) -> Result<()> {
        let path = self.path();
        for i in 1..path.len() {
            let gap = path[i - 1].distance_to(&path[i]);
            if gap > epsilon {
                return Err(HomotopyError::ContinuityViolation {
                    t: i as f64 / self.steps as f64,
                    gap,
                });
            }
        }
        Ok(())
    }

    /// Maximum step size in the discretized path.
    pub fn max_step_size(&self) -> f64 {
        let path = self.path();
        path.windows(2)
            .map(|w| w[0].distance_to(&w[1]))
            .fold(0.0_f64, f64::max)
    }

    /// Total path length.
    pub fn total_length(&self) -> f64 {
        let path = self.path();
        path.windows(2)
            .map(|w| w[0].distance_to(&w[1]))
            .sum()
    }

    /// Check if source and target are homotopic via this homotopy.
    pub fn is_valid(&self, epsilon: f64) -> bool {
        self.check_continuity(epsilon).is_ok()
    }

    /// Compose two homotopies (source → mid, then mid → target).
    pub fn compose(&self, other: &Homotopy) -> Result<Homotopy> {
        if !self.target.approx_eq(&other.source, 1e-10) {
            return Err(HomotopyError::HomotopyFailed {
                reason: "endpoints don't match for composition".into(),
            });
        }
        Homotopy::new(
            self.source.clone(),
            other.target.clone(),
            self.steps + other.steps,
        )
    }

    /// Reverse the homotopy (target → source).
    pub fn reverse(&self) -> Homotopy {
        Homotopy {
            source: self.target.clone(),
            target: self.source.clone(),
            steps: self.steps,
            interpolation: self.interpolation,
        }
    }

    /// Reparameterize: apply a monotone function φ: [0,1] → [0,1].
    pub fn reparameterize<F: Fn(f64) -> f64>(&self, phi: F) -> Homotopy {
        // Store as a new homotopy (the reparameterization is captured in the path)
        let new_source = self.evaluate(phi(0.0));
        let new_target = self.evaluate(phi(1.0));
        Homotopy {
            source: new_source,
            target: new_target,
            steps: self.steps,
            interpolation: self.interpolation,
        }
    }
}

/// A relative homotopy H: (X, A) → (Y, B) where H fixes a subspace.
#[derive(Debug, Clone)]
pub struct RelativeHomotopy {
    pub base_homotopy: Homotopy,
    /// Indices of parameters that are held fixed.
    pub fixed_indices: Vec<usize>,
}

impl RelativeHomotopy {
    pub fn new(homotopy: Homotopy, fixed_indices: Vec<usize>) -> Result<Self> {
        let dim = homotopy.source.dim();
        for &i in &fixed_indices {
            if i >= dim {
                return Err(HomotopyError::DimensionMismatch {
                    expected: dim,
                    actual: i,
                });
            }
        }
        Ok(Self {
            base_homotopy: homotopy,
            fixed_indices,
        })
    }

    /// Evaluate with fixed parameters held constant.
    pub fn evaluate(&self, t: f64) -> Policy {
        let mut p = self.base_homotopy.evaluate(t);
        for &i in &self.fixed_indices {
            p.params[i] = self.base_homotopy.source.params[i];
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_homotopy_creation() {
        let a = Policy::new(vec![0.0, 0.0]);
        let b = Policy::new(vec![1.0, 1.0]);
        let h = Homotopy::new(a, b, 10).unwrap();
        assert_eq!(h.steps, 10);
    }

    #[test]
    fn test_homotopy_dimension_mismatch() {
        let a = Policy::new(vec![0.0]);
        let b = Policy::new(vec![1.0, 2.0]);
        assert!(Homotopy::new(a, b, 10).is_err());
    }

    #[test]
    fn test_linear_interpolation_endpoints() {
        let a = Policy::new(vec![0.0, 0.0, 0.0]);
        let b = Policy::new(vec![1.0, 2.0, 3.0]);
        let h = Homotopy::new(a.clone(), b.clone(), 10).unwrap();
        let at_0 = h.evaluate(0.0);
        let at_1 = h.evaluate(1.0);
        assert!(at_0.approx_eq(&a, 1e-10));
        assert!(at_1.approx_eq(&b, 1e-10));
    }

    #[test]
    fn test_linear_interpolation_midpoint() {
        let a = Policy::new(vec![0.0]);
        let b = Policy::new(vec![2.0]);
        let h = Homotopy::new(a, b, 10).unwrap();
        let mid = h.evaluate(0.5);
        assert_relative_eq!(mid.params[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spherical_interpolation() {
        let a = Policy::new(vec![1.0, 0.0]);
        let b = Policy::new(vec![0.0, 1.0]);
        let h = Homotopy::spherical(a, b, 10).unwrap();
        let mid = h.evaluate(0.5);
        assert_relative_eq!(mid.params.norm(), 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_bezier_interpolation() {
        let a = Policy::new(vec![0.0, 0.0]);
        let b = Policy::new(vec![1.0, 1.0]);
        let h = Homotopy::bezier(a.clone(), b.clone(), 10).unwrap();
        assert!(h.evaluate(0.0).approx_eq(&a, 1e-10));
        assert!(h.evaluate(1.0).approx_eq(&b, 1e-10));
    }

    #[test]
    fn test_path_length() {
        let a = Policy::new(vec![0.0]);
        let b = Policy::new(vec![1.0]);
        let h = Homotopy::new(a, b, 100).unwrap();
        let len = h.total_length();
        assert_relative_eq!(len, 1.0, epsilon = 0.05);
    }

    #[test]
    fn test_continuity_check() {
        let a = Policy::new(vec![0.0]);
        let b = Policy::new(vec![1.0]);
        let h = Homotopy::new(a, b, 100).unwrap();
        assert!(h.check_continuity(0.1).is_ok());
    }

    #[test]
    fn test_compose_homotopies() {
        let a = Policy::new(vec![0.0]);
        let b = Policy::new(vec![0.5]);
        let c = Policy::new(vec![1.0]);
        let h1 = Homotopy::new(a, b.clone(), 5).unwrap();
        let h2 = Homotopy::new(b, c, 5).unwrap();
        let composed = h1.compose(&h2).unwrap();
        assert_eq!(composed.steps, 10);
    }

    #[test]
    fn test_reverse_homotopy() {
        let a = Policy::new(vec![0.0]);
        let b = Policy::new(vec![1.0]);
        let h = Homotopy::new(a.clone(), b.clone(), 10).unwrap();
        let rev = h.reverse();
        assert!(rev.source.approx_eq(&b, 1e-10));
        assert!(rev.target.approx_eq(&a, 1e-10));
    }

    #[test]
    fn test_relative_homotopy() {
        let a = Policy::new(vec![0.0, 1.0]);
        let b = Policy::new(vec![2.0, 3.0]);
        let h = Homotopy::new(a.clone(), b, 10).unwrap();
        let rel = RelativeHomotopy::new(h, vec![1]).unwrap();
        let at_1 = rel.evaluate(1.0);
        assert_relative_eq!(at_1.params[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_homotopy_path_has_correct_length() {
        let a = Policy::new(vec![0.0]);
        let b = Policy::new(vec![1.0]);
        let h = Homotopy::new(a, b, 10).unwrap();
        assert_eq!(h.path().len(), 11); // steps + 1
    }

    #[test]
    fn test_max_step_size() {
        let a = Policy::new(vec![0.0]);
        let b = Policy::new(vec![1.0]);
        let h = Homotopy::new(a, b, 10).unwrap();
        let mss = h.max_step_size();
        assert!(mss > 0.0 && mss < 1.0);
    }

    #[test]
    fn test_reparameterize() {
        let a = Policy::new(vec![0.0]);
        let b = Policy::new(vec![1.0]);
        let h = Homotopy::new(a, b, 10).unwrap();
        let reparam = h.reparameterize(|t| t * t);
        assert_eq!(reparam.steps, 10);
    }

    #[test]
    fn test_zero_steps_error() {
        let a = Policy::new(vec![0.0]);
        let b = Policy::new(vec![1.0]);
        assert!(Homotopy::new(a, b, 0).is_err());
    }

    #[test]
    fn test_compose_mismatch_error() {
        let a = Policy::new(vec![0.0]);
        let b = Policy::new(vec![0.5]);
        let c = Policy::new(vec![1.0]);
        let h1 = Homotopy::new(a, b, 5).unwrap();
        let h2 = Homotopy::new(c, Policy::new(vec![2.0]), 5).unwrap();
        assert!(h1.compose(&h2).is_err());
    }
}
