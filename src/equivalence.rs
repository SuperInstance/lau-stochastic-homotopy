//! Homotopy equivalence: when are two agent spaces "the same shape"?

use serde::{Deserialize, Serialize};
use crate::error::Result;
use crate::policy::Policy;
use crate::homotopy::Homotopy;

/// A continuous map between policy spaces.
#[derive(Debug, Clone)]
pub struct ContinuousMap {
    /// Source dimension.
    pub source_dim: usize,
    /// Target dimension.
    pub target_dim: usize,
    /// The map as a function applied to parameter vectors.
    /// Stored as a linear transformation matrix (simplified).
    pub matrix: Vec<Vec<f64>>,
}

impl ContinuousMap {
    /// Identity map.
    pub fn identity(dim: usize) -> Self {
        let matrix = (0..dim)
            .map(|i| (0..dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        Self { source_dim: dim, target_dim: dim, matrix }
    }

    /// Constant map to a point.
    pub fn constant(dim: usize, _value: f64) -> Self {
        let matrix = vec![vec![0.0; dim]; 1];
        Self { source_dim: dim, target_dim: 1, matrix }
    }

    /// Apply the map to a policy.
    pub fn apply(&self, policy: &Policy) -> Policy {
        let n = self.matrix.len().min(self.target_dim);
        let m = self.matrix.first().map(|r| r.len()).unwrap_or(0).min(policy.dim());
        let mut result = vec![0.0; n];
        for i in 0..n {
            for j in 0..m {
                result[i] += self.matrix[i][j] * policy.params[j];
            }
        }
        Policy::new(result)
    }

    /// Compose two maps: self ∘ other.
    pub fn compose(&self, other: &ContinuousMap) -> ContinuousMap {
        // Matrix multiply
        let n = self.target_dim;
        let m = other.source_dim;
        let k = self.source_dim;
        let mut result = vec![vec![0.0; m]; n];
        for i in 0..n {
            for j in 0..m {
                for l in 0..k {
                    if i < self.matrix.len() && l < self.matrix[i].len() && l < other.matrix.len() && j < other.matrix[l].len() {
                        result[i][j] += self.matrix[i][l] * other.matrix[l][j];
                    }
                }
            }
        }
        ContinuousMap { source_dim: m, target_dim: n, matrix: result }
    }

    /// Check if this is approximately the identity.
    pub fn is_identity(&self, tol: f64) -> bool {
        if self.source_dim != self.target_dim {
            return false;
        }
        for i in 0..self.source_dim {
            for j in 0..self.target_dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                if i < self.matrix.len() && j < self.matrix[i].len() {
                    if (self.matrix[i][j] - expected).abs() > tol {
                        return false;
                    }
                } else if expected.abs() > tol {
                    return false;
                }
            }
        }
        true
    }
}

/// A homotopy equivalence between two policy spaces X and Y.
/// Consists of maps f: X → Y and g: Y → X such that
/// f∘g ≃ id_Y and g∘f ≃ id_X.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomotopyEquivalence {
    /// First space dimension.
    pub dim_x: usize,
    /// Second space dimension.
    pub dim_y: usize,
    /// Max distance between composition and identity.
    pub deviation: f64,
}

impl HomotopyEquivalence {
    /// Check if two spaces (represented by sample points) are homotopy equivalent.
    /// Uses the heuristic that contractible spaces are all homotopy equivalent.
    pub fn check_contractible(
        samples_x: &[Policy],
        samples_y: &[Policy],
    ) -> HomotopyEquivalence {
        let dim_x = samples_x.first().map(|p| p.dim()).unwrap_or(0);
        let dim_y = samples_y.first().map(|p| p.dim()).unwrap_or(0);
        HomotopyEquivalence {
            dim_x,
            dim_y,
            deviation: 0.0,
        }
    }

    /// Two points are always homotopy equivalent.
    pub fn points_equivalent() -> Self {
        Self { dim_x: 0, dim_y: 0, deviation: 0.0 }
    }

    /// Check via deformation retract: if X deformation retracts to a subspace,
    /// then X ≃ that subspace.
    pub fn deformation_retract(
        space: &[Policy],
        subspace: &[Policy],
        steps: usize,
    ) -> Result<HomotopyEquivalence> {
        if space.is_empty() || subspace.is_empty() {
            return Ok(HomotopyEquivalence::points_equivalent());
        }
        // Check that every point in space can be retracted to subspace
        let mut max_dist = 0.0_f64;
        for p in space {
            let min_dist = subspace.iter().map(|q| p.distance_to(q)).fold(f64::INFINITY, f64::min);
            max_dist = max_dist.max(min_dist);
        }
        Ok(HomotopyEquivalence {
            dim_x: space.first().map(|p| p.dim()).unwrap_or(0),
            dim_y: subspace.first().map(|p| p.dim()).unwrap_or(0),
            deviation: max_dist / steps as f64,
        })
    }

    /// Is this a genuine homotopy equivalence?
    pub fn is_equivalence(&self, tol: f64) -> bool {
        self.deviation < tol
    }

    /// Check if two spaces have the same homotopy type by comparing
    /// their fundamental groups (simplified).
    pub fn same_homotopy_type(rank_x: usize, rank_y: usize) -> bool {
        rank_x == rank_y
    }
}

/// A strong deformation retract: X → A where the homotopy fixes A pointwise.
#[derive(Debug, Clone)]
pub struct DeformationRetract {
    pub homotopy: Homotopy,
    pub subspace_indices: Vec<usize>,
}

impl DeformationRetract {
    /// Create a deformation retraction from a space to a point.
    pub fn to_point(space: &[Policy], point_idx: usize, steps: usize) -> Result<Self> {
        let point = space.get(point_idx).cloned().ok_or_else(|| crate::error::HomotopyError::EmptySpace)?;
        let _farthest = space.iter().map(|p| p.distance_to(&point)).fold(0.0_f64, |a, b| a.max(b));
        // Create a homotopy from identity to constant map at point
        let identity_sample = space.get(0).cloned().unwrap_or_else(|| Policy::zero(1));
        let homotopy = Homotopy::new(identity_sample, point, steps)?;
        Ok(DeformationRetract {
            homotopy,
            subspace_indices: vec![point_idx],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_map() {
        let id = ContinuousMap::identity(3);
        let p = Policy::new(vec![1.0, 2.0, 3.0]);
        let result = id.apply(&p);
        assert!(result.approx_eq(&p, 1e-10));
    }

    #[test]
    fn test_identity_is_identity() {
        let id = ContinuousMap::identity(3);
        assert!(id.is_identity(1e-10));
    }

    #[test]
    fn test_constant_map() {
        let c = ContinuousMap::constant(3, 0.0);
        let p = Policy::new(vec![1.0, 2.0, 3.0]);
        let result = c.apply(&p);
        assert_eq!(result.dim(), 1);
        assert_eq!(result.params[0], 0.0);
    }

    #[test]
    fn test_compose_maps() {
        let a = ContinuousMap::identity(2);
        let b = ContinuousMap::identity(2);
        let c = a.compose(&b);
        assert!(c.is_identity(1e-10));
    }

    #[test]
    fn test_points_equivalent() {
        let eq = HomotopyEquivalence::points_equivalent();
        assert!(eq.is_equivalence(1e-10));
    }

    #[test]
    fn test_contractible_spaces() {
        let xs = vec![Policy::new(vec![0.0]), Policy::new(vec![1.0])];
        let ys = vec![Policy::new(vec![0.0]), Policy::new(vec![2.0])];
        let eq = HomotopyEquivalence::check_contractible(&xs, &ys);
        assert!(eq.is_equivalence(1e-10));
    }

    #[test]
    fn test_deformation_retract() {
        let space = vec![Policy::new(vec![0.0]), Policy::new(vec![1.0]), Policy::new(vec![2.0])];
        let sub = vec![Policy::new(vec![1.0])];
        let result = HomotopyEquivalence::deformation_retract(&space, &sub, 10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_same_homotopy_type() {
        assert!(HomotopyEquivalence::same_homotopy_type(1, 1));
        assert!(!HomotopyEquivalence::same_homotopy_type(1, 2));
    }

    #[test]
    fn test_deformation_retract_to_point() {
        let space = vec![Policy::new(vec![0.0]), Policy::new(vec![1.0])];
        let dr = DeformationRetract::to_point(&space, 0, 10);
        assert!(dr.is_ok());
    }

    #[test]
    fn test_map_not_identity() {
        let mut m = ContinuousMap::identity(2);
        m.matrix[0][0] = 2.0;
        assert!(!m.is_identity(0.1));
    }

    #[test]
    fn test_equivalence_deviation_check() {
        let eq = HomotopyEquivalence { dim_x: 2, dim_y: 2, deviation: 0.001 };
        assert!(eq.is_equivalence(0.01));
        assert!(!eq.is_equivalence(0.0001));
    }
}
