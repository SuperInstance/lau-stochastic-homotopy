//! Lifts and covers: when can a local policy change be extended globally?

use serde::{Deserialize, Serialize};
use crate::error::{HomotopyError, Result};
use crate::policy::{Policy, PolicyPath};

/// A covering space p: E → B.
/// A policy change in B can be lifted to E if the path doesn't cross branch points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoveringSpace {
    /// Base space dimension.
    pub base_dim: usize,
    /// Number of sheets (fiber cardinality).
    pub num_sheets: usize,
    /// Branch points where lifting fails.
    pub branch_points: Vec<Policy>,
    /// Tolerance for branch point detection.
    pub tolerance: f64,
}

impl CoveringSpace {
    /// Create a simple covering space.
    pub fn new(base_dim: usize, num_sheets: usize) -> Self {
        Self {
            base_dim,
            num_sheets,
            branch_points: Vec::new(),
            tolerance: 1e-6,
        }
    }

    /// Add a branch point.
    pub fn add_branch_point(&mut self, point: Policy) {
        self.branch_points.push(point);
    }

    /// Check if a point is near a branch point.
    pub fn is_branch_point(&self, point: &Policy) -> bool {
        self.branch_points.iter().any(|bp| bp.distance_to(point) < self.tolerance)
    }

    /// The fiber over a base point: the preimage p⁻¹(b).
    pub fn fiber(&self, base_point: &Policy) -> Vec<Policy> {
        (0..self.num_sheets)
            .map(|sheet| {
                let mut params = base_point.params.clone();
                // Lift: add sheet-dependent offset
                for (i, val) in params.iter_mut().enumerate() {
                    *val += (sheet as f64) * self.tolerance * (1.0 + (i as f64).sin());
                }
                Policy { params, label: Some(format!("sheet_{}", sheet)) }
            })
            .collect()
    }

    /// Lift a path from the base space to the covering space.
    /// Returns the lifted path starting at the given sheet.
    pub fn lift_path(&self, path: &PolicyPath, sheet: usize) -> Result<PolicyPath> {
        if sheet >= self.num_sheets {
            return Err(HomotopyError::LiftFailed);
        }

        // Check that the path doesn't pass through branch points
        for wp in &path.waypoints {
            if self.is_branch_point(wp) {
                return Err(HomotopyError::LiftFailed);
            }
        }

        // Lift each waypoint to the specified sheet
        let lifted: Vec<Policy> = path.waypoints.iter()
            .map(|wp| {
                let mut params = wp.params.clone();
                for (i, val) in params.iter_mut().enumerate() {
                    *val += (sheet as f64) * self.tolerance * (1.0 + (i as f64).sin());
                }
                Policy { params, label: Some(format!("sheet_{}", sheet)) }
            })
            .collect();

        PolicyPath::new(lifted)
    }

    /// The deck transformation group: automorphisms of the covering.
    pub fn deck_group_order(&self) -> usize {
        self.num_sheets
    }

    /// Check if a lifted path corresponds to a specific deck transformation.
    pub fn apply_deck_transform(&self, policy: &Policy, from_sheet: usize, to_sheet: usize) -> Policy {
        let mut params = policy.params.clone();
        let delta = (to_sheet as f64 - from_sheet as f64) * self.tolerance;
        for (i, val) in params.iter_mut().enumerate() {
            *val += delta * (1.0 + (i as f64).sin());
        }
        Policy { params, label: Some(format!("sheet_{}", to_sheet)) }
    }
}

/// A fiber bundle over policy space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiberBundle {
    /// Base space dimension.
    pub base_dim: usize,
    /// Fiber dimension.
    pub fiber_dim: usize,
    /// Is the bundle trivial (product)?
    pub is_trivial: bool,
}

impl FiberBundle {
    /// A trivial bundle: base × fiber.
    pub fn trivial(base_dim: usize, fiber_dim: usize) -> Self {
        Self { base_dim, fiber_dim, is_trivial: true }
    }

    /// A non-trivial bundle (e.g., Möbius strip, Hopf fibration).
    pub fn nontrivial(base_dim: usize, fiber_dim: usize) -> Self {
        Self { base_dim, fiber_dim, is_trivial: false }
    }

    /// Total space dimension.
    pub fn total_dim(&self) -> usize {
        self.base_dim + self.fiber_dim
    }

    /// Project from total space to base.
    pub fn project(&self, total_policy: &Policy) -> Policy {
        let base_params: Vec<f64> = total_policy.params.iter()
            .take(self.base_dim)
            .cloned()
            .collect();
        Policy::new(base_params)
    }

    /// Include base into total space (zero section).
    pub fn include(&self, base_policy: &Policy) -> Policy {
        let mut params = vec![0.0; self.total_dim()];
        for (i, val) in base_policy.params.iter().take(self.base_dim).enumerate() {
            params[i] = *val;
        }
        Policy::new(params)
    }
}

/// Monodromy: how the fiber changes as you go around a loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Monodromy {
    pub covering: CoveringSpace,
}

impl Monodromy {
    pub fn new(covering: CoveringSpace) -> Self {
        Self { covering }
    }

    /// Compute the monodromy permutation for a loop.
    /// Returns which sheets get permuted.
    pub fn compute(&self, _loop_path: &PolicyPath) -> Vec<usize> {
        // Simplified: identity permutation unless the loop encloses branch points
        (0..self.covering.num_sheets).collect()
    }

    /// Check if the monodromy is trivial (identity permutation).
    pub fn is_trivial(&self, permutation: &[usize]) -> bool {
        permutation.iter().enumerate().all(|(i, &j)| i == j)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covering_space_creation() {
        let cs = CoveringSpace::new(2, 3);
        assert_eq!(cs.num_sheets, 3);
        assert!(cs.branch_points.is_empty());
    }

    #[test]
    fn test_add_branch_point() {
        let mut cs = CoveringSpace::new(2, 2);
        cs.add_branch_point(Policy::new(vec![0.0, 0.0]));
        assert_eq!(cs.branch_points.len(), 1);
    }

    #[test]
    fn test_is_branch_point() {
        let mut cs = CoveringSpace::new(2, 2);
        cs.add_branch_point(Policy::new(vec![1.0, 1.0]));
        assert!(cs.is_branch_point(&Policy::new(vec![1.0, 1.0])));
        assert!(!cs.is_branch_point(&Policy::new(vec![5.0, 5.0])));
    }

    #[test]
    fn test_fiber() {
        let cs = CoveringSpace::new(2, 3);
        let base = Policy::new(vec![1.0, 2.0]);
        let fiber = cs.fiber(&base);
        assert_eq!(fiber.len(), 3);
    }

    #[test]
    fn test_lift_path() {
        let cs = CoveringSpace::new(2, 2);
        let path = PolicyPath::new(vec![
            Policy::new(vec![0.0, 0.0]),
            Policy::new(vec![1.0, 1.0]),
        ]).unwrap();
        let lifted = cs.lift_path(&path, 0);
        assert!(lifted.is_ok());
    }

    #[test]
    fn test_lift_through_branch_point_fails() {
        let mut cs = CoveringSpace::new(2, 2);
        cs.add_branch_point(Policy::new(vec![0.5, 0.5]));
        let path = PolicyPath::new(vec![
            Policy::new(vec![0.0, 0.0]),
            Policy::new(vec![0.5, 0.5]),
            Policy::new(vec![1.0, 1.0]),
        ]).unwrap();
        assert!(cs.lift_path(&path, 0).is_err());
    }

    #[test]
    fn test_lift_invalid_sheet() {
        let cs = CoveringSpace::new(2, 2);
        let path = PolicyPath::new(vec![Policy::new(vec![0.0, 0.0])]).unwrap();
        assert!(cs.lift_path(&path, 5).is_err());
    }

    #[test]
    fn test_deck_group_order() {
        let cs = CoveringSpace::new(2, 4);
        assert_eq!(cs.deck_group_order(), 4);
    }

    #[test]
    fn test_deck_transform() {
        let cs = CoveringSpace::new(2, 3);
        let p = Policy::new(vec![1.0, 2.0]);
        let transformed = cs.apply_deck_transform(&p, 0, 1);
        assert!(transformed.distance_to(&p) > 0.0);
    }

    #[test]
    fn test_trivial_fiber_bundle() {
        let fb = FiberBundle::trivial(3, 2);
        assert!(fb.is_trivial);
        assert_eq!(fb.total_dim(), 5);
    }

    #[test]
    fn test_nontrivial_fiber_bundle() {
        let fb = FiberBundle::nontrivial(2, 1);
        assert!(!fb.is_trivial);
    }

    #[test]
    fn test_bundle_project() {
        let fb = FiberBundle::trivial(2, 1);
        let total = Policy::new(vec![1.0, 2.0, 3.0]);
        let base = fb.project(&total);
        assert_eq!(base.dim(), 2);
    }

    #[test]
    fn test_bundle_include() {
        let fb = FiberBundle::trivial(2, 1);
        let base = Policy::new(vec![1.0, 2.0]);
        let total = fb.include(&base);
        assert_eq!(total.dim(), 3);
    }

    #[test]
    fn test_monodromy() {
        let cs = CoveringSpace::new(2, 3);
        let mono = Monodromy::new(cs);
        let base = Policy::new(vec![0.0, 0.0]);
        let path = PolicyPath::new(vec![base.clone(), base.clone()]).unwrap();
        let perm = mono.compute(&path);
        assert_eq!(perm.len(), 3);
    }

    #[test]
    fn test_monodromy_trivial() {
        let perm = vec![0, 1, 2];
        let cs = CoveringSpace::new(2, 3);
        let mono = Monodromy::new(cs);
        assert!(mono.is_trivial(&perm));
    }
}
