//! Obstruction theory: what prevents extending a local deformation to a global one?

use serde::{Deserialize, Serialize};
use crate::error::{HomotopyError, Result};
use crate::policy::Policy;

/// An obstruction class: an element of a cohomology group that measures
/// whether a map can be extended over the next skeleton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObstructionClass {
    /// Dimension of the obstruction (lives in Hⁿ⁺¹).
    pub dimension: usize,
    /// The obstruction value (simplified as a single number).
    pub value: f64,
    /// Description.
    pub description: String,
}

/// Obstruction theory determines if a local policy deformation can be
/// extended globally by checking cohomological obstructions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Obstruction {
    /// Policy space dimension.
    pub space_dim: usize,
    /// Collected obstruction classes.
    pub obstructions: Vec<ObstructionClass>,
}

impl Obstruction {
    /// Create a new obstruction analyzer.
    pub fn new(space_dim: usize) -> Self {
        Self { space_dim, obstructions: Vec::new() }
    }

    /// Compute the primary obstruction to extending a map f: X^(n) → Y
    /// from the n-skeleton to the (n+1)-skeleton.
    pub fn primary_obstruction(
        &mut self,
        local_policies: &[Policy],
        global_target: &Policy,
    ) -> Result<()> {
        // The obstruction is non-zero if local policies can't be consistently
        // extended to the target
        let distances: Vec<f64> = local_policies.iter()
            .map(|p| p.distance_to(global_target))
            .collect();

        let max_dist = distances.iter().fold(0.0_f64, |a, &b| a.max(b));
        let mean_dist = distances.iter().sum::<f64>() / distances.len().max(1) as f64;

        if max_dist > 0.0 {
            self.obstructions.push(ObstructionClass {
                dimension: 1,
                value: max_dist,
                description: format!(
                    "primary obstruction: max distance {:.4}, mean {:.4}",
                    max_dist, mean_dist
                ),
            });
        }

        Ok(())
    }

    /// Compute the secondary obstruction (dimension 2).
    pub fn secondary_obstruction(
        &mut self,
        policies: &[Policy],
        constraints: &[(usize, usize, f64)], // (i, j, max_gap) pairs
    ) -> Result<()> {
        let mut max_violation = 0.0_f64;
        for &(i, j, max_gap) in constraints {
            if i < policies.len() && j < policies.len() {
                let gap = policies[i].distance_to(&policies[j]);
                if gap > max_gap {
                    max_violation = max_violation.max(gap - max_gap);
                }
            }
        }

        if max_violation > 0.0 {
            self.obstructions.push(ObstructionClass {
                dimension: 2,
                value: max_violation,
                description: format!("secondary obstruction: constraint violation {:.4}", max_violation),
            });
        }

        Ok(())
    }

    /// Check if all obstructions vanish (extension is possible).
    pub fn all_obstructions_vanish(&self) -> bool {
        self.obstructions.is_empty() || self.obstructions.iter().all(|o| o.value.abs() < 1e-10)
    }

    /// Get the total obstruction (sum of absolute values).
    pub fn total_obstruction(&self) -> f64 {
        self.obstructions.iter().map(|o| o.value.abs()).sum()
    }

    /// Get obstructions at a specific dimension.
    pub fn obstructions_at(&self, dim: usize) -> Vec<&ObstructionClass> {
        self.obstructions.iter().filter(|o| o.dimension == dim).collect()
    }

    /// Clear all obstructions.
    pub fn clear(&mut self) {
        self.obstructions.clear();
    }

    /// Check if a local homotopy can be extended to a global one.
    /// Uses the Extension Lemma: extension exists iff all obstructions vanish.
    pub fn can_extend(&self) -> Result<()> {
        if self.all_obstructions_vanish() {
            Ok(())
        } else {
            let dim = self.obstructions.first().map(|o| o.dimension).unwrap_or(0);
            Err(HomotopyError::ObstructionNonZero { class: dim })
        }
    }
}

/// Postnikov tower: a sequence of approximations to a space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostnikovTower {
    pub stages: Vec<PostnikovStage>,
}

/// A stage in the Postnikov tower.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostnikovStage {
    /// Truncation level n: this stage knows about π_k for k ≤ n.
    pub n: usize,
    /// k-invariants: cohomology classes that determine how to build the next stage.
    pub k_invariants: Vec<f64>,
}

impl PostnikovTower {
    /// Build a Postnikov tower for a space with given homotopy groups.
    pub fn build(homotopy_ranks: &[usize]) -> Self {
        let stages: Vec<PostnikovStage> = homotopy_ranks.iter().enumerate()
            .map(|(n, &rank)| {
                PostnikovStage {
                    n: n + 1,
                    k_invariants: if rank > 0 { vec![1.0] } else { vec![] },
                }
            })
            .collect();
        Self { stages }
    }

    /// Number of stages.
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// The n-th stage captures all homotopy info up to dimension n.
    pub fn stage(&self, n: usize) -> Option<&PostnikovStage> {
        self.stages.get(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obstruction_new() {
        let obs = Obstruction::new(3);
        assert!(obs.obstructions.is_empty());
        assert!(obs.all_obstructions_vanish());
    }

    #[test]
    fn test_primary_obstruction_vanishes() {
        let mut obs = Obstruction::new(2);
        let policies = vec![Policy::new(vec![1.0, 1.0])];
        let target = Policy::new(vec![1.0, 1.0]);
        obs.primary_obstruction(&policies, &target).unwrap();
        assert!(obs.all_obstructions_vanish());
    }

    #[test]
    fn test_primary_obstruction_nonzero() {
        let mut obs = Obstruction::new(2);
        let policies = vec![Policy::new(vec![0.0, 0.0]), Policy::new(vec![10.0, 10.0])];
        let target = Policy::new(vec![5.0, 5.0]);
        obs.primary_obstruction(&policies, &target).unwrap();
        assert!(!obs.all_obstructions_vanish());
    }

    #[test]
    fn test_total_obstruction() {
        let mut obs = Obstruction::new(2);
        let policies = vec![Policy::new(vec![0.0])];
        let target = Policy::new(vec![3.0]);
        obs.primary_obstruction(&policies, &target).unwrap();
        assert!(obs.total_obstruction() > 0.0);
    }

    #[test]
    fn test_secondary_obstruction() {
        let mut obs = Obstruction::new(2);
        let policies = vec![Policy::new(vec![0.0]), Policy::new(vec![5.0])];
        let constraints = vec![(0, 1, 1.0)]; // max gap of 1.0 between policies 0 and 1
        obs.secondary_obstruction(&policies, &constraints).unwrap();
        assert!(!obs.all_obstructions_vanish());
    }

    #[test]
    fn test_can_extend_no_obstruction() {
        let obs = Obstruction::new(2);
        assert!(obs.can_extend().is_ok());
    }

    #[test]
    fn test_can_extend_with_obstruction() {
        let mut obs = Obstruction::new(2);
        obs.obstructions.push(ObstructionClass {
            dimension: 1,
            value: 5.0,
            description: "test".into(),
        });
        assert!(obs.can_extend().is_err());
    }

    #[test]
    fn test_clear_obstructions() {
        let mut obs = Obstruction::new(2);
        obs.obstructions.push(ObstructionClass {
            dimension: 1, value: 1.0, description: "test".into(),
        });
        obs.clear();
        assert!(obs.obstructions.is_empty());
    }

    #[test]
    fn test_obstructions_at_dimension() {
        let mut obs = Obstruction::new(2);
        obs.obstructions.push(ObstructionClass { dimension: 1, value: 1.0, description: "d1".into() });
        obs.obstructions.push(ObstructionClass { dimension: 2, value: 2.0, description: "d2".into() });
        assert_eq!(obs.obstructions_at(1).len(), 1);
        assert_eq!(obs.obstructions_at(2).len(), 1);
    }

    #[test]
    fn test_postnikov_tower() {
        let tower = PostnikovTower::build(&[1, 0, 1]); // π₁=ℤ, π₂=0, π₃=ℤ
        assert_eq!(tower.num_stages(), 3);
    }

    #[test]
    fn test_postnikov_stage() {
        let tower = PostnikovTower::build(&[1, 0, 1]);
        let stage0 = tower.stage(0);
        assert!(stage0.is_some());
        assert_eq!(stage0.unwrap().n, 1);
    }

    #[test]
    fn test_extension_lemma() {
        // No obstructions → extension possible
        let obs = Obstruction::new(2);
        assert!(obs.can_extend().is_ok());

        // Non-zero obstruction → extension blocked
        let mut obs2 = Obstruction::new(2);
        obs2.obstructions.push(ObstructionClass {
            dimension: 1, value: 42.0, description: "blocked".into(),
        });
        assert!(obs2.can_extend().is_err());
    }
}
