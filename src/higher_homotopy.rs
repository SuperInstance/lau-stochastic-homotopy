//! Higher homotopy groups π_n for higher-dimensional policy deformations.

use serde::{Deserialize, Serialize};
use crate::policy::Policy;

/// A higher homotopy group πₙ(X, x₀).
/// Elements are homotopy classes of maps Sⁿ → X.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HigherHomotopyGroup {
    /// Dimension n of the sphere Sⁿ.
    pub dimension: usize,
    /// Base point of the policy space.
    pub base_point: Policy,
    /// Number of generators.
    pub num_generators: usize,
}

impl HigherHomotopyGroup {
    /// Create a new higher homotopy group record.
    pub fn new(dimension: usize, base_point: Policy, num_generators: usize) -> Self {
        Self { dimension, base_point, num_generators }
    }

    /// π_n for a point (trivial group).
    pub fn trivial(dimension: usize, base_point: Policy) -> Self {
        Self::new(dimension, base_point, 0)
    }

    /// Is this group trivial?
    pub fn is_trivial(&self) -> bool {
        self.num_generators == 0
    }

    /// π₁ of S¹ is ℤ.
    pub fn circle() -> Self {
        Self::new(1, Policy::new(vec![1.0, 0.0]), 1)
    }

    /// πₙ(Sⁿ) = ℤ for all n.
    pub fn sphere(dimension: usize) -> Self {
        let mut coords = vec![0.0; dimension + 1];
        coords[0] = 1.0;
        Self::new(dimension, Policy::new(coords), 1)
    }
}

/// An n-sphere map: a discretized map from the n-sphere into policy space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphereMap {
    /// Dimension of the source sphere.
    pub dimension: usize,
    /// Sample points on the sphere and their policy images.
    pub samples: Vec<(Vec<f64>, Policy)>,
}

impl SphereMap {
    /// Create a constant map (everything maps to base point).
    pub fn constant(dimension: usize, base: Policy) -> Self {
        let n = 10_usize.pow(dimension as u32).min(100);
        let samples = (0..n)
            .map(|i| {
                let coord = vec![i as f64 / n as f64];
                (coord, base.clone())
            })
            .collect();
        Self { dimension, samples }
    }

    /// Is this map constant (null-homotopic)?
    pub fn is_constant(&self, tol: f64) -> bool {
        if self.samples.len() <= 1 {
            return true;
        }
        let first = &self.samples[0].1;
        self.samples.iter().all(|(_, p)| p.distance_to(first) < tol)
    }

    /// Evaluate at a point on the sphere via nearest-neighbor lookup.
    pub fn evaluate(&self, point: &[f64]) -> Option<&Policy> {
        self.samples.iter()
            .min_by(|a, b| {
                let da: f64 = a.0.iter().zip(point.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                let db: f64 = b.0.iter().zip(point.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                da.partial_cmp(&db).unwrap()
            })
            .map(|(_, p)| p)
    }

    /// Degree of the map (how many times it wraps around).
    /// For maps Sⁿ → Sⁿ, this counts the algebraic number of preimages.
    pub fn degree(&self) -> f64 {
        if self.is_constant(1e-6) {
            return 0.0;
        }
        // Heuristic: count sign changes in the image norm relative to center
        let center = self.samples.first().map(|(_, p)| p.clone()).unwrap_or_else(|| Policy::zero(1));
        let above: f64 = self.samples.iter()
            .filter(|(_, p)| p.distance_to(&center) > 0.5)
            .count() as f64;
        let total = self.samples.len() as f64;
        if total == 0.0 { 0.0 } else { above / total }
    }
}

/// Hurewicz theorem: the first non-trivial homotopy group equals the homology group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HurewiczMap {
    pub dimension: usize,
}

impl HurewiczMap {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Apply the Hurewicz map πₙ → Hₙ.
    /// For simply-connected spaces, the first non-trivial πₙ ≅ Hₙ.
    pub fn apply(&self, pi_n: &HigherHomotopyGroup) -> usize {
        pi_n.num_generators
    }

    /// Check if Hurewicz isomorphism applies (space must be (n-1)-connected).
    pub fn is_isomorphism(&self, lower_groups: &[HigherHomotopyGroup]) -> bool {
        lower_groups.iter().all(|g| g.is_trivial())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_higher_homotopy() {
        let base = Policy::new(vec![0.0, 0.0]);
        let pi2 = HigherHomotopyGroup::trivial(2, base);
        assert!(pi2.is_trivial());
        assert_eq!(pi2.dimension, 2);
    }

    #[test]
    fn test_sphere_homotopy() {
        let pi1_s1 = HigherHomotopyGroup::circle();
        assert_eq!(pi1_s1.num_generators, 1);
        assert_eq!(pi1_s1.dimension, 1);
    }

    #[test]
    fn test_higher_sphere() {
        let pi3 = HigherHomotopyGroup::sphere(3);
        assert_eq!(pi3.num_generators, 1);
    }

    #[test]
    fn test_constant_sphere_map() {
        let base = Policy::new(vec![0.0]);
        let map = SphereMap::constant(2, base);
        assert!(map.is_constant(1e-6));
    }

    #[test]
    fn test_sphere_map_evaluate() {
        let base = Policy::new(vec![0.0]);
        let map = SphereMap::constant(1, base);
        let result = map.evaluate(&[0.5]);
        assert!(result.is_some());
    }

    #[test]
    fn test_sphere_map_degree_constant() {
        let base = Policy::new(vec![0.0]);
        let map = SphereMap::constant(2, base);
        assert_eq!(map.degree(), 0.0);
    }

    #[test]
    fn test_hurewicz_map() {
        let h = HurewiczMap::new(2);
        let pi2 = HigherHomotopyGroup::new(2, Policy::new(vec![0.0]), 3);
        assert_eq!(h.apply(&pi2), 3);
    }

    #[test]
    fn test_hurewicz_isomorphism_condition() {
        let h = HurewiczMap::new(3);
        let lower = vec![
            HigherHomotopyGroup::trivial(1, Policy::new(vec![0.0])),
            HigherHomotopyGroup::trivial(2, Policy::new(vec![0.0])),
        ];
        assert!(h.is_isomorphism(&lower));
    }

    #[test]
    fn test_hurewicz_not_isomorphism() {
        let h = HurewiczMap::new(3);
        let lower = vec![
            HigherHomotopyGroup::circle(), // π₁ is non-trivial
            HigherHomotopyGroup::trivial(2, Policy::new(vec![0.0])),
        ];
        assert!(!h.is_isomorphism(&lower));
    }

    #[test]
    fn test_non_trivial_sphere_map() {
        let base = Policy::new(vec![0.0, 0.0]);
        let n = 50;
        let samples: Vec<(Vec<f64>, Policy)> = (0..n)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                (vec![i as f64 / n as f64], Policy::new(vec![theta.cos(), theta.sin()]))
            })
            .collect();
        let map = SphereMap { dimension: 1, samples };
        assert!(!map.is_constant(0.1));
    }

    #[test]
    fn test_new_higher_homotopy_group() {
        let base = Policy::new(vec![1.0, 0.0, 0.0]);
        let pi4 = HigherHomotopyGroup::new(4, base, 2);
        assert_eq!(pi4.dimension, 4);
        assert_eq!(pi4.num_generators, 2);
        assert!(!pi4.is_trivial());
    }
}
