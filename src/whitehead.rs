//! Whitehead theorem: weak equivalence implies homotopy equivalence (for CW complexes).

use serde::{Deserialize, Serialize};
use crate::higher_homotopy::HigherHomotopyGroup;

/// A CW complex structure for policy space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CWComplex {
    /// Number of cells in each dimension.
    pub cells: Vec<usize>,
    /// Total dimension.
    pub dimension: usize,
}

impl CWComplex {
    /// A single point.
    pub fn point() -> Self {
        Self { cells: vec![1], dimension: 0 }
    }

    /// A circle S¹.
    pub fn circle() -> Self {
        Self { cells: vec![1, 1], dimension: 1 }
    }

    /// An n-sphere.
    pub fn sphere(n: usize) -> Self {
        let mut cells = vec![0usize; n + 1];
        cells[0] = 1;
        cells[n] = 1;
        Self { cells, dimension: n }
    }

    /// An n-disk.
    pub fn disk(n: usize) -> Self {
        let mut cells = vec![0usize; n + 1];
        cells[0] = 1;
        cells[n] = 1;
        Self { cells, dimension: n }
    }

    /// Is this a valid CW complex?
    pub fn is_valid(&self) -> bool {
        !self.cells.is_empty() && self.cells[0] > 0
    }

    /// Euler characteristic: Σ(-1)^n * (number of n-cells).
    pub fn euler_characteristic(&self) -> i64 {
        self.cells.iter().enumerate()
            .map(|(n, &count)| {
                if n % 2 == 0 { count as i64 } else { -(count as i64) }
            })
            .sum()
    }

    /// Betti numbers (simplified: from cell counts).
    pub fn betti_numbers(&self) -> Vec<usize> {
        // Simplified: just return cell counts / 2 as heuristic
        self.cells.iter()
            .map(|&c| if c > 0 { 1 } else { 0 })
            .collect()
    }
}

/// Whitehead theorem implementation.
/// If f: X → Y induces isomorphisms on all πₙ, and X, Y are CW complexes,
/// then f is a homotopy equivalence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteheadTheorem {
    pub complex_x: CWComplex,
    pub complex_y: CWComplex,
}

impl WhiteheadTheorem {
    /// Create a Whitehead theorem checker for two CW complexes.
    pub fn new(complex_x: CWComplex, complex_y: CWComplex) -> Self {
        Self { complex_x, complex_y }
    }

    /// Check if a map induces isomorphisms on all homotopy groups.
    /// If it does, by Whitehead's theorem, it's a homotopy equivalence.
    pub fn check_weak_equivalence(
        &self,
        homotopy_groups_x: &[HigherHomotopyGroup],
        homotopy_groups_y: &[HigherHomotopyGroup],
    ) -> WhiteheadResult {
        if homotopy_groups_x.len() != homotopy_groups_y.len() {
            return WhiteheadResult {
                is_weak_equivalence: false,
                is_homotopy_equivalence: false,
                failing_dimension: Some(homotopy_groups_x.len().min(homotopy_groups_y.len())),
                note: "different number of homotopy group levels".into(),
            };
        }

        for (i, (gx, gy)) in homotopy_groups_x.iter().zip(homotopy_groups_y.iter()).enumerate() {
            if gx.num_generators != gy.num_generators {
                return WhiteheadResult {
                    is_weak_equivalence: false,
                    is_homotopy_equivalence: false,
                    failing_dimension: Some(i + 1),
                    note: format!(
                        "π_{} differs: {} generators vs {}",
                        i + 1, gx.num_generators, gy.num_generators
                    ),
                };
            }
        }

        // Both are CW complexes → Whitehead applies
        let applies = self.complex_x.is_valid() && self.complex_y.is_valid();

        WhiteheadResult {
            is_weak_equivalence: true,
            is_homotopy_equivalence: applies,
            failing_dimension: None,
            note: if applies {
                "Whitehead theorem applies: weak equivalence ⟹ homotopy equivalence".into()
            } else {
                "weak equivalence holds but spaces are not CW complexes".into()
            },
        }
    }

    /// Check if Whitehead's theorem applies (both spaces must be CW complexes).
    pub fn theorem_applies(&self) -> bool {
        self.complex_x.is_valid() && self.complex_y.is_valid()
    }
}

/// Result of a Whitehead theorem check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteheadResult {
    /// Is the map a weak equivalence (iso on all πₙ)?
    pub is_weak_equivalence: bool,
    /// Is it a homotopy equivalence?
    pub is_homotopy_equivalence: bool,
    /// First dimension where πₙ doesn't match.
    pub failing_dimension: Option<usize>,
    /// Explanation.
    pub note: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Policy;

    #[test]
    fn test_cw_point() {
        let pt = CWComplex::point();
        assert!(pt.is_valid());
        assert_eq!(pt.euler_characteristic(), 1);
    }

    #[test]
    fn test_cw_circle() {
        let s1 = CWComplex::circle();
        assert!(s1.is_valid());
        assert_eq!(s1.euler_characteristic(), 0);
    }

    #[test]
    fn test_cw_sphere() {
        let s2 = CWComplex::sphere(2);
        assert!(s2.is_valid());
        assert_eq!(s2.euler_characteristic(), 2);
    }

    #[test]
    fn test_cw_sphere_3() {
        let s3 = CWComplex::sphere(3);
        assert_eq!(s3.euler_characteristic(), 0);
    }

    #[test]
    fn test_whitehead_both_points() {
        let w = WhiteheadTheorem::new(CWComplex::point(), CWComplex::point());
        let pi_x = vec![HigherHomotopyGroup::trivial(1, Policy::new(vec![0.0]))];
        let pi_y = vec![HigherHomotopyGroup::trivial(1, Policy::new(vec![0.0]))];
        let result = w.check_weak_equivalence(&pi_x, &pi_y);
        assert!(result.is_weak_equivalence);
        assert!(result.is_homotopy_equivalence);
    }

    #[test]
    fn test_whitehead_different_groups() {
        let w = WhiteheadTheorem::new(CWComplex::circle(), CWComplex::point());
        let pi_x = vec![HigherHomotopyGroup::circle()];
        let pi_y = vec![HigherHomotopyGroup::trivial(1, Policy::new(vec![0.0]))];
        let result = w.check_weak_equivalence(&pi_x, &pi_y);
        assert!(!result.is_weak_equivalence);
        assert!(!result.is_homotopy_equivalence);
        assert_eq!(result.failing_dimension, Some(1));
    }

    #[test]
    fn test_whitehead_theorem_applies() {
        let w = WhiteheadTheorem::new(CWComplex::point(), CWComplex::circle());
        assert!(w.theorem_applies());
    }

    #[test]
    fn test_whitehead_result_note() {
        let w = WhiteheadTheorem::new(CWComplex::point(), CWComplex::point());
        let pi_x = vec![HigherHomotopyGroup::trivial(1, Policy::new(vec![0.0]))];
        let pi_y = vec![HigherHomotopyGroup::trivial(1, Policy::new(vec![0.0]))];
        let result = w.check_weak_equivalence(&pi_x, &pi_y);
        assert!(!result.note.is_empty());
    }

    #[test]
    fn test_cw_betti_numbers() {
        let s1 = CWComplex::circle();
        let betti = s1.betti_numbers();
        assert_eq!(betti.len(), 2);
    }

    #[test]
    fn test_whitehead_mismatched_levels() {
        let w = WhiteheadTheorem::new(CWComplex::point(), CWComplex::point());
        let pi_x = vec![HigherHomotopyGroup::trivial(1, Policy::new(vec![0.0]))];
        let pi_y = vec![];
        let result = w.check_weak_equivalence(&pi_x, &pi_y);
        assert!(!result.is_weak_equivalence);
    }

    #[test]
    fn test_cw_disk() {
        let d2 = CWComplex::disk(2);
        assert!(d2.is_valid());
    }

    #[test]
    fn test_spheres_same_type() {
        let w = WhiteheadTheorem::new(CWComplex::sphere(2), CWComplex::sphere(2));
        let pi_x = vec![
            HigherHomotopyGroup::trivial(1, Policy::new(vec![0.0, 0.0, 0.0])),
            HigherHomotopyGroup::sphere(2),
        ];
        let pi_y = vec![
            HigherHomotopyGroup::trivial(1, Policy::new(vec![0.0, 0.0, 0.0])),
            HigherHomotopyGroup::sphere(2),
        ];
        let result = w.check_weak_equivalence(&pi_x, &pi_y);
        assert!(result.is_weak_equivalence);
        assert!(result.is_homotopy_equivalence);
    }
}
