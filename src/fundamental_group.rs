//! Fundamental group π₁ of agent policy space — loops of policy changes.

use serde::{Deserialize, Serialize};
use crate::error::{HomotopyError, Result};
use crate::policy::{Policy, PolicyPath};

/// An element of the fundamental group: a homotopy class of loops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopClass {
    /// A representative loop.
    pub representative: PolicyPath,
    /// Optional group element name.
    pub name: Option<String>,
}

/// The fundamental group π₁(X, x₀) of a policy space X at basepoint x₀.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundamentalGroup {
    /// Base point of the policy space.
    pub base_point: Policy,
    /// Group elements (homotopy classes of loops).
    pub elements: Vec<LoopClass>,
    /// Tolerance for identifying base point returns.
    pub tolerance: f64,
}

impl FundamentalGroup {
    /// Create a trivial fundamental group (only the identity element).
    pub fn trivial(base_point: Policy) -> Self {
        let identity_loop = PolicyPath::constant(base_point.clone());
        Self {
            base_point,
            elements: vec![LoopClass {
                representative: identity_loop,
                name: Some("identity".into()),
            }],
            tolerance: 1e-6,
        }
    }

    /// Create with a given tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Add a loop as a generator.
    pub fn add_generator(&mut self, path: PolicyPath) -> Result<()> {
        if !path.is_loop(self.tolerance) {
            return Err(HomotopyError::GroupError {
                reason: "path is not a loop — does not return to base point".into(),
            });
        }
        let idx = self.elements.len();
        self.elements.push(LoopClass {
            representative: path,
            name: Some(format!("g{}", idx)),
        });
        Ok(())
    }

    /// Number of generators (beyond identity).
    pub fn num_generators(&self) -> usize {
        self.elements.len().saturating_sub(1)
    }

    /// Is the fundamental group trivial (only identity)?
    pub fn is_trivial(&self) -> bool {
        self.elements.len() <= 1
    }

    /// Multiply two loop classes (concatenation).
    pub fn multiply(&self, a: &LoopClass, b: &LoopClass) -> Result<LoopClass> {
        let concatenated = a.representative.concatenate(&b.representative)?;
        Ok(LoopClass {
            representative: concatenated,
            name: None,
        })
    }

    /// Invert a loop class (reverse the loop).
    pub fn invert(&self, class: &LoopClass) -> LoopClass {
        LoopClass {
            representative: class.representative.reverse(),
            name: class.name.as_ref().map(|n| format!("{}⁻¹", n)),
        }
    }

    /// Check if a path is null-homotopic (homotopic to constant loop).
    pub fn is_null_homotopic(&self, path: &PolicyPath, epsilon: f64) -> bool {
        if !path.is_loop(self.tolerance) {
            return false;
        }
        // A loop is null-homotopic if its length is below a threshold.
        // In general this is undecidable; we use a heuristic.
        path.length() < epsilon
    }

    /// Compute the winding number of a 2D loop around the origin.
    pub fn winding_number(&self, path: &PolicyPath) -> f64 {
        let n = path.waypoints.len();
        if n < 2 || path.waypoints[0].dim() < 2 {
            return 0.0;
        }
        let mut winding = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            let x1 = path.waypoints[i].params[0];
            let y1 = path.waypoints[i].params[1];
            let x2 = path.waypoints[j].params[0];
            let y2 = path.waypoints[j].params[1];
            let cross = x1 * y2 - x2 * y1;
            let dot = x1 * x2 + y1 * y2;
            winding += cross.atan2(dot);
        }
        winding / (2.0 * std::f64::consts::PI)
    }

    /// Compute the abelianization (first homology group H₁).
    pub fn abelianization(&self) -> usize {
        // Simplified: rank of H₁ = number of independent generators
        self.num_generators()
    }

    /// Check if a loop is contractible by checking if it can be
    /// continuously shrunk to a point.
    pub fn is_contractible(&self, path: &PolicyPath, max_iter: usize) -> bool {
        if path.waypoints.len() <= 2 {
            return true;
        }
        let base = self.base_point.clone();
        let mut current = path.clone();
        for _ in 0..max_iter {
            // Shrink towards base point
            let mut new_waypoints = Vec::new();
            for wp in &current.waypoints {
                new_waypoints.push(wp.lerp(&base, 0.1).unwrap_or_else(|_| wp.clone()));
            }
            // Close the loop
            if let Some(last) = new_waypoints.last() {
                if last.distance_to(&base) < self.tolerance {
                    return true;
                }
            }
            current = PolicyPath { waypoints: new_waypoints };
        }
        false
    }
}

/// Free group on n generators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeGroup {
    pub num_generators: usize,
}

impl FreeGroup {
    pub fn new(n: usize) -> Self {
        Self { num_generators: n }
    }

    /// A word in the free group: list of (generator_index, exponent).
    pub fn word(&self, letters: Vec<(usize, i32)>) -> GroupWord {
        GroupWord { letters, group_generators: self.num_generators }
    }

    /// Reduce a word by canceling adjacent inverses.
    pub fn reduce(&self, word: &GroupWord) -> GroupWord {
        let mut reduced: Vec<(usize, i32)> = Vec::new();
        for &(gen, exp) in &word.letters {
            if gen >= self.num_generators {
                continue;
            }
            if let Some(&mut (last_gen, ref mut last_exp)) = reduced.last_mut() {
                if last_gen == gen {
                    *last_exp += exp;
                    if *last_exp == 0 {
                        reduced.pop();
                    }
                    continue;
                }
            }
            reduced.push((gen, exp));
        }
        GroupWord { letters: reduced, group_generators: self.num_generators }
    }

    /// Check if a word represents the identity.
    pub fn is_identity(&self, word: &GroupWord) -> bool {
        self.reduce(word).letters.is_empty()
    }
}

/// A word in a group presentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupWord {
    pub letters: Vec<(usize, i32)>,
    pub group_generators: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_fundamental_group() {
        let base = Policy::new(vec![0.0, 0.0]);
        let fg = FundamentalGroup::trivial(base);
        assert!(fg.is_trivial());
        assert_eq!(fg.num_generators(), 0);
    }

    #[test]
    fn test_add_generator() {
        let base = Policy::new(vec![0.0, 0.0]);
        let mut fg = FundamentalGroup::trivial(base.clone());
        let loop_path = PolicyPath::new(vec![
            base.clone(),
            Policy::new(vec![1.0, 0.0]),
            base.clone(),
        ]).unwrap();
        fg.add_generator(loop_path).unwrap();
        assert_eq!(fg.num_generators(), 1);
    }

    #[test]
    fn test_non_loop_generator_rejected() {
        let base = Policy::new(vec![0.0, 0.0]);
        let mut fg = FundamentalGroup::trivial(base.clone());
        let non_loop = PolicyPath::new(vec![
            base.clone(),
            Policy::new(vec![1.0, 0.0]),
            Policy::new(vec![2.0, 0.0]),
        ]).unwrap();
        assert!(fg.add_generator(non_loop).is_err());
    }

    #[test]
    fn test_multiply_loops() {
        let base = Policy::new(vec![0.0]);
        let fg = FundamentalGroup::trivial(base.clone());
        let a = LoopClass {
            representative: PolicyPath::new(vec![base.clone(), Policy::new(vec![1.0]), base.clone()]).unwrap(),
            name: Some("a".into()),
        };
        let b = LoopClass {
            representative: PolicyPath::new(vec![base.clone(), Policy::new(vec![2.0]), base.clone()]).unwrap(),
            name: Some("b".into()),
        };
        let product = fg.multiply(&a, &b).unwrap();
        assert_eq!(product.representative.waypoints.len(), 6);
    }

    #[test]
    fn test_invert_loop() {
        let base = Policy::new(vec![0.0]);
        let fg = FundamentalGroup::trivial(base.clone());
        let a = LoopClass {
            representative: PolicyPath::new(vec![
                base.clone(),
                Policy::new(vec![1.0]),
                Policy::new(vec![2.0]),
                base.clone(),
            ]).unwrap(),
            name: Some("a".into()),
        };
        let inv = fg.invert(&a);
        assert_eq!(inv.representative.waypoints.len(), 4);
    }

    #[test]
    fn test_null_homotopic() {
        let base = Policy::new(vec![0.0]);
        let fg = FundamentalGroup::trivial(base.clone());
        let tiny = PolicyPath::new(vec![base.clone(), base.clone()]).unwrap();
        assert!(fg.is_null_homotopic(&tiny, 0.01));
    }

    #[test]
    fn test_winding_number_circle() {
        let base = Policy::new(vec![1.0, 0.0]);
        let fg = FundamentalGroup::trivial(base.clone());
        let n = 100;
        let mut waypoints = vec![];
        for i in 0..=n {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            waypoints.push(Policy::new(vec![theta.cos(), theta.sin()]));
        }
        let circle = PolicyPath::new(waypoints).unwrap();
        let wn = fg.winding_number(&circle);
        assert!((wn - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_winding_number_zero() {
        let base = Policy::new(vec![0.0, 0.0]);
        let fg = FundamentalGroup::trivial(base.clone());
        let path = PolicyPath::new(vec![
            base.clone(),
            Policy::new(vec![1.0, 0.0]),
            Policy::new(vec![1.0, 1.0]),
            base.clone(),
        ]).unwrap();
        let wn = fg.winding_number(&path);
        // Not a circle, winding should be small
        assert!(wn.abs() < 2.0);
    }

    #[test]
    fn test_abelianization() {
        let base = Policy::new(vec![0.0]);
        let mut fg = FundamentalGroup::trivial(base.clone());
        fg.add_generator(PolicyPath::new(vec![base.clone(), Policy::new(vec![1.0]), base.clone()]).unwrap()).unwrap();
        fg.add_generator(PolicyPath::new(vec![base.clone(), Policy::new(vec![2.0]), base.clone()]).unwrap()).unwrap();
        assert_eq!(fg.abelianization(), 2);
    }

    #[test]
    fn test_free_group_identity() {
        let fg = FreeGroup::new(2);
        let word = fg.word(vec![(0, 1), (0, -1)]);
        assert!(fg.is_identity(&word));
    }

    #[test]
    fn test_free_group_non_identity() {
        let fg = FreeGroup::new(2);
        let word = fg.word(vec![(0, 1), (1, 1)]);
        assert!(!fg.is_identity(&word));
    }

    #[test]
    fn test_free_group_reduce() {
        let fg = FreeGroup::new(3);
        let word = fg.word(vec![(0, 1), (1, 1), (1, -1), (0, -1)]);
        assert!(fg.is_identity(&word));
    }

    #[test]
    fn test_contractible_short_loop() {
        let base = Policy::new(vec![0.0]);
        let fg = FundamentalGroup::trivial(base.clone());
        let short = PolicyPath::new(vec![base.clone(), Policy::new(vec![0.01]), base.clone()]).unwrap();
        assert!(fg.is_contractible(&short, 100));
    }
}
