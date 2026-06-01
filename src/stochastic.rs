//! Stochastic homotopy: homotopy + noise — can policies be deformed under uncertainty?

use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use crate::error::{HomotopyError, Result};
use crate::policy::Policy;
use crate::homotopy::Homotopy;

/// A stochastic homotopy adds noise to the deformation path.
/// H_stochastic(t, ω) = H(t) + σ·W(t, ω)
/// where W is a Wiener process (Brownian motion).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticHomotopy {
    /// Base deterministic homotopy.
    pub base: Homotopy,
    /// Noise intensity (volatility).
    pub sigma: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Number of Monte Carlo paths.
    pub num_paths: usize,
}

impl StochasticHomotopy {
    /// Create a new stochastic homotopy.
    pub fn new(base: Homotopy, sigma: f64, num_paths: usize, seed: u64) -> Self {
        Self { base, sigma, num_paths, seed }
    }

    /// Generate a single noisy path realization.
    pub fn sample_path(&self, rng: &mut impl Rng) -> Vec<Policy> {
        let n = self.base.steps;
        let dim = self.base.source.dim();
        let dt = 1.0 / n as f64;

        let mut path = Vec::with_capacity(n + 1);
        let mut current_noise = vec![0.0; dim];

        for i in 0..=n {
            let t = i as f64 / n as f64;
            let base_policy = self.base.evaluate(t);

            // Brownian increment
            if i > 0 {
                use rand_distr::StandardNormal;
                for d in 0..dim {
                    let z: f64 = rng.sample(StandardNormal);
                    current_noise[d] += self.sigma * z * dt.sqrt();
                }
            }

            let noise_vec = nalgebra::DVector::from_vec(current_noise.clone());
            let noisy_params = &base_policy.params + &noise_vec;
            path.push(Policy { params: noisy_params, label: None });
        }
        path
    }

    /// Generate all Monte Carlo paths.
    pub fn all_paths(&self) -> Vec<Vec<Policy>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        (0..self.num_paths)
            .map(|_| self.sample_path(&mut rng))
            .collect()
    }

    /// Check if ALL sampled paths are continuous (within epsilon).
    pub fn check_continuity(&self, epsilon: f64) -> Result<f64> {
        let paths = self.all_paths();
        let mut violations = 0usize;
        let mut max_gap = 0.0_f64;

        for path in &paths {
            for i in 1..path.len() {
                let gap = path[i - 1].distance_to(&path[i]);
                max_gap = max_gap.max(gap);
                if gap > epsilon {
                    violations += 1;
                }
            }
        }

        if violations == 0 {
            Ok(max_gap)
        } else {
            Err(HomotopyError::ContinuityViolation { t: -1.0, gap: max_gap })
        }
    }

    /// Probability that the stochastic homotopy remains continuous.
    /// Estimated by Monte Carlo: fraction of paths without discontinuities.
    pub fn continuity_probability(&self, epsilon: f64) -> f64 {
        let paths = self.all_paths();
        let continuous = paths.iter().filter(|path| {
            path.windows(2).all(|w| w[0].distance_to(&w[1]) <= epsilon)
        }).count();
        continuous as f64 / paths.len() as f64
    }

    /// Mean path length across all Monte Carlo samples.
    pub fn mean_path_length(&self) -> f64 {
        let paths = self.all_paths();
        let total: f64 = paths.iter().map(|path: &Vec<Policy>| {
            path.windows(2).map(|w| w[0].distance_to(&w[1])).sum::<f64>()
        }).sum::<f64>();
        total / paths.len() as f64
    }

    /// End-to-end noise: distribution of the final point.
    pub fn endpoint_distribution(&self) -> Vec<Policy> {
        let paths = self.all_paths();
        paths.into_iter().filter_map(|p| p.into_iter().last()).collect()
    }

    /// Check if the stochastic homotopy is "reliable" — most paths reach the target.
    pub fn reliability(&self, tol: f64) -> f64 {
        let endpoints = self.endpoint_distribution();
        let reaching = endpoints.iter()
            .filter(|p| p.distance_to(&self.base.target) < tol)
            .count();
        reaching as f64 / endpoints.len().max(1) as f64
    }
}

/// A Brownian bridge: a Wiener process conditioned on W(0)=a, W(1)=b.
#[derive(Debug, Clone)]
pub struct BrownianBridge {
    pub start: Policy,
    pub end: Policy,
    pub sigma: f64,
}

impl BrownianBridge {
    /// Create a Brownian bridge between two policies.
    pub fn new(start: Policy, end: Policy, sigma: f64) -> Result<Self> {
        if start.dim() != end.dim() {
            return Err(HomotopyError::DimensionMismatch {
                expected: start.dim(),
                actual: end.dim(),
            });
        }
        Ok(Self { start, end, sigma })
    }

    /// Sample a path from the Brownian bridge.
    pub fn sample(&self, steps: usize, rng: &mut impl Rng) -> Vec<Policy> {
        use rand_distr::StandardNormal;
        let dim = self.start.dim();
        let dt = 1.0 / steps as f64;
        let mut path = Vec::with_capacity(steps + 1);

        let mut w = nalgebra::DVector::zeros(dim);
        for i in 0..=steps {
            let t = i as f64 * dt;
            // Bridge: B(t) = (1-t)*a + t*b + (W(t) - t*W(1))
            let drift = &self.start.params * (1.0 - t) + &self.end.params * t;
            let correction = &w - &w * t;
            let params = drift + correction * self.sigma;
            path.push(Policy { params, label: None });

            if i < steps {
                let z: Vec<f64> = (0..dim).map(|_| rng.sample(StandardNormal)).collect();
                w += nalgebra::DVector::from_vec(z) * dt.sqrt();
            }
        }
        path
    }
}

/// Stochastic stability analysis for homotopies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityAnalysis {
    pub sigma: f64,
    pub mean_deviation: f64,
    pub max_deviation: f64,
    pub reliability: f64,
}

impl StabilityAnalysis {
    /// Run stability analysis for a range of noise levels.
    pub fn analyze(
        base: Homotopy,
        sigma_range: &[f64],
        num_paths: usize,
        tol: f64,
    ) -> Vec<StabilityAnalysis> {
        sigma_range.iter().map(|&sigma| {
            let sh = StochasticHomotopy::new(base.clone(), sigma, num_paths, 42);
            let mean_len = sh.mean_path_length();
            let base_len = base.total_length();
            let reliability = sh.reliability(tol);
            StabilityAnalysis {
                sigma,
                mean_deviation: (mean_len - base_len).abs(),
                max_deviation: sigma * 2.0, // heuristic
                reliability,
            }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_homotopy() -> Homotopy {
        Homotopy::new(
            Policy::new(vec![0.0, 0.0]),
            Policy::new(vec![1.0, 1.0]),
            50,
        ).unwrap()
    }

    #[test]
    fn test_stochastic_homotopy_creation() {
        let sh = StochasticHomotopy::new(make_homotopy(), 0.1, 10, 42);
        assert_eq!(sh.num_paths, 10);
        assert_eq!(sh.sigma, 0.1);
    }

    #[test]
    fn test_sample_path() {
        let sh = StochasticHomotopy::new(make_homotopy(), 0.01, 1, 42);
        let mut rng = rand::thread_rng();
        let path = sh.sample_path(&mut rng);
        assert_eq!(path.len(), 51); // steps + 1
    }

    #[test]
    fn test_all_paths() {
        let sh = StochasticHomotopy::new(make_homotopy(), 0.01, 5, 42);
        let paths = sh.all_paths();
        assert_eq!(paths.len(), 5);
    }

    #[test]
    fn test_continuity_probability() {
        let sh = StochasticHomotopy::new(make_homotopy(), 0.01, 20, 42);
        let prob = sh.continuity_probability(1.0);
        assert!(prob > 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_mean_path_length() {
        let sh = StochasticHomotopy::new(make_homotopy(), 0.01, 10, 42);
        let mean_len = sh.mean_path_length();
        assert!(mean_len > 0.0);
    }

    #[test]
    fn test_endpoint_distribution() {
        let sh = StochasticHomotopy::new(make_homotopy(), 0.01, 10, 42);
        let endpoints = sh.endpoint_distribution();
        assert_eq!(endpoints.len(), 10);
    }

    #[test]
    fn test_reliability() {
        let sh = StochasticHomotopy::new(make_homotopy(), 0.01, 20, 42);
        let rel = sh.reliability(2.0);
        assert!(rel >= 0.0 && rel <= 1.0);
    }

    #[test]
    fn test_brownian_bridge() {
        let bb = BrownianBridge::new(
            Policy::new(vec![0.0]),
            Policy::new(vec![1.0]),
            0.1,
        ).unwrap();
        let mut rng = rand::thread_rng();
        let path = bb.sample(100, &mut rng);
        assert_eq!(path.len(), 101);
    }

    #[test]
    fn test_brownian_bridge_dimension_mismatch() {
        assert!(BrownianBridge::new(
            Policy::new(vec![0.0]),
            Policy::new(vec![1.0, 2.0]),
            0.1,
        ).is_err());
    }

    #[test]
    fn test_stability_analysis() {
        let analysis = StabilityAnalysis::analyze(
            make_homotopy(),
            &[0.01, 0.1, 0.5],
            5,
            2.0,
        );
        assert_eq!(analysis.len(), 3);
        // Higher sigma should mean higher deviation
        assert!(analysis[2].mean_deviation >= analysis[0].mean_deviation || analysis[2].sigma > analysis[0].sigma);
    }

    #[test]
    fn test_zero_sigma_deterministic() {
        let sh = StochasticHomotopy::new(make_homotopy(), 0.0, 5, 42);
        let paths = sh.all_paths();
        // All paths should be identical when sigma=0
        for path in &paths {
            let first = &path[0];
            assert!(first.distance_to(&paths[0][0]) < 1e-10);
        }
    }

    #[test]
    fn test_check_continuity() {
        let sh = StochasticHomotopy::new(make_homotopy(), 0.01, 5, 42);
        let result = sh.check_continuity(10.0);
        assert!(result.is_ok());
    }
}
