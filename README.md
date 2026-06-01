# lau-stochastic-homotopy

**Stochastic processes meet homotopy theory — determining whether agent policies can be safely deformed into each other under uncertainty.**

A Rust library that models agent policies as points in a continuous parameter space and uses algebraic topology (homotopy, fundamental groups, covering spaces, obstruction theory, Whitehead's theorem, Seifert–van Kampen) combined with stochastic analysis (Brownian motion, Monte Carlo reliability) to answer: *can an agent transition from one behavior to another without crossing a singularity?*

[![122 tests passing](https://img.shields.io/badge/tests-122%20passing-brightgreen)]()

---

## What This Does

In reinforcement learning and multi-agent systems, policies map states to actions. Two policies are *homotopic* if one can be continuously deformed into the other. This library:

1. **Represents policies** as points in ℝⁿ parameter space
2. **Constructs homotopies** (linear, spherical SLERP, Bézier) between policies
3. **Checks continuity** and validates deformation paths
4. **Adds stochastic noise** (Brownian motion) to test robustness under uncertainty
5. **Diagnoses obstructions** — cohomological barriers that prevent extending local deformations globally
6. **Computes fundamental groups** (π₁), higher homotopy groups (πₙ), and applies Whitehead's theorem
7. **Uses Seifert–van Kampen** to glue subspaces and compute combined topology
8. **Provides a high-level `PolicyTransitionChecker`** that answers: *is this transition safe?*

## Key Idea

A homotopy H: [0,1] × PolicySpace → PolicySpace continuously deforms policy f into policy g. If such H exists and remains continuous under noise, the transition is safe. If obstructions exist (non-vanishing cohomology classes), the transition is blocked. The stochastic layer adds Brownian motion W(t) to the homotopy path and estimates the probability that the noisy path remains continuous — the *reliability*.

## Install

```toml
[dependencies]
lau-stochastic-homotopy = "0.1.0"
```

Requires Rust 2021 edition. Dependencies: `nalgebra` (with serde), `serde`, `serde_json`, `rand`, `rand_distr`, `thiserror`.

## Quick Start

```rust
use lau_stochastic_homotopy::{Policy, PolicyTransitionChecker};

fn main() {
    // Define two policies
    let current = Policy::new(vec![0.0, 0.0, 0.0]);
    let target = Policy::new(vec![1.0, 0.5, 0.2]);

    // Check if transition is safe
    let checker = PolicyTransitionChecker::new();
    let result = checker.check(&current, &target);

    println!("Homotopic: {}", result.is_homotopic);
    println!("Path length: {:.4}", result.path_length);
    println!("Reliability: {:.2}%", result.reliability.unwrap_or(0.0) * 100.0);
    println!("Risk: {:?}", result.risk);
    println!("{}", result.summary);
}
```

## API Reference

### Module: `policy` — Policies as Points in Parameter Space

| Type / Method | Description |
|---|---|
| `Policy` | Parameter vector in continuous policy space |
| `Policy::new(params: Vec<f64>)` | Create from parameter vector |
| `Policy::labeled(params, label)` | Create with a label |
| `Policy::zero(dim)` | Origin of policy space |
| `.dim() → usize` | Parameter dimension |
| `.evaluate(state_idx) → f64` | Read parameter at index |
| `.distance_to(other) → f64` | L² distance |
| `.lerp(other, t) → Policy` | Linear interpolation at t ∈ [0,1] |
| `.normalize() → Policy` | Unit-length normalization |
| `.add_noise(sigma, rng) → Policy` | Gaussian perturbation |
| `.approx_eq(other, tol) → bool` | Approximate equality |
| `PolicyPath` | Parameterized path through policy space |
| `PolicyPath::new(waypoints)` | Create from waypoints |
| `PolicyPath::constant(policy)` | Single-point path |
| `.evaluate(t) → Policy` | Interpolate at t ∈ [0,1] |
| `.length() → f64` | Total path length |
| `.is_loop(tol) → bool` | Start ≈ end? |
| `.concatenate(other) → PolicyPath` | Join two paths |
| `.reverse() → PolicyPath` | Reverse direction |

### Module: `homotopy` — Continuous Deformations

| Type / Method | Description |
|---|---|
| `Homotopy` | H: [0,1] × PolicySpace → PolicySpace |
| `Homotopy::new(source, target, steps)` | Linear interpolation homotopy |
| `Homotopy::spherical(source, target, steps)` | SLERP (preserves norms) |
| `Homotopy::bezier(source, target, steps)` | Quadratic Bézier with curvature |
| `.evaluate(t) → Policy` | Evaluate at t ∈ [0,1] |
| `.path() → Vec<Policy>` | Full discretized path |
| `.check_continuity(epsilon) → Result` | Verify no large gaps |
| `.max_step_size() → f64` | Largest step in discretization |
| `.total_length() → f64` | Arc length |
| `.is_valid(epsilon) → bool` | Continuity check |
| `.compose(other) → Homotopy` | Chain two homotopies |
| `.reverse() → Homotopy` | Reverse direction |
| `.reparameterize(φ) → Homotopy` | Apply monotone reparametrization |
| `InterpolationType` | Linear, Spherical, Bezier |
| `RelativeHomotopy` | Homotopy fixing specified parameters |
| `RelativeHomotopy::new(homotopy, fixed_indices)` | Create with fixed dimensions |
| `.evaluate(t) → Policy` | Evaluate with fixed parameters held constant |

### Module: `stochastic` — Noisy Homotopies and Brownian Bridges

| Type / Method | Description |
|---|---|
| `StochasticHomotopy` | H(t,ω) = H(t) + σ·W(t,ω), W = Wiener process |
| `StochasticHomotopy::new(base, sigma, num_paths, seed)` | Create with noise level and MC paths |
| `.sample_path(rng) → Vec<Policy>` | Single noisy realization |
| `.all_paths() → Vec<Vec<Policy>>` | All Monte Carlo paths |
| `.check_continuity(epsilon) → Result<f64>` | All paths continuous? |
| `.continuity_probability(epsilon) → f64` | Fraction of continuous paths |
| `.mean_path_length() → f64` | Average arc length across samples |
| `.endpoint_distribution() → Vec<Policy>` | Final points of all paths |
| `.reliability(tol) → f64` | Fraction reaching target within tol |
| `BrownianBridge` | Wiener process conditioned on start and end |
| `BrownianBridge::new(start, end, sigma)` | Create bridge |
| `.sample(steps, rng) → Vec<Policy>` | Sample a bridge path |
| `StabilityAnalysis` | Sweep noise levels |
| `StabilityAnalysis::analyze(base, sigma_range, num_paths, tol)` | Reliability vs noise |

### Module: `fundamental_group` — π₁ of Policy Space

| Type / Method | Description |
|---|---|
| `FundamentalGroup` | π₁(X, x₀) — homotopy classes of loops |
| `FundamentalGroup::trivial(base_point)` | Trivial group (contractible space) |
| `.add_generator(path)` | Add a loop as generator |
| `.num_generators() → usize` | Count beyond identity |
| `.is_trivial() → bool` | Only identity element? |
| `.multiply(a, b) → LoopClass` | Concatenate two loops |
| `.invert(class) → LoopClass` | Reverse a loop |
| `.is_null_homotopic(path, epsilon) → bool` | Heuristic: is loop contractible? |
| `.winding_number(path) → f64` | Winding number of 2D loop |
| `.abelianization() → usize` | Rank of H₁ |
| `.is_contractible(path, max_iter) → bool` | Iterative shrinking test |
| `LoopClass` | Representative loop + name |
| `FreeGroup` | Free group on n generators |
| `FreeGroup::new(n)` | Create |
| `.word(letters) → GroupWord` | Build a word |
| `.reduce(word) → GroupWord` | Cancel adjacent inverses |
| `.is_identity(word) → bool` | Reduced word empty? |
| `GroupWord` | Word as list of (generator, exponent) |

### Module: `higher_homotopy` — Higher Homotopy Groups πₙ

| Type / Method | Description |
|---|---|
| `HigherHomotopyGroup` | πₙ(X, x₀) |
| `HigherHomotopyGroup::new(n, base_point, generators)` | Create |
| `HigherHomotopyGroup::trivial(n, base)` | Zero generators |
| `HigherHomotopyGroup::circle()` | π₁(S¹) = ℤ |
| `HigherHomotopyGroup::sphere(n)` | πₙ(Sⁿ) = ℤ |
| `.is_trivial() → bool` | No generators? |
| `SphereMap` | Discretized map Sⁿ → PolicySpace |
| `SphereMap::constant(n, base)` | Everything maps to base |
| `.is_constant(tol) → bool` | All images near each other? |
| `.evaluate(point) → Option<&Policy>` | Nearest-neighbor lookup |
| `.degree() → f64` | Heuristic degree of map |
| `HurewiczMap` | Hurewicz homomorphism πₙ → Hₙ |
| `HurewiczMap::new(n)` | Create |
| `.apply(πₙ) → usize` | Apply to homotopy group |
| `.is_isomorphism(lower_groups) → bool` | All lower groups trivial? |

### Module: `equivalence` — Homotopy Equivalence

| Type / Method | Description |
|---|---|
| `ContinuousMap` | Linear map between policy spaces |
| `ContinuousMap::identity(dim)` | Identity map |
| `ContinuousMap::constant(dim, value)` | Constant map |
| `.apply(policy) → Policy` | Apply transformation |
| `.compose(other) → ContinuousMap` | Compose maps |
| `.is_identity(tol) → bool` | Check if ≈ identity |
| `HomotopyEquivalence` | Records f: X→Y, g: Y→X with f∘g ≃ id, g∘f ≃ id |
| `HomotopyEquivalence::check_contractible(xs, ys)` | Contractible spaces always equivalent |
| `HomotopyEquivalence::points_equivalent()` | Single points equivalent |
| `HomotopyEquivalence::deformation_retract(space, subspace, steps)` | Check deformation retract |
| `.is_equivalence(tol) → bool` | Deviation below tolerance? |
| `.same_homotopy_type(rank_x, rank_y) → bool` | Same fundamental group rank? |
| `DeformationRetract` | Strong deformation retraction |
| `DeformationRetract::to_point(space, point_idx, steps)` | Retract entire space to one point |

### Module: `lifts` — Covering Spaces and Fiber Bundles

| Type / Method | Description |
|---|---|
| `CoveringSpace` | p: E → B with branch points |
| `CoveringSpace::new(base_dim, num_sheets)` | Create |
| `.add_branch_point(point)` | Add a branch point |
| `.is_branch_point(point) → bool` | Check proximity to branch point |
| `.fiber(base_point) → Vec<Policy>` | Preimage p⁻¹(b) |
| `.lift_path(path, sheet) → PolicyPath` | Lift path to specified sheet |
| `.deck_group_order() → usize` | Number of sheets |
| `.apply_deck_transform(policy, from, to) → Policy` | Apply deck transformation |
| `FiberBundle` | Base × Fiber (or twisted) |
| `FiberBundle::trivial(base_dim, fiber_dim)` | Product bundle |
| `FiberBundle::nontrivial(base_dim, fiber_dim)` | Non-trivial bundle |
| `.total_dim() → usize` | base + fiber |
| `.project(total_policy) → Policy` | Project to base |
| `.include(base_policy) → Policy` | Zero section inclusion |
| `Monodromy` | How fiber changes around a loop |
| `Monodromy::new(covering)` | Create |
| `.compute(loop_path) → Vec<usize>` | Permutation of sheets |
| `.is_trivial(perm) → bool` | Identity permutation? |

### Module: `obstruction` — Obstruction Theory

| Type / Method | Description |
|---|---|
| `ObstructionClass` | Element of Hⁿ⁺¹ measuring extension failure |
| `Obstruction` | Collects obstruction classes |
| `Obstruction::new(space_dim)` | Create analyzer |
| `.primary_obstruction(policies, target)` | Check if policies extend to target |
| `.secondary_obstruction(policies, constraints)` | Check pairwise constraints |
| `.all_obstructions_vanish() → bool` | Can extend? |
| `.total_obstruction() → f64` | Sum of absolute values |
| `.obstructions_at(dim) → Vec<&ObstructionClass>` | Filter by dimension |
| `.can_extend() → Result` | Extension Lemma: all obstructions vanish? |
| `.clear()` | Reset |
| `PostnikovTower` | Sequence of approximations to space |
| `PostnikovTower::build(homotopy_ranks)` | Build from homotopy group data |
| `.num_stages() → usize` | Tower height |
| `.stage(n) → Option<&PostnikovStage>` | n-th stage |
| `PostnikovStage` | Level n with k-invariants |

### Module: `whitehead` — Whitehead's Theorem

| Type / Method | Description |
|---|---|
| `CWComplex` | CW complex: cell counts per dimension |
| `CWComplex::point()` | 0-dimensional |
| `CWComplex::circle()` | S¹ |
| `CWComplex::sphere(n)` | Sⁿ |
| `CWComplex::disk(n)` | Dⁿ |
| `.is_valid() → bool` | Has ≥1 zero-cell? |
| `.euler_characteristic() → i64` | Σ(-1)ⁿ · cells_n |
| `.betti_numbers() → Vec<usize>` | Heuristic Betti numbers |
| `WhiteheadTheorem` | Checker for weak equivalence ⟹ homotopy equivalence |
| `WhiteheadTheorem::new(X, Y)` | Create for two CW complexes |
| `.check_weak_equivalence(π_X, π_Y) → WhiteheadResult` | Compare homotopy groups |
| `.theorem_applies() → bool` | Both spaces CW? |
| `WhiteheadResult` | is_weak_equivalence, is_homotopy_equivalence, failing_dimension, note |

### Module: `van_kampen` — Seifert–van Kampen Theorem

| Type / Method | Description |
|---|---|
| `Subspace` | Named region with sample points and π₁ rank |
| `Subspace::new(name, points, rank)` | Create |
| `SeifertVanKampen` | Decompose X = U ∪ V |
| `SeifertVanKampen::new(U, V, U∩V)` | Create decomposition |
| `.compute_fundamental_group() → VanKampenResult` | π₁(U∪V) = π₁(U) * π₁(V) / relations |
| `.is_simply_connected() → bool` | Result rank = 0? |
| `.amalgamated_product() → AmalgamatedProduct` | Explicit presentation |
| `VanKampenResult` | free_product_rank, num_relations, result_rank, description |
| `AmalgamatedProduct` | u_generators, v_generators, relations |
| `.total_generators() → usize` | U + V generators |
| `.num_relations() → usize` | From intersection |

### Module: `application` — Policy Transition Checker

| Type / Method | Description |
|---|---|
| `PolicyTransitionChecker` | High-level: is this transition safe? |
| `PolicyTransitionChecker::new()` | Default: 100 steps, ε=0.5, 20 MC paths, σ=0.1 |
| `PolicyTransitionChecker::with_params(steps, ε, paths, σ)` | Custom parameters |
| `.check(source, target) → TransitionResult` | Full safety analysis |
| `.check_spherical(source, target) → TransitionResult` | Using SLERP interpolation |
| `.find_safest(source, candidates) → (idx, TransitionResult)` | Best among multiple targets |
| `TransitionResult` | is_homotopic, path_length, max_step, continuous, reliability, has_obstructions, risk, summary |
| `RiskLevel` | Low, Medium, High |

## How It Works

The library models the topology of policy space through layers of increasing sophistication:

1. **`policy`** — Foundation. Policies are parameter vectors in ℝⁿ. Paths are sequences of waypoints with linear interpolation.

2. **`homotopy`** — Continuous deformations between two policies. Linear interpolation, spherical interpolation (SLERP for norm-preserving paths), and Bézier curves (for smooth curved paths). Continuity is checked by verifying consecutive waypoints are within ε.

3. **`stochastic`** — Adds Brownian noise to homotopy paths. Monte Carlo sampling estimates reliability: what fraction of noisy paths reach the target? Brownian bridges condition the noise on start/endpoints. Stability analysis sweeps noise levels.

4. **`fundamental_group`** — π₁ captures the "loop structure" of policy space. Non-trivial π₁ means there are holes — loops that can't be contracted. Winding numbers detect loops around the origin.

5. **`higher_homotopy`** — πₙ captures higher-dimensional holes. Maps from Sⁿ into policy space. The Hurewicz isomorphism connects the first non-trivial πₙ to homology Hₙ.

6. **`equivalence`** — Two policy spaces have the same homotopy type if there exist continuous maps f: X→Y, g: Y→X with f∘g ≃ id and g∘f ≃ id. Deformation retracts provide a practical way to check.

7. **`lifts`** — Covering spaces: when can a path in policy space be uniquely lifted to a "sheet"? Branch points block lifting. Monodromy tracks how the fiber permutes around loops.

8. **`obstruction`** — Cohomological barriers: obstruction classes in Hⁿ⁺¹ measure whether a map defined on the n-skeleton extends to the (n+1)-skeleton. Postnikov towers approximate the space stage by stage.

9. **`whitehead`** — Whitehead's theorem: for CW complexes, a weak homotopy equivalence (isomorphism on all πₙ) implies a genuine homotopy equivalence. Euler characteristic and Betti numbers provide invariants.

10. **`van_kampen`** — Decompose policy space into overlapping regions U and V. The fundamental group of the union is an amalgamated free product: π₁(U∪V) = π₁(U) * π₁(V) / ⟨i₁(g)·i₂(g)⁻¹⟩.

11. **`application`** — `PolicyTransitionChecker` ties everything together: create a homotopy, check continuity, run Monte Carlo reliability, check obstructions, and return a risk level.

## The Math

### Homotopy
A homotopy between policies f and g is a continuous map H: [0,1] → PolicySpace with H(0) = f, H(1) = g. Linear: H(t) = (1−t)f + tg. SLERP: H(t) = sin((1−t)Ω)/sin(Ω) · f̂ + sin(tΩ)/sin(Ω) · ĝ where Ω = arccos(f̂·ĝ).

### Fundamental Group
π₁(X, x₀) consists of homotopy classes of loops based at x₀. The group operation is path concatenation. For S¹, π₁ = ℤ (winding numbers). For contractible spaces, π₁ = {e} (trivial).

### Stochastic Homotopy
H_stochastic(t, ω) = H(t) + σ · W(t, ω) where W is a Wiener process with increments ΔW ~ N(0, Δt). The Brownian bridge conditions on W(0) = 0, W(1) = 0: B(t) = W(t) − t·W(1).

### Obstruction Theory
Given f: X^(n) → Y on the n-skeleton, the obstruction to extending to X^(n+1) is a class [o_f] ∈ H^(n+1)(X; π_n(Y)). If this class vanishes, extension is possible. Postnikov towers build X from Eilenberg-MacLane spaces K(π_n, n) using k-invariants.

### Whitehead's Theorem
If f: X → Y is a weak homotopy equivalence (induces isomorphisms f_*: π_n(X) → π_n(Y) for all n) and X, Y are CW complexes, then f is a homotopy equivalence.

### Seifert–van Kampen
For X = U ∪ V with U ∩ V path-connected: π₁(X) ≅ π₁(U) * π₁(V) / N, where N is the normal closure of {i₁(g)·i₂(g)⁻¹ : g ∈ π₁(U∩V)} and i₁, i₂ are inclusion maps.

## License

MIT
