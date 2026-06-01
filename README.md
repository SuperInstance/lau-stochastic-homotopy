# lau-stochastic-homotopy

**Stochastic processes meet homotopy theory — continuous deformation of agent policies under uncertainty.**

A homotopy between two agent policies is a continuous path through policy space. If two policies are homotopic, an agent can deform one into the other without crossing a singularity. This crate adds *noise* — Brownian motion, Monte Carlo sampling — to the classical theory, asking: **can an agent still safely transition between behaviors when the world is stochastic?**

122 tests · MIT license · pure Rust · zero unsafe

---

## What This Does

| Module | Concept | What you get |
|---|---|---|
| `policy` | Policies as points in ℝⁿ | Parameter vectors, distances, interpolation, noise injection |
| `homotopy` | H: [0,1] × PolicySpace → PolicySpace | Linear / spherical / Bézier deformation paths |
| `fundamental_group` | π₁ of policy space | Loops, winding numbers, free group operations |
| `higher_homotopy` | πₙ for n ≥ 2 | Sphere maps, Hurewicz theorem, degree computation |
| `equivalence` | Homotopy-equivalent spaces | Deformation retracts, CW complexes, continuous maps |
| `stochastic` | H(t,ω) = H(t) + σ·W(t,ω) | Brownian bridges, Monte Carlo reliability, stability analysis |
| `lifts` | Covering spaces & fiber bundles | Path lifting, deck transformations, monodromy |
| `obstruction` | Obstruction theory | Primary/secondary cohomological obstructions, Postnikov towers |
| `whitehead` | Whitehead's theorem | Weak equivalence ⟹ homotopy equivalence for CW complexes |
| `van_kampen` | Seifert–van Kampen | Glue subspaces, compute π₁ of the union |
| `application` | Policy transition checker | End-to-end: is a policy switch safe? Risk level, reliability |

---

## Key Idea

> Two agent policies are *homotopic* if there exists a continuous deformation from one to the other. In a stochastic world, we ask not "is there a path?" but "what fraction of noisy paths actually arrive?"

The `PolicyTransitionChecker` runs a deterministic homotopy, verifies continuity, checks for cohomological obstructions, then blasts it with Brownian noise via Monte Carlo to estimate the probability the agent still reaches the target. The result is a `RiskLevel` (Low / Medium / High) you can use to gate live policy switches.

---

## Install

```toml
[dependencies]
lau-stochastic-homotopy = { git = "https://github.com/SuperInstance/lau-stochastic-homotopy" }
```

Requires Rust 2021 edition. Dependencies: `nalgebra`, `serde`, `rand`, `rand_distr`, `thiserror`.

---

## Quick Start

```rust
use lau_stochastic_homotopy::{
    Policy, PolicyTransitionChecker,
};

let source = Policy::new(vec![0.0, 0.0, 0.0]);
let target = Policy::new(vec![1.0, 0.5, -0.3]);

let checker = PolicyTransitionChecker::with_params(
    100,   // discretization steps
    0.5,   // continuity tolerance ε
    20,    // Monte Carlo paths
    0.1,   // noise σ
);

let result = checker.check(&source, &target);
println!("homotopic? {}  risk: {:?}  reliability: {:.3}",
    result.is_homotopic, result.risk,
    result.reliability.unwrap_or(0.0));
```

Pick the safest among several candidates:

```rust
let candidates = vec![
    Policy::new(vec![0.1, 0.1, 0.0]),
    Policy::new(vec![5.0, 5.0, 5.0]),
    Policy::new(vec![0.5, 0.3, -0.1]),
];
let (idx, result) = checker.find_safest(&source, &candidates);
```

---

## API Reference

### Core Types

**`Policy`** — a point in parameter space.
- `Policy::new(params: Vec<f64>)` — construct from a vector.
- `policy.distance_to(&other)` — L2 distance.
- `policy.lerp(&other, t)` — linear interpolation at `t ∈ [0,1]`.
- `policy.add_noise(sigma, &mut rng)` — Gaussian perturbation.
- `policy.normalize()` — unit-vector normalization.

**`PolicyPath`** — a piecewise-linear path through policy space.
- `PolicyPath::new(waypoints)` — from explicit waypoints.
- `path.evaluate(t)` — interpolate at parameter `t`.
- `path.is_loop(tol)` — does it return to the start?
- `path.length()` — total arc length.

**`Homotopy`** — a continuous deformation H(t) between two policies.
- `Homotopy::new(source, target, steps)` — linear interpolation.
- `Homotopy::spherical(...)` — SLERP (norm-preserving).
- `Homotopy::bezier(...)` — quadratic Bézier with curvature.
- `h.evaluate(t)` — policy at parameter `t`.
- `h.path()` — full discretized path.
- `h.check_continuity(epsilon)` — verify no jumps exceed `epsilon`.
- `h.compose(&other)` / `h.reverse()` — algebraic operations.
- `RelativeHomotopy` — deformation that holds selected dimensions fixed.

### Topology

**`FundamentalGroup`** — π₁(X, x₀), the group of homotopy classes of loops.
- `FundamentalGroup::trivial(base_point)` — simply-connected space.
- `fg.add_generator(loop_path)` — register a non-trivial loop.
- `fg.multiply(&a, &b)` / `fg.invert(&a)` — group operations.
- `fg.winding_number(&path)` — 2D winding around the origin.
- `fg.is_contractible(&path, max_iter)` — can the loop be shrunk to a point?
- `fg.abelianization()` — rank of H₁.

**`HigherHomotopyGroup`** — πₙ for n ≥ 2.
- `HigherHomotopyGroup::sphere(n)` — πₙ(Sⁿ) ≅ ℤ.
- `SphereMap` — discretized map from Sⁿ into policy space.
- `HurewiczMap` — connects πₙ to Hₙ.

**`HomotopyEquivalence`** — when do two spaces have the same shape?
- `HomotopyEquivalence::deformation_retract(&space, &subspace, steps)` — is the subspace a deformation retract?
- `ContinuousMap` — linear maps between policy spaces, with composition.

**`WhiteheadTheorem`** — for CW complexes, weak equivalence implies homotopy equivalence.
- `WhiteheadTheorem::new(complex_x, complex_y)` — set up the check.
- `w.check_weak_equivalence(&pi_x, &pi_y)` — compare homotopy groups.
- `CWComplex` — cell decomposition with Euler characteristic and Betti numbers.

**`SeifertVanKampen`** — compute π₁ of a union from π₁ of pieces.
- `SeifertVanKampen::new(u, v, intersection)` — the three subspaces.
- `vk.compute_fundamental_group()` — returns rank and description.
- `vk.amalgamated_product()` — explicit generators and relations.

### Covering Spaces

**`CoveringSpace`** — a covering map p: E → B.
- `CoveringSpace::new(base_dim, num_sheets)` — `num_sheets`-sheeted cover.
- `cs.add_branch_point(point)` — where lifting fails.
- `cs.fiber(&base_point)` — preimage (all sheets).
- `cs.lift_path(&path, sheet)` — lift a path to a specific sheet.
- `cs.deck_group_order()` / `cs.apply_deck_transform(...)`.

**`FiberBundle`** — bundles over policy space.
- `FiberBundle::trivial(base_dim, fiber_dim)` / `nontrivial(...)`.
- `bundle.project(&total)` / `bundle.include(&base)`.
- `Monodromy` — permutation of sheets around a loop.

### Obstruction Theory

**`Obstruction`** — cohomological barriers to extending local deformations globally.
- `obs.primary_obstruction(&local_policies, &target)` — distance-based obstruction.
- `obs.secondary_obstruction(&policies, &constraints)` — constraint-violation obstruction.
- `obs.can_extend()` — do all obstructions vanish?
- `PostnikovTower` — stage-by-stage approximation of the space.

### Stochastic

**`StochasticHomotopy`** — homotopy + Brownian motion.
- `StochasticHomotopy::new(base_homotopy, sigma, num_paths, seed)`.
- `sh.sample_path(&mut rng)` — one noisy realization.
- `sh.all_paths()` — all Monte Carlo paths.
- `sh.continuity_probability(epsilon)` — fraction of continuous paths.
- `sh.reliability(tol)` — fraction that reach the target.
- `sh.mean_path_length()` — average arc length.

**`BrownianBridge`** — Wiener process pinned at both ends.
- `BrownianBridge::new(start, end, sigma)`.
- `bb.sample(steps, &mut rng)` — a bridge path.

**`StabilityAnalysis`** — sweep noise levels.
- `StabilityAnalysis::analyze(base_homotopy, &[sigma_values], num_paths, tol)`.

### Application

**`PolicyTransitionChecker`** — the main entry point.
- `PolicyTransitionChecker::new()` — sensible defaults.
- `PolicyTransitionChecker::with_params(steps, epsilon, paths, sigma)`.
- `checker.check(&source, &target)` → `TransitionResult`.
- `checker.check_spherical(&source, &target)`.
- `checker.find_safest(&source, &candidates)`.

**`TransitionResult`**:
- `is_homotopic`, `continuous`, `has_obstructions` — booleans.
- `path_length`, `max_step` — geometry.
- `reliability` — Monte Carlo probability.
- `risk` — `RiskLevel::Low`, `Medium`, or `High`.
- `summary` — human-readable string.

---

## How It Works

1. **Discretize**: The homotopy is sampled at `steps+1` evenly-spaced parameter values in [0, 1]. At each sample, a `Policy` is computed via the chosen interpolation (linear, spherical, or Bézier).

2. **Check continuity**: Consecutive policies must be closer than `epsilon`. If any gap exceeds the threshold, the homotopy is discontinuous at that resolution.

3. **Check obstructions**: A cohomological obstruction test verifies that local policy patches can be consistently extended to a global deformation. If not, the transition is blocked regardless of continuity.

4. **Add noise**: Each Monte Carlo path adds scaled Brownian increments to the deterministic homotopy. The result is a cloud of noisy trajectories through policy space.

5. **Measure reliability**: The fraction of Monte Carlo paths whose endpoints land within `tol` of the target policy.

6. **Classify risk**: High reliability + continuous + no obstructions → Low risk. Anything less escalates through Medium to High.

---

## The Math

### Homotopy

Two continuous maps f, g: X → Y are **homotopic** (f ≃ g) if there exists a continuous H: [0,1] × X → Y with H(0, ·) = f and H(1, ·) = g. In this crate, X is a single point and Y is policy parameter space ℝⁿ, so a homotopy is simply a path H: [0,1] → ℝⁿ.

### Fundamental Group

The **fundamental group** π₁(X, x₀) consists of homotopy classes of loops based at x₀, with group operation given by concatenation. A contractible space has trivial π₁. The winding number of a 2D loop around the origin detects non-trivial elements.

### Seifert–van Kampen

For X = U ∪ V with U ∩ V path-connected, π₁(X) is the **amalgamated free product** π₁(U) ∗ π₁(V) with relations identifying the images of π₁(U ∩ V) in both.

### Whitehead's Theorem

If f: X → Y is a map between CW complexes that induces isomorphisms on all homotopy groups πₙ, then f is a **homotopy equivalence**. This crate implements the check by comparing generator counts at each dimension.

### Obstruction Theory

Given a map defined on the n-skeleton of a CW complex, it extends to the (n+1)-skeleton iff a certain cohomology class in Hⁿ⁺¹ vanishes. This crate tracks these obstructions as distance-based and constraint-based numerical values.

### Stochastic Homotopy

The stochastic homotopy H_s(t, ω) = H(t) + σ · W(t, ω) adds a Wiener process (Brownian motion) to the deterministic path. Properties like continuity and endpoint accuracy become *probabilistic*: we estimate P(H_s is continuous) and P(‖H_s(1) − target‖ < ε) via Monte Carlo.

A **Brownian bridge** B(t) is a Wiener process conditioned on B(0) = a and B(1) = b, implemented as B(t) = (1−t)a + tb + [W(t) − tW(1)].

### Hurewicz Theorem

For an (n−1)-connected space, the first non-trivial homotopy group πₙ is isomorphic to the homology group Hₙ. The `HurewiczMap` struct checks the connectivity condition and returns the isomorphism.

---

## License

MIT
