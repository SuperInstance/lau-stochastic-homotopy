# lau-stochastic-homotopy

> Stochastic processes meet homotopy theory — continuous deformation of agent policies.

A **homotopy** between two policies is a continuous path in policy space. If two policies are homotopic, they can be deformed into each other without crossing a singularity. This determines whether an agent can transition between behaviors safely.

## Core Concepts

| Module | Description |
|--------|-------------|
| `homotopy` | Continuous deformation between agent policies (paths in function space) |
| `fundamental_group` | π₁ of agent policy space — loops of policy changes |
| `higher_homotopy` | πₙ for higher-dimensional policy deformations |
| `equivalence` | Homotopy equivalence — when are two agent spaces "the same shape"? |
| `stochastic` | Homotopy + noise — can policies be deformed under uncertainty? |
| `lifts` | When can a local policy change be extended globally? |
| `obstruction` | What prevents extending a local deformation to a global one? |
| `whitehead` | Weak equivalence implies homotopy equivalence (for CW complexes) |
| `van_kampen` | Compute fundamental group by gluing agent subspaces |
| `application` | Determine if an agent can safely transition between two policies |

## Quick Start

```rust
use lau_stochastic_homotopy::{Policy, PolicyTransitionChecker};

let source = Policy::new(vec![0.0, 0.0, 0.0]);
let target = Policy::new(vec![1.0, 0.5, 0.2]);

let checker = PolicyTransitionChecker::new();
let result = checker.check(&source, &target);

if result.is_homotopic {
    println!("Safe to transition! Risk: {:?}", result.risk);
} else {
    println!("Transition blocked: {}", result.summary);
}
```

## Features

- **Linear, spherical (SLERP), and Bézier interpolation** between policies
- **Fundamental group computation** with generators, loops, and winding numbers
- **Higher homotopy groups** with Hurewicz theorem support
- **Stochastic homotopy** with Brownian motion and Monte Carlo path sampling
- **Covering spaces** with branch point detection and path lifting
- **Obstruction theory** with primary/secondary obstruction classes and Postnikov towers
- **Whitehead theorem** for CW complexes
- **Seifert–van Kampen** for computing fundamental groups of glued spaces
- **Policy transition safety checker** with risk assessment

## Dependencies

- `nalgebra` — linear algebra
- `serde` / `serde_json` — serialization
- `rand` / `rand_distr` — stochastic sampling

## License

MIT
