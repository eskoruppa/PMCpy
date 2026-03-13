# PMCpy

**PMCpy** is a Python package for Polymer Monte Carlo (PMC) simulations of double-stranded DNA (dsDNA). It implements sequence-dependent, coarse-grained rigid base pair-step models that capture structural and mechanical properties of DNA at the base pair level — enabling efficient conformational sampling of DNA molecules spanning hundreds to thousands of base pairs.

---

## Features

- **Sequence-dependent energy model** — base pair-step stiffness and ground-state parameters derived from atomistic MD simulations (Lankas / RBP parametrisation), with support for systematic coarse-graining to lower-resolution representations.
- **Flexible boundary conditions** — open linear chains and closed (circular) DNA.
- **Rich set of Monte Carlo moves** — Pivot, Double Pivot, Crankshaft, Cluster Translation, Single Triad, and Midstep Move, all with Metropolis acceptance.
- **Excluded volume** — bead-based excluded-volume interactions with optional self-crossing detection.
- **External constraints and forces** — stretching forces (tweezer geometry), repulsion planes, and user-defined fixed triads.
- **Equilibration protocol** — automated convergence detection for burn-in.
- **Built-in observables** — tangent-tangent correlation (persistence length), writhe and linking number via the PyLk submodule.
- **Trajectory I/O** — XYZ-format trajectory writing and reading.
- **Optional Numba acceleration** — just-in-time compilation of performance-critical kernels.

---

## Installation

Clone the repository together with all required git submodules:

```console
git clone --recurse-submodules -j8 git@github.com:eskoruppa/PMCpy.git
```

Then install the package in editable mode:

```console
cd PMCpy
pip install -e .
```

### Optional dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `numba` | `numba` | JIT-accelerated MC moves and evaluators |
| `plot` | `matplotlib` | Visualisation utilities |
| `all` | all of the above | Full feature set |

Install extras with, e.g.:

```console
pip install -e ".[all]"
```

> **Note:** The submodules (`SO3`, `RBPStiff`, `pyConDec`, `PyLk`) are included as git submodules and are not separately published on PyPI. The recursive clone above is the recommended installation path.

---

## Quick start

```python
import numpy as np
from pmcpy import Run
from pmcpy.GenConfs.straight import gen_straight

nbp      = 500
sequence = "".join(np.random.choice(list("ATCG"), nbp))
triads, positions = gen_straight(nbp)

sim = Run(
    triads=triads,
    positions=positions,
    sequence=sequence,
    closed=False,
    endpoints_fixed=True,
    temp=300,
    exvol_rad=2.0,
    parameter_set="md",
)

sim.run(num_steps=100_000, dump_every=1_000, outfile="traj.xyz")
```

---

## Citation

If you use PMCpy in your research, please cite the following papers:

> Enrico Skoruppa, Helmut Schiessel,  
> **Systematic coarse-graining of sequence-dependent structure and elasticity of double-stranded DNA**,  
> *Physical Review Research* **7**, 013044 (2025).  
> DOI: [10.1103/PhysRevResearch.7.013044](https://doi.org/10.1103/PhysRevResearch.7.013044)

> Willem Vanderlinden, Enrico Skoruppa, Pauline J. Kolbeck, Enrico Carlon, Jan Lipfert,  
> **DNA fluctuations reveal the size and dynamics of topological domains**,  
> *PNAS Nexus* **1**(5), pgac268 (2022).  
> DOI: [10.1093/pnasnexus/pgac268](https://doi.org/10.1093/pnasnexus/pgac268)

---

## License

PMCpy is released under the [GNU General Public License v2.0](LICENSE).
