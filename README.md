# Homopolymer knotted and unknotted chromosome globules

Reproducible molecular-dynamics workflows for comparing **knotted** and
**unknotted** compact circular polymers. The models are implemented in
[`src/hp.py`](src/hp.py) with
[OpenMiChroM](https://github.com/junioreif/OpenMiChroM) and
[OpenMM](https://openmm.org), and are used as baseline polymer references for
the topology of the *Salmonella* chromosome.

The code is intentionally focused on the homopolymer (HP) backbone and the two
topological ensembles. It does not attempt to document every historical or
special-purpose workflow in `hp.py` here.

## Scientific context

The HP model represents a chromosome as a coarse-grained circular bead-spring
polymer. Its generic polymer interactions provide the common physical basis for
both ensembles; the difference is the topological history allowed during
equilibration:

| Model | Topological condition | Purpose |
| --- | --- | --- |
| `unknotted` | Chain crossings are prevented, preserving the initially unknotted ring topology. | Topology-preserving reference globule. |
| `knotted` | Strand passage is allowed during equilibration, allowing the polymer to sample a topologically equilibrated, highly knotted globule ensemble. | Topologically equilibrated reference globule. |

Both ensembles are compact circular polymers, so their comparison isolates the
effect of topology rather than a gross change in polymer compaction. In the
associated *Salmonella* study, coarse-grained 20-vertex polygons sampled from
these ensembles provide reference distributions for ensemble-level knotting
analysis.

## Model implementation

`src/hp.py` defines `HPConfig` and `HP`. The shared HP force scaffold includes
FENE bonds, angular stiffness, and a soft-core excluded-volume interaction. The
two model selections retain the parameters encoded in the implementation:

| Setting | `knotted` | `unknotted` |
| --- | ---: | ---: |
| Equilibration temperature | 1.5 | 0.7 |
| Soft-core repulsion cutoff | 3.0 | 500.0 |

All quantities use the reduced/OpenMiChroM units of the implementation; this
repository does not convert them. The models operate on a circular chain
(`isRing=True`).

## How to run

Use Python 3.9 or newer with the dependencies declared in
[`src/OpenMiChroM/setup.py`](src/OpenMiChroM/setup.py): NumPy, SciPy,
scikit-learn, h5py, pandas, and OpenMM. Install the bundled OpenMiChroM source
into the environment:

```bash
python -m pip install -e src/OpenMiChroM
```

Then run the selected `hp.py` workflow with a new output directory and the
supplied chromosome-sequence file:

```bash
python src/hp.py all_at_once output_folder inputs/seq_10k.txt --model unknotted
```

## Project folder

This `hp/` folder is a self-contained simulation project. Its compact layout
keeps the model implementation, bundled dependency source, and example input
together:

| Path | Description |
| --- | --- |
| `src/` | HP model implementation (`hp.py`) and bundled OpenMiChroM source. |
| `inputs/` | Input files, including the 10,000-bead circular-polymer sequence. |
| `README.md` | Project overview, model scope, usage, and citations. |
| `LICENSE` | MIT license for this source code. |

Use a new, user-chosen output directory for every simulation run. Generated
structures, trajectories, and derived results are not source inputs and should
be kept separate from this project folder.

## Citation

Please cite the scientific work that corresponds to the model or analysis you
use:

1. Oliveira, R. J., Oliveira Junior, A. B., Contessoto, V. G., and Onuchic,
   J. N. *The synergy between compartmentalization and motorization in
   chromatin architecture.* **The Journal of Chemical Physics** 162, 114116
   (2025). [https://doi.org/10.1063/5.0239634](https://doi.org/10.1063/5.0239634)
2. Gkountaroulis, D. *et al.* *The Salmonella genome is strongly
   underknotted.* Manuscript in preparation (2026).

## Contributors

**Ronaldo J. Oliveira** and **Angel Mendieta**.

## License

The source code is released under the [MIT License](LICENSE).
