# Example 4: Molecular Dynamics Simulations
This example demonstrates how to perform molecular dynamics (MD) simulations using the ByteFF-Pol force field with OpenMM.

## Overview
The MD simulations example shows how to:
* Run NPT simulations for density calculations
* Run liquid and gas phase simulations to evaluate evaporation enthalpy (Hvap).
* Conduct a simulation to compute transport properties such as viscosity, conductivity and so on.

## How to Run
0. Set PYTHONPATH
```bash
export PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH}
```
1. Run MD simulations
If you want to run MD simulations for density calculations, run:
```bash
python run_md.py --config density_config.json
```
The config files for other simulations, like evaporation enthalpy (Hvap) and transport properties, are also provided. To run these simulations, simply replace `density_config.json` with the corresponding config file.

## Configuration File Details (*_config.json)

This configuration is used for running transport property simulations (viscosity and conductivity) on electrolyte systems:

* **protocol**: "Transport" - Specifies the simulation protocol type, including `Transport`, `Density` and `HVap`.
* **temperature**: 298 - Simulation temperature in Kelvin
* **natoms**: 10000 - Total number of atoms in the box
* **components**: Molecular composition with **molecule ratio**:
  - **DMC**: 249 
  - **EC**: 170 
  - ...
* **smiles**: SMILES strings for each component.
  - **DMC**: "COC(=O)OC"
  - ...

Optional box controls (for all protocols):
- `box_length` (number, nm): Overrides the initial cubic box edge length used to pack molecules (GROMACS `editconf -box`).
- `box_scale` (number): Multiplier applied to the internally predicted box length; ignored if `box_length` is provided.

Optional composition and timing controls:
- `components_as_counts` (bool): If true, the values in `components` are treated as exact molecule counts. `natoms` is recomputed from those counts to stay consistent.
- `components_counts` (object): Alternatively provide a separate map of exact molecule counts here; if present it takes precedence over `components`.
- `npt_steps`, `nvt_steps`, `nonequ_steps` (integers): Override default MD lengths (steps) for the respective phases.
- `npt_time_ns`/`ps`, `nvt_time_ns`/`ps`, `nonequ_time_ns`/`ps` (numbers): Specify total time instead of steps. Conversion uses 2 fs for NPT/NVT, and 1 fs for the non-equilibrium viscosity run.

Restart controls:
- `resume` (bool): If true and a phase checkpoint exists, resumes NPT/NVT/nonequ runs from the latest checkpoint (continues CSV/DCD/viscosity logs when supported).
- `checkpoint_interval` (int, steps): Frequency to write checkpoints during a run (default 5000).

Files produced per phase for restart:
- NPT: `npt.chk`, `npt_state.csv`, `npt.dcd`
- NVT: `nvt.chk`, `nvt_state.csv`, `nvt.dcd`
- Nonequilibrium: `nonequ.chk`, `viscosity.csv`

Notes:
- Packing still auto-expands the box by 5% if insertion fails (to ensure the requested composition/atom count fits).
- The periodic unit cell used for MD comes from the generated `.gro` box.
- Ionic conductivity and viscosity controls (Transport protocol):
  - `compute_viscosity` (bool, default true): Runs the non-equilibrium segment and computes viscosity from `viscosity.csv`.
  - `compute_conductivity` (bool, default true): Computes ionic conductivity from NVT frames using Onsager theory; results saved to `results.json`.
  - `viscosity_cP` (number, optional): If provided and `compute_viscosity` is false, uses this value for Yeh–Hummer finite-size correction. If not provided, the conductivity calculation proceeds but skips YH correction (flagged in results as `yh_correction_applied: false`).

Notes on conductivity:
- Conductivity is computed from the equilibrium NVT trajectory (`nvt.dcd`). The non-equilibrium run is only used to compute viscosity for Yeh–Hummer correction of diffusivities; Onsager conductivity itself does not require the non-equilibrium frames.
