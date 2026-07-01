# Rashba Parameter Analysis Code

This repository provides a Python script (`rashba.py`) to calculate the **Rashba spin-splitting parameter** from *ab initio* band structures obtained using **VASP**.
The script parses a VASP `OUTCAR`/`KPOINTS` pair from a line-mode band-structure calculation and extracts the **Rashba coefficient** ($\alpha_R$) at the spin-splitting points near the valence-band maximum (VBM) and conduction-band minimum (CBM).

---

## Background

The **Rashba effect** arises from **spin-orbit coupling (SOC)** in systems lacking inversion symmetry, producing spin-split energy bands near high-symmetry points in the Brillouin zone.

The Rashba coefficient is estimated as the secant slope between a band extremum and the k-point where the two spin-split branches merge:

$$
\alpha_R = \frac{2 \Delta E}{\Delta k}
$$

where $\Delta E$ is the energy difference between the two points and $\Delta k$ is their separation in reciprocal space, computed from the reciprocal lattice vectors in `OUTCAR`.

---

## Features

- Parsing of VASP `OUTCAR` band energies/occupations and `KPOINTS` (line-mode) high-symmetry points
- Automatic detection of whether the system is metallic or has a band gap
- Identification of the VBM, CBM, and band gap
- Detection of secondary extrema near the VBM/CBM (for Rashba-split bands)
- Computation of the Rashba coefficient ($\alpha_R$) at detected splitting points on the valence and conduction bands
- Band structure plot with VBM/CBM/splitting points highlighted (saved to `band_structure.png` if no display is available)

## Requirements

See `requirements.txt` (numpy, matplotlib).

## Usage

Run the script inside a VASP band-structure calculation directory containing `OUTCAR` and `KPOINTS` (line-mode):

```bash
python rashba.py
```