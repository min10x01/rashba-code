# Rashba Parameter Analysis Code

This repository provides Python scripts to calculate the **Rashba spin–splitting parameters** from *ab initio* band structures obtained using **VASP**, **Quantum ESPRESSO**, or similar DFT codes.  
The scripts extract the **Rashba coefficient (α<sub>R</sub>)**, **momentum offset (k₀)**, and **Rashba energy (E₀)** from spin–orbit–coupled band data.

---

## 🧠 Background

The **Rashba effect** arises from **spin–orbit coupling (SOC)** in systems lacking inversion symmetry, producing spin-split energy bands near high-symmetry points in the Brillouin zone.  

The dispersion relation near the Rashba point can be expressed as:

$$
E(k) = E_0 + \frac{\hbar^2 k^2}{2m^*} \pm \alpha_R k
$$

where:

- $ \alpha_R $ — Rashba parameter (eV·Å)  
- $ k_0 $ — momentum offset (Å⁻¹)  
- $ E_0 $ — Rashba energy (eV)  
- $ m^* $ — effective mass of the charge carrier  

The Rashba coefficient is evaluated as:

$$
\alpha_R = \frac{2E_R}{k_R}
$$

---

## ⚙️ Features

- Automatic parsing of band-structure outputs from **VASP** or **Quantum ESPRESSO**  
- Extraction of spin-split branches and parabolic fitting near the Rashba point  
- Computation of:
  - Rashba coefficient ($\alpha_R$)  
  - Momentum offset ($k_0$)  
  - Rashba energy ($E_0$)
- Optional visualization of Rashba splitting
- Modular Python structure for easy customization

--

