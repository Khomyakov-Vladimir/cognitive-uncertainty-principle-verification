**Reproducibility Package (Versions 1.0 and 2.0 and 3.0)**  
# The Epistemic Phase Boundary in Cognitive Uncertainty Principle (3.0)  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17370034.svg)](https://doi.org/10.5281/zenodo.17370034)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  

---

## Overview  

This repository provides a complete reproducibility package for both the **empirical** and **theoretical** verification of the *Cognitive Uncertainty Principle (CUP)*.  

- **Version 1.0** — Empirical bootstrap verification of the inequality Δβ × ΔKL ≥ C.  
- **Version 2.0** — Extended theoretical formulation using a jump–diffusion process and perceptual trade-off analysis within the information-geometric framework.  
- **Version 3.0** — Resolution of the apparent discrepancy between theoretical and empirical constants through the introduction of the *epistemic phase boundary*—a critical transition between local (Fisher-based) and global (KL-based) cognitive regimes. Empirical validation of the scale-adaptive uncertainty constant C(ξ,ε) and formal identification of the critical jump intensity λ_c ≈ 0.01 using systematic simulations across 252 parameter configurations.  

All code, data, and figures required to regenerate the published results are included.  

---

## DOI and Citation  

| Version | DOI | Description |  
|----------|-----|-------------|  
| **v1.0** | [10.5281/zenodo.17370035](https://doi.org/10.5281/zenodo.17370035) | Empirical bootstrap verification |  
| **v2.0** | [10.5281/zenodo.17465968](https://doi.org/10.5281/zenodo.17465968) | Dual empirical–theoretical verification |  
| **v3.0** | [10.5281/zenodo.17523490](https://doi.org/10.5281/zenodo.17523490) | The Epistemic Phase Boundary in Cognitive Uncertainty Principle: dual-regime duality, scaling law C(ξ,ε), and critical transition at λ_c |  

**Citation:**  

> Khomyakov, V. (2025). *The Epistemic Phase Boundary in Cognitive Uncertainty Principle (3.0)*. Zenodo.  
> DOI: [10.5281/zenodo.17523490](https://doi.org/10.5281/zenodo.17523490)  

---

## Repository Structure  

```
cognitive-uncertainty-principle-verification/  
│
├── README.md  
│
├── scripts/  
│   ├── qcot_theoretical_framework_enhanced.py   # Empirical verification (v1.0)  
│   ├── jump-diffusion.py                        # Ontological model (v2.0)  
│   ├── kl_tradeoff_analysis.py                  # Perceptual trade-off (v2.0)  
│   ├── cup_sim_config.json                      #  (v3.0) Standard configuration template  
│   ├── cup_sim_config.yaml                      #  (v3.0) Standard configuration template  
│   └── cup_scaling_pipeline.py                  #  (v3.0) Integrated Cognitive Uncertainty Principle Scaling Analysis with Phase Transitions  
│
├── figures/  
│   ├── bootstrap_distribution.pdf               # v1.0 — Bootstrap distribution of C  
│   ├── uncertainty_verification.pdf             # v1.0 — Verification plot with violations  
│   ├── violation_analysis.pdf                   # v1.0 — Violation morphology  
│   ├── KL_tradeoff_results.pdf                  # v2.0 — Jump–diffusion results  
│   ├── canonical_relation.pdf                   # v2.0 — Canonical Δβ·ΔKL relation  
│   ├── perceptual_relation.pdf                  # v2.0 — Perceptual Δε·ΔKL relation  
│   ├── cup_scaling_heatmap_kl_median.pdf        # v3.0 — KL divergence heatmap  
│   └── cup_phase_transition.pdf                 # v3.0 — Phase transition diagram with λ_c  
│
└── data/  
    ├── violations.json                          # v1.0 — Violation metadata  
    ├── robustness_report.txt                    # v1.0 — Statistical report  
    ├── kl_tradeoff_extended.npz                 # v2.0 — Extended trade-off dataset  
    ├── phase_transition_sensitivity.csv         # v3.0 — Sensitivity analysis results  
    └── cup_sim_output/  
      ├── fit_results.json                       # v3.0 — Fitted scaling parameters with confidence intervals  
      ├── cup_grid_results.csv                   # v3.0 — Complete parameter sweep results  
      └── cell_XXXX.pkl                          # v3.0 — Intermediate cell results (optional)  
```

---

## Key Results  

| Constant | Source | Value | Interpretation |  
|-----------|---------|--------|----------------|  
| **C β** | Empirical bootstrap | 3.94 × 10⁻⁴ | Canonical uncertainty bound (Δβ × ΔKL ≥ C β) |  
| **C ε** | Perceptual analysis | 1.71 × 10⁻² | Perceptual uncertainty bound (Δε × ΔKL ≥ C ε) |  
| **C** | Jump–diffusion theory | 1.17 × 10⁻⁴ | Ontological constant from Fisher-information model |  

---

## Reproducibility Instructions  

Create the analysis environment and install dependencies:  

```bash  
conda env create -f environment.yml  
conda activate QCOT  
pip install -r requirements.txt  
```

Run the empirical verification (version 1.0):  

```bash  
python scripts/qcot_theoretical_framework_enhanced.py  
```

Run the theoretical extensions (version 2.0):  

```bash  
python scripts/jump-diffusion.py  
python scripts/kl_tradeoff_analysis.py  
```

Complete pipeline execution (version 3.0):  

```bash  
python cup_scaling_pipeline.py --config cup_sim_config.yaml  
```

Phase analysis only (existing data) (version 3.0):  

```bash  
python cup_scaling_pipeline.py --data-path data/cup_grid_results.csv --skip-data-gen  
```

Skip phase analysis (version 3.0):  

```bash
python cup_scaling_pipeline.py --config config.yaml --skip-phase-analysis  
```

All generated figures and numerical outputs will appear in `figures/` and `data/`.  

---
NOTE:  
"RuntimeWarning" messages are an intrinsic and expected part of the model.  
They reflect the inherent instability of the observer–world interaction 
under finite perceptual resolution. This instability drives the 
self-organization processes responsible for the emergence of 
1/f spectral structure and Benford-like digit distributions.  
  
Suppressing or correcting these warnings would artificially remove 
the feedback mechanism that constitutes the model’s physical content. 
Therefore, the simulation must be executed in its original form, 
preserving the adaptive selection of the optimal perceptual threshold 
(epsilon_best) as postulated in Subjective Physics.  

---

## Dependencies  

The scripts require the following Python packages:  

| Package     | Version |  
|--------------|----------|  
| Python       | 3.8+     |  
| NumPy        | 1.20+    |  
| SciPy        | 1.6+     |  
| Matplotlib   | 3.3+     |  

Exact versions used for verification and reproducibility are pinned in `requirements.txt`.  

Install the pinned versions with:  

```bash  
pip install -r requirements.txt  
```

The `requirements.txt` file included in this archive pins the package versions used to reproduce the results reported in this repository. If you prefer a conda-managed environment, use the provided `environment.yml` file.  

### requirements.txt (pinned versions used for verification and reproducibility)  

```
numpy>=1.20  
scipy>=1.6  
matplotlib>=3.3  
```

### Python Environment (conda)  

File: `environment.yml`  

```yaml  
name: QCOT  
channels:  
  - conda-forge  
  - defaults  
dependencies:  
  - python>=3.8  
  - numpy>=1.20  
  - scipy>=1.6  
  - matplotlib>=3.3  
```

You can activate this environment with:  

```bash  
conda env create -f environment.yml  
conda activate QCOT  
```

---

## Summary  

This package unifies the numerical and theoretical verification of the Cognitive Uncertainty Principle across two complementary approaches:  
1. **Empirical bootstrap estimation** of the constant C β.  
2. **Theoretical jump–diffusion modeling** confirming the invariant limit Δε·ΔD KL ≥ C.  
3. **Scale-adaptive uncertainty law** formalizing CUP as a family of observer-dependent bounds that continuously interpolate between differential and statistical cognition via the scaling function C(ξ,ε), with robust empirical confirmation of the phase boundary at λ_c ≈ 0.01.  

All materials are released for transparent replication and further analysis.  

---

## License  

All code, figures, and data are distributed under an open scientific license.  
Reproduction and derivative research are encouraged with appropriate citation.  

---

## Version History  

| Version | Description |  
|----------|-------------|  
| **v1.0** | Initial empirical verification with bootstrap estimation, confidence intervals, and violation morphology. |  
| **v2.0** | Extended dual verification including jump–diffusion theoretical model and perceptual uncertainty trade-off. |  
| **v3.0** | Introduction of the epistemic phase boundary and dual cognitive regimes; empirical validation of the scaling law C(ξ,ε) via 252-configuration sweep; formal statement of Theorem 1 (Observer-locality breakdown) and Hypothesis 1 (Dual-regime duality); identification of critical jump intensity λ_c ≈ 0.01. |  
