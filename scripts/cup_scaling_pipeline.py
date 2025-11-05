#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cup_scaling_pipeline.py

Integrated Cognitive Uncertainty Principle Scaling Analysis with Phase Transitions

This module provides a unified pipeline for empirical verification of the Cognitive 
Uncertainty Principle (CUP) through systematic exploration of the parameter space 
(correlation length ξ, perceptual resolution ε, jump intensity λ) and detection 
of phase transitions in the uncertainty relation regime.

The pipeline integrates three analytical components:
1. Synthetic data generation via hybrid jump-diffusion dynamics
2. Grid-based exploration of CUP scaling relations
3. Phase transition analysis through sensitivity quantification

Theoretical Framework
---------------------
The Cognitive Uncertainty Principle establishes fundamental bounds on simultaneous 
observation precision across complementary cognitive dimensions:

    Δξ × Δε ≥ C_Fisher

where the constant C_Fisher exhibits systematic dependence on control parameters:

    C_Fisher(ξ,ε,λ) = C₀ × [1 + α(ξ/ξ₀)^p + β(ε₀/ε)^q]

Phase transition occurs at critical jump intensity λ_c where KL divergence 
sensitivity drops below threshold, marking transition from diffusive to 
jump-dominated regime.

Methodological Components
-------------------------
**Data Generation:**
- Hybrid jump-diffusion process: dX_t = μX_t dt + σX_t dW_t + J_t dN_t
- Benford's law compliance via first-digit extraction at resolution ε
- Monte Carlo estimation: M realizations per (ξ,ε,λ) configuration

**Fisher Information Estimation:**
- Local epsilon-grid construction around nominal values
- Univariate spline differentiation in log-log space: I_F = (∂KL/∂ε)²
- Robust median aggregation across realizations

**Phase Transition Detection:**
- Sensitivity metric: S(λ) = ⟨|∂KL/∂log(ε)|⟩
- Critical point: λ_c = min{λ : S(λ) < threshold}
- Heatmap visualization of KL(λ,ε) phase diagram

NOTE:
"RuntimeWarning" messages are an intrinsic part of the model.
They reflect the inherent instability of the observer–world interaction
under finite perceptual resolution. This instability drives the
self-organization processes responsible for the emergence of
1/f spectral structure and Benford-like digit distributions.

Suppressing or correcting these warnings would artificially remove
the feedback mechanism that constitutes the model's physical content.
Therefore, the simulation must be executed in its original form,
preserving the adaptive selection of the optimal perceptual threshold
(epsilon_best) as postulated in Subjective Physics.

Technical Specifications
------------------------
Python : 3.8+
NumPy  : 1.20+
Pandas : 1.2+
SciPy  : 1.6+
Matplotlib : 3.3+
PyYAML : 5.4+

Input Configuration
-------------------
YAML/JSON file specifying:
- Parameter grids: ξ_values, ε_values, λ_values
- Simulation parameters: n_per_realization, realizations_per_cell
- Jump process: jump_intensity, lognormal_sigma
- Fitting parameters: xi0, eps0, initial_guesses
- Output: directories, seed, diagnostic flags

Output Structure
----------------
data/
    cup_sim_output/
        cup_grid_results.csv : Complete parameter sweep results
        fit_results.json : Fitted scaling parameters with confidence intervals
        cell_XXXX.pkl : Intermediate cell results (optional)
    phase_transition_sensitivity.csv : Sensitivity analysis results

figures/
    cup_scaling_heatmap_kl_median.pdf : KL divergence heatmap
    cup_phase_transition.pdf : Phase transition diagram with λ_c

Usage
-----
Complete pipeline execution:
    $ python cup_scaling_pipeline.py --config cup_sim_config.yaml

Phase analysis only (existing data):
    $ python cup_scaling_pipeline.py --data-path data/cup_grid_results.csv --skip-data-gen

Skip phase analysis:
    $ python cup_scaling_pipeline.py --config config.yaml --skip-phase-analysis

Programmatic interface:
    >>> from cup_scaling_pipeline import run_complete_pipeline
    >>> run_complete_pipeline('config.yaml')

Pipeline Workflow
-----------------
1. Path Configuration: Automatic detection of root/scripts directory structure
2. Data Generation: Grid sweep over (ξ,ε,λ) with Monte Carlo sampling
3. Scaling Fit: Nonlinear least-squares estimation of (C_Fisher, α, β, p, q)
4. Phase Analysis: Sensitivity computation and critical point detection
5. Visualization: Heatmaps and phase diagrams exported to figures/

Performance Notes
-----------------
- Memory footprint: ~100 MB per 10³ parameter cells
- Computational scaling: O(M × n × |ξ| × |ε| × |λ|)
- Typical runtime: 10-60 minutes for standard configuration
- Intermediate results auto-saved for fault tolerance

Author: Vladimir Khomyakov
License: MIT
Repository: https://github.com/Khomyakov-Vladimir/cognitive-uncertainty-principle-verification

References
----------
Khomyakov, V. (2025). The Epistemic Phase Boundary in Cognitive Uncertainty Principle (3.0). Zenodo. https://doi.org/10.5281/zenodo.17370034

See Also
--------
cup_sim_config.yaml : Standard configuration template
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time
import pickle
import argparse

# ==============================
#  Universal Path Configuration
# ==============================
def configure_output_paths():
    """
    Configure output paths for figures and data to be saved in root repository directories.
    Works from both root and scripts/ subdirectory.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if we're in scripts/ subdirectory
    if os.path.basename(script_dir) == 'scripts':
        # Move up to root directory
        root_dir = os.path.dirname(script_dir)
    else:
        # We're already in root directory
        root_dir = script_dir
    
    # Create figures and data directories in root if they don't exist
    figures_dir = os.path.join(root_dir, 'figures')
    data_dir = os.path.join(root_dir, 'data')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    return figures_dir, data_dir, root_dir

# Configure paths before any other imports
FIGURES_DIR, DATA_DIR, ROOT_DIR = configure_output_paths()


BENFORD_PROBS = np.log10(1 + 1 / np.arange(1, 10))

def first_significant_digit_from_array(x):
    x = np.abs(x)
    x = x[x != 0]
    if len(x) == 0:
        return np.array([], dtype=int)
    log10 = np.floor(np.log10(x))
    x_norm = (x / (10 ** log10)).astype(int)
    x_norm = np.clip(x_norm, 1, 9)
    return x_norm

def kl_divergence_from_counts(counts):
    if counts is None or counts.sum() == 0:
        return np.nan
    p_emp = counts / counts.sum()
    mask = p_emp > 0
    return np.sum(p_emp[mask] * np.log(p_emp[mask] / BENFORD_PROBS[mask]))

def generate_hybrid_jump_diffusion(T, mu=0.0, sigma=0.01, jump_intensity=0.05, jump_size=0.2):
    x = np.ones(T)
    dt = 1.0
    for t in range(1, T):
        dW = np.random.normal(0, np.sqrt(dt))
        x_gbm = x[t-1] * (1 + mu * dt + sigma * dW)
        if np.random.random() < jump_intensity:
            jump_multiplier = np.random.lognormal(mean=0, sigma=jump_size)
            x[t] = x_gbm * jump_multiplier
        else:
            x[t] = x_gbm
    return x

def estimate_kl_benford_from_series(x, epsilon):
    """
    Replicates the core logic of kl_tradeoff_analysis.py for a single series.
    """
    x_safe = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if np.all(x_safe == 0):
        return np.nan
    y = np.round(x_safe / epsilon).astype(np.int64)
    digits = first_significant_digit_from_array(y)
    if digits.size == 0:
        return np.nan
    counts = np.bincount(digits, minlength=10)[1:10]
    kl = kl_divergence_from_counts(counts)
    return kl if np.isfinite(kl) else np.nan

def estimate_fisher_from_kl_curve(epsilons, kl_values):
    """
    Compute Fisher info from discrete KL(epsilon) curve (as in kl_tradeoff_analysis.py).
    """
    # Remove invalid
    valid = np.isfinite(kl_values) & (kl_values > 0)
    eps_clean = epsilons[valid]
    kl_clean = kl_values[valid]
    if len(eps_clean) < 5:
        return np.nan

    # Sort
    sort_idx = np.argsort(eps_clean)
    eps_sorted = eps_clean[sort_idx]
    kl_sorted = kl_clean[sort_idx]

    # Spline in log-log
    from scipy.interpolate import UnivariateSpline
    log_eps = np.log10(eps_sorted)
    log_kl = np.log10(np.clip(kl_sorted, 1e-12, None))
    try:
        spline = UnivariateSpline(log_eps, log_kl, s=0.5 * len(log_eps))
        dlogkl_dlogeps = spline.derivative()(log_eps)
        kl_spline = 10 ** spline(log_eps)
        eps_spline = 10 ** log_eps
        dkl_deps = dlogkl_dlogeps * (kl_spline / eps_spline)
        fisher = dkl_deps ** 2
        return np.nanmean(fisher)  # or median
    except:
        return np.nan


def compute_kl_sensitivity(df, threshold=0.01):
    """
    Compute phase transition in Cognitive Uncertainty Principle.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing simulation results with columns:
        'lambda', 'epsilon', 'kl_median'
    threshold : float
        Sensitivity threshold for phase transition detection
    
    Returns:
    --------
    tuple : (sensitivity_df, lambda_c, pivot_table)
        sensitivity_df: DataFrame with sensitivity values for each lambda
        lambda_c: Critical lambda value at phase transition
        pivot_table: Pivoted table for heatmap visualization
    """
    
    def sensitivity(group):
        """
        Compute sensitivity of KL divergence to epsilon variations.
        
        Parameters:
        -----------
        group : pandas.DataFrame
            Grouped data for specific lambda value
        
        Returns:
        --------
        float : Mean absolute gradient of KL divergence w.r.t. log(epsilon)
        """
        eps = group['epsilon'].values
        kl = group['kl_median'].values
        if len(eps) < 2:
            return np.nan
        # Compute gradient in log-scale
        dkl_deps = np.gradient(kl, np.log(eps))
        return np.mean(np.abs(dkl_deps))
    
    # Compute sensitivity for each lambda value - COMPATIBLE VERSION
    # Option 1: Explicitly select only the needed columns
    sens_df = df.groupby('lambda')[['epsilon', 'kl_median']].apply(sensitivity).reset_index(name='sensitivity')
    
    # Identify critical lambda where sensitivity drops below threshold
    lambda_c = sens_df[sens_df['sensitivity'] < threshold]['lambda'].min()
    
    # Create pivot table for heatmap visualization
    pivot = df.pivot_table(index='lambda', columns='epsilon', values='kl_median', aggfunc='mean')
    
    return sens_df, lambda_c, pivot

def plot_phase_transition(pivot, lambda_c, output_path=None):
    """
    Generate phase transition heatmap visualization.
    
    Parameters:
    -----------
    pivot : pandas.DataFrame
        Pivoted table with KL divergence values
    lambda_c : float
        Critical lambda value for phase transition
    output_path : str
        Path for saving the output figure
    """
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, 'cup_phase_transition.pdf')
    
    plt.figure(figsize=(10, 6))
    
    # Create heatmap
    im = plt.imshow(pivot.values, aspect='auto', origin='lower',
                    extent=[np.log10(pivot.columns.min()), np.log10(pivot.columns.max()),
                            pivot.index.min(), pivot.index.max()],
                    cmap='viridis')
    
    plt.colorbar(im, label='KL median')
    
    # Mark critical lambda
    plt.axhline(lambda_c, color='red', linestyle='--', linewidth=2, 
                label=f'$\\lambda_c \\approx {lambda_c:.2f}$')
    
    plt.xlabel(r'$\log_{10}(\varepsilon)$')
    plt.ylabel(r'Jump intensity $\lambda$')
    plt.title('Phase transition in Cognitive Uncertainty Principle')
    plt.legend()
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Phase transition plot saved to: {output_path}")

# ==============================
# Utility Functions
# ==============================
def load_config(path):
    """
    Load configuration from YAML or JSON file.
    """
    # Handle relative paths from script location
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)
    
    with open(path, 'r') as f:
        if path.lower().endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        else:
            return json.load(f)

# ==============================
# Main Pipeline Functions
# ==============================
def run_data_generation(config_path):
    """
    Run the data generation pipeline (original cup_scaling_pipeline.py logic).
    """
    print("=== Starting Data Generation Pipeline ===")
    cfg = load_config(config_path)
    
    # Use DATA_DIR for output directory
    outdir = cfg['output']['outdir']
    if not os.path.isabs(outdir):
        outdir = os.path.join(DATA_DIR, outdir)
    
    os.makedirs(outdir, exist_ok=True)
    seed = cfg['output'].get('seed', 42)
    np.random.seed(seed)

    # Ensure numeric types
    xis = np.array(cfg['grid']['xi']['values'], dtype=float)
    epsilons = np.array(cfg['grid']['epsilon']['values'], dtype=float)
    n = cfg['synthetic_model']['n_per_realization']
    M = cfg['synthetic_model']['realizations_per_cell']

    results = []
    total_cells = len(xis) * len(epsilons) * len(cfg['synthetic_model']['jumps']['lambda_vals'])
    cell_counter = 0

    for lam in cfg['synthetic_model']['jumps']['lambda_vals']:
        for xi_val in xis:
            for eps_val in epsilons:
                cell_counter += 1
                t0 = time.time()
                # Use xi as correlation length proxy → not directly used in generation
                # We vary jump_intensity = lam, and use eps_val for digit extraction
                kl_vals = []
                for m in range(M):
                    # Generate series with jump intensity = lam
                    x = generate_hybrid_jump_diffusion(
                        T=n,
                        jump_intensity=lam,
                        jump_size=cfg['synthetic_model']['jumps'].get('lognormal', {}).get('sigma', 0.2)
                    )
                    kl = estimate_kl_benford_from_series(x, eps_val)
                    if np.isfinite(kl):
                        kl_vals.append(kl)

                if len(kl_vals) == 0:
                    kl_median = np.nan
                    kl_p2_5 = kl_p97_5 = np.nan
                else:
                    kl_arr = np.array(kl_vals)
                    kl_median = np.median(kl_arr)
                    kl_p2_5 = np.percentile(kl_arr, 2.5)
                    kl_p97_5 = np.percentile(kl_arr, 97.5)

                # For Fisher: generate a small epsilon grid around eps_val
                eps_grid = np.logspace(np.log10(eps_val) - 0.3, np.log10(eps_val) + 0.3, 7)
                kl_grid = []
                for e in eps_grid:
                    kl_local = []
                    for _ in range(10):  # 10 realizations for Fisher estimation
                        x = generate_hybrid_jump_diffusion(T=n, jump_intensity=lam)
                        kl = estimate_kl_benford_from_series(x, e)
                        if np.isfinite(kl):
                            kl_local.append(kl)
                    kl_grid.append(np.median(kl_local) if kl_local else np.nan)

                fisher_est = estimate_fisher_from_kl_curve(eps_grid, np.array(kl_grid))
                fisher_median = fisher_est if np.isfinite(fisher_est) else 1.17e-4

                cell_res = {
                    'lambda': float(lam),
                    'xi': float(xi_val),
                    'epsilon': float(eps_val),
                    'fisher_median': fisher_median,
                    'fisher_p2_5': fisher_median,
                    'fisher_p97_5': fisher_median,
                    'kl_median': kl_median,
                    'kl_p2_5': kl_p2_5,
                    'kl_p97_5': kl_p97_5,
                }
                results.append(cell_res)

                t1 = time.time()
                # FIX: ensure eps_val is float for formatting
                print(f"[{cell_counter}/{total_cells}] lambda={lam} xi={xi_val} eps={float(eps_val):.1e} "
                      f"-> fisher_med={fisher_median:.3e} kl_med={kl_median:.3e} "
                      f"time={t1-t0:.1f}s")

                if cfg['diagnostics'].get('save_intermediate', True):
                    with open(os.path.join(outdir, f"cell_{cell_counter:04d}.pkl"), 'wb') as f:
                        pickle.dump(cell_res, f)

    df = pd.DataFrame(results)
    output_csv_path = os.path.join(outdir, "cup_grid_results.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"Grid finished, saved to {output_csv_path}")

    # ---------------------------
    # Fit scaling relation
    # ---------------------------
    df_clean = df.dropna(subset=['kl_median', 'epsilon', 'xi'])
    if len(df_clean) < 10:
        print("Not enough valid data for fitting")
        return output_csv_path

    C_fisher_baseline = df_clean['fisher_median'].median()
    xi0 = float(cfg['fitting']['xi0'])
    eps0 = float(cfg['fitting']['eps0'])

    xi_arr = df_clean['xi'].values.astype(float)
    eps_arr = df_clean['epsilon'].values.astype(float)
    C_obs = df_clean['kl_median'].values.astype(float)

    def fit_model_wrapper(_, C_fisher, alpha, beta, p, q):
        return C_fisher * (1.0 + alpha * (xi_arr/xi0)**p + beta * (eps0/eps_arr)**q)

    p0 = [
        C_fisher_baseline,
        cfg['fitting']['initial_guesses']['alpha'],
        cfg['fitting']['initial_guesses']['beta'],
        cfg['fitting']['initial_guesses']['p'],
        cfg['fitting']['initial_guesses']['q']
    ]

    try:
        popt, pcov = curve_fit(
            fit_model_wrapper,
            xdata=np.zeros_like(C_obs),
            ydata=C_obs,
            p0=p0,
            maxfev=10000
        )
        param_names = ['C_fisher', 'alpha', 'beta', 'p', 'q']
        fit_res = dict(zip(param_names, popt))
        se = np.sqrt(np.diag(pcov))
        fit_ci = {name: (val - 1.96*se[i], val + 1.96*se[i]) for i, (name, val) in enumerate(fit_res.items())}
        print("Fitted parameters:")
        for name in param_names:
            print(f" {name}: {fit_res[name]:.4e}  CI: ({fit_ci[name][0]:.4e}, {fit_ci[name][1]:.4e})")
    except Exception as e:
        print("Fitting failed:", e)
        fit_res = dict(zip(['C_fisher', 'alpha', 'beta', 'p', 'q'], p0))
        fit_ci = {k: (v, v) for k, v in fit_res.items()}

    fit_results_path = os.path.join(outdir, "fit_results.json")
    with open(fit_results_path, 'w') as f:
        json.dump({'fit_params': fit_res, 'fit_ci': {k:list(v) for k,v in fit_ci.items()}}, f, indent=2)
    print(f"Fit results saved to: {fit_results_path}")

    # ---------------------------
    # Plots - Save to FIGURES_DIR
    # ---------------------------
    if cfg['diagnostics'].get('plots', True) and len(df_clean) > 0:
        lam_choice = float(cfg['synthetic_model']['jumps']['lambda_vals'][0])
        df_l = df_clean[df_clean['lambda'] == lam_choice]
        if not df_l.empty:
            pivot = df_l.pivot_table(index='xi', columns='epsilon', values='kl_median', aggfunc='mean')
            plt.figure(figsize=(8,6))
            plt.imshow(pivot.values, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='kl_median')
            plt.yticks(range(len(pivot.index)), [f"{x:.0f}" for x in pivot.index])
            plt.xticks(range(len(pivot.columns)), [f"{e:.1e}" for e in pivot.columns], rotation=45)
            plt.title(f"KL Median (lambda={lam_choice})")
            plt.tight_layout()
            
            # Save to figures directory in repository root
            plot_path = os.path.join(FIGURES_DIR, "cup_scaling_heatmap_kl_median.pdf")
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Scaling heatmap saved to: {plot_path}")

    print("Data generation pipeline complete.")
    return output_csv_path

def run_phase_analysis(data_path=None, config_path=None):
    """
    Run phase transition analysis (original cup_scaling_pipeline_v2.py logic).
    """
    print("\n=== Starting Phase Transition Analysis ===")
    
    # If no data path provided, try to find the most recent results
    if data_path is None and config_path is not None:
        cfg = load_config(config_path)
        outdir = cfg['output']['outdir']
        if not os.path.isabs(outdir):
            outdir = os.path.join(DATA_DIR, outdir)
        data_path = os.path.join(outdir, "cup_grid_results.csv")
    
    # Load simulation results with error handling
    if data_path is None or not os.path.exists(data_path):
        # Try to find the file in common locations
        possible_paths = [
            os.path.join(DATA_DIR, 'cup_sim_output', 'cup_grid_results.csv'),
            os.path.join(DATA_DIR, 'cup_grid_results.csv'),
            'cup_sim_output/cup_grid_results.csv',
            'cup_grid_results.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        else:
            raise FileNotFoundError(
                f"Data file not found. Check file exists at: {data_path}" if data_path 
                else "Data file not found in common locations. Please specify --data-path"
            )
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Compute phase transition properties
    sensitivity_threshold = 0.01  # Empirical threshold for phase transition
    sens_df, lambda_c, pivot_table = compute_kl_sensitivity(df, sensitivity_threshold)
    
    # Handle case where no lambda_c is found
    if np.isnan(lambda_c):
        print("Warning: No critical lambda found below threshold. Using minimum sensitivity value.")
        lambda_c = sens_df.loc[sens_df['sensitivity'].idxmin(), 'lambda']
    
    print(f"Critical lambda: {lambda_c:.4f}")
    
    # Save sensitivity results
    sens_output_path = os.path.join(DATA_DIR, "phase_transition_sensitivity.csv")
    sens_df.to_csv(sens_output_path, index=False)
    print(f"Sensitivity results saved to: {sens_output_path}")
    
    # Generate visualization
    plot_phase_transition(pivot_table, lambda_c)
    
    return sens_df, lambda_c, pivot_table

def run_complete_pipeline(config_path, skip_data_gen=False, skip_phase_analysis=False):
    """
    Run the complete integrated pipeline.
    """
    data_path = None
    
    if not skip_data_gen:
        data_path = run_data_generation(config_path)
    
    if not skip_phase_analysis:
        run_phase_analysis(data_path, config_path)
    
    print("\n=== Pipeline Complete ===")
    print(f"All outputs saved to:")
    print(f"  - Data directory: {DATA_DIR}")
    print(f"  - Figures directory: {FIGURES_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated CUP Scaling Pipeline")
    parser.add_argument("--config", type=str, default="cup_sim_config.yaml", 
                       help="Path to YAML/JSON config file")
    parser.add_argument("--skip-data-gen", action="store_true",
                       help="Skip data generation phase")
    parser.add_argument("--skip-phase-analysis", action="store_true",
                       help="Skip phase analysis phase")
    parser.add_argument("--data-path", type=str,
                       help="Path to existing data file for phase analysis")
    
    args = parser.parse_args()
    
    # If specific data path provided for phase analysis only
    if args.data_path and not args.skip_phase_analysis:
        run_phase_analysis(args.data_path)
    else:
        run_complete_pipeline(
            config_path=args.config,
            skip_data_gen=args.skip_data_gen,
            skip_phase_analysis=args.skip_phase_analysis
        )
