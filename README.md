# NO2_Global_pkg

Deep-learning pipeline for estimating global surface NO₂ concentrations, combining
GEOS-Chem High Performance (GCHP) simulations, satellite NO₂ retrievals (TROPOMI, OMI),
ground-based observations, and geophysical/land-use predictors.

> **Status:** Research code accompanying the paper *"A Global Dataset of Surface NO₂
> Estimation Based on Machine Learning Using Process-based Information"*
> (Yan et al., 2026). Archived release: https://doi.org/10.5281/zenodo.19740022.

---

## Overview

The pipeline has four stages:

1. **Data processing** (`Data_Processing/`) — build the model input predictors and the
   observational training label:
   - Regrid GCHP output (`Regrid_GCHP/`)
   - Derive the geophysical NO₂ prior from satellite retrievals via tessellation
     (`Derive_Geophysical_NO2/`)
   - Derive input predictors: meteorology, emissions (CEDS, GFED), land cover, NDVI,
     ISA, population, road density (OpenStreetMap), geography (`Get_*_Input/`)
   - Derive the ground-observation label (`Derive_Label/obs_pipeline_v7/`)
   - Assemble the training dataset (`derive_TrainingDatasets/`)
2. **Training** (`Training_pkg/`) — train the deep-learning model (ResNet-style CNN and a
   LightGBM benchmark), with feature selection via LassoNet (`lassonet/`).
3. **Estimation** (`Estimation_pkg/`) — apply the trained model to produce global NO₂ maps.
4. **Evaluation & uncertainty** (`Evaluation_pkg/`, `Uncertainty_pkg/`,
   `Mahalanobis_Uncertainty_pkg/`) — spatial cross-validation (BLCO/BLOO), SHAP analysis,
   and Mahalanobis-distance-based uncertainty quantification.

Plotting utilities live in `visualization_pkg/`. The main entry point is `main.py`,
configured through `config.toml`.

## Repository structure

```
main.py                       Entry point (training / estimation / evaluation driver)
config.toml                   Central configuration (paths, switches, hyperparameters)
wandb_*_config.py             Weights & Biases sweep configs (ResNet, LightGBM)
Training_pkg/                 Model construction, data loaders, loss, training loop
Estimation_pkg/               Prediction over the global grid + quality control
Evaluation_pkg/               Cross-validation, SHAP, hyperparameter search
Uncertainty_pkg/              Baseline uncertainty estimation
Mahalanobis_Uncertainty_pkg/  Mahalanobis-distance uncertainty (adapted, see Credits)
Mask_func_pkg/                Land/region masking utilities
visualization_pkg/            Figure generation
lassonet/                     LassoNet feature selection (third-party, see Credits)
Data_Processing/              Input-predictor and label derivation (see Overview)
```

## Requirements

- Python 3.8+
- Core libraries: NumPy, pandas, PyTorch, LightGBM, xarray, netCDF4, SciPy,
  scikit-learn, Cartopy, matplotlib
- Optional: Weights & Biases (`wandb`) for hyperparameter sweeps
- Fortran compiler (gfortran or Intel `ifx`) for the tessellation regridder in
  `Data_Processing/Derive_Geophysical_NO2/Tessellation/gfortran_0p025_global/`

External data credentials (set as environment variables, **never** hardcoded):

```bash
export OPENAQ_API_KEY="your-openaq-key"   # required by Data_Processing/.../openaq/
```

## Usage

Configuration is driven by `config.toml`. Paths in the configuration and some
processing scripts refer to the authors' compute environment and must be adjusted
to your own data locations.

```bash
# Train / estimate / evaluate (behaviour controlled by switches in config.toml)
python main.py
```

Each `Data_Processing/` sub-stage has its own `main.py` and is run independently to
generate the corresponding input variable; see the sub-folder scripts.

## Credits and third-party code

This project builds on and adapts previously published code, gratefully acknowledged:

- **Modelling framework** adapted from the PM2.5 chemical-component estimation code of
  Shen et al.: https://github.com/Siyuan-Shen/PM25_Species_NA
- **Mahalanobis-distance uncertainty** (`Mahalanobis_Uncertainty_pkg/`) adapted from:
  https://github.com/Siyuan-Shen/Mahalanobis_Uncertainty_Regional_Composition
- **LassoNet** (`lassonet/`) — feature selection, from the open-source LassoNet package
  (Lemhadri, Ruan, Abraham & Tibshirani, JMLR 2021; MIT License):
  https://github.com/lasso-net/lassonet

All other analyses use standard open-source Python libraries (NumPy, pandas, PyTorch,
LightGBM, xarray, SciPy, scikit-learn, Cartopy).

## License

Released under the MIT License — see [LICENSE](LICENSE). Note that adapted third-party
components remain subject to their original licenses (see Credits).

## Citation

If you use this code, please cite the associated paper and the software archive:

> Yan, Y., Shen, S., Zhang, Y., Li, C., Zhu, H., Chatterjee, D., Singh, I., &
> Martin, R. V. (2026). *A Global Dataset of Surface NO₂ Estimation Based on
> Machine Learning Using Process-based Information.* Nature Scientific Data (Under Review).

See [CITATION.cff](CITATION.cff) for machine-readable metadata.
