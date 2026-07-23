import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import toml
import os
import pickle
from pathlib import Path
import joblib

# ===== CRITICAL FIX: Patch torch.load to always use CPU =====
# This prevents CUDA deserialization errors when loading models trained on GPU
_original_torch_load = torch.load

def patched_torch_load(f, map_location=None, *args, **kwargs):
    """Patched torch.load that defaults to CPU if no map_location specified"""
    if map_location is None:
        map_location = torch.device('cpu')
    return _original_torch_load(f, map_location=map_location, *args, **kwargs)

torch.load = patched_torch_load
print("✓ Applied CPU map_location patch to torch.load")
# ===== END OF PATCH =====

from Training_pkg.iostream import load_TrainingVariables, load_geophysical_biases_data, load_geophysical_species_data, load_monthly_obs_data, Learning_Object_Datasets
from Training_pkg.utils import *
from Training_pkg.data_func import normalize_Func, get_trainingdata_within_start_end_YEAR
from Training_pkg.Net_Construction import *
from Training_pkg.Statistic_Func import linear_regression

from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import Get_valid_index_for_temporal_periods,Get_month_based_Index,Get_month_based_XY_indices,GetXIndex,GetYIndex,Get_XY_indices, Get_XY_arraies, Get_final_output, ForcedSlopeUnity_Func, CalculateAnnualR2, CalculateMonthR2, calculate_Statistics_results
from Evaluation_pkg.iostream import *

from lassonet import LassoNetRegressor, LassoNetRegressorCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from functools import partial

from visualization_pkg.LassoNet_plot import plot_Stability_Selection

# *------------------------------------------------------------------------------*#
##   Initialize the array, variables and constants.
# *------------------------------------------------------------------------------*#
print("Loading data...")
### Get training data, label data, initial observation data and geophysical species
width, height, sitesnumber, start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=LassoNet_channel_names)
SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
geophysical_species, geolat, geolon = load_geophysical_species_data(species=species)
true_input, mean, std = Learning_Object_Datasets(bias=bias, Normalized_bias=normalize_bias, Normlized_Speices=normalize_species, Absolute_Species=absolute_species, Log_PM25=log_species, species=species)
Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets, observation_data=SPECIES_OBS)
population_data = load_coMonitor_Population()

# Prepare training data
imodel_year = 0
Normalized_TrainingData = get_trainingdata_within_start_end_YEAR(
    initial_array=Initial_Normalized_TrainingData, 
    training_start_YYYY=LassoNet_beginyears[imodel_year],
    training_end_YYYY=LassoNet_endyears[imodel_year],
    start_YYYY=start_YYYY,
    sitesnumber=sitesnumber
)

print('Shape of Normalized_TrainingData:', Normalized_TrainingData.shape)

# Get valid indices
valid_sites_index, temp_index_of_initial_array = Get_valid_index_for_temporal_periods(
    SPECIES_OBS=SPECIES_OBS,
    beginyear=beginyears[imodel_year],
    endyear=endyears[imodel_year],
    month_range=list(range(0, 12)),
    sitesnumber=sitesnumber
)

# Prepare data for LassoNet (no train/test split needed)
# Extract the center pixel (2,2) and remove NaN values
X_data = Normalized_TrainingData[:, :, 2, 2]  # Shape: (samples, features)
y_data = true_input

# Remove samples with NaN in target
valid_mask = ~np.isnan(y_data)
X_masked = X_data[valid_mask]
y_masked = y_data[valid_mask]

print(f"Data shape after masking: X={X_masked.shape}, y={y_masked.shape}")

def save_stability_results(results_tuple, results_dir, tag, version):
    """
    Efficiently save stability selection results
    Converts any torch tensors to CPU before saving to avoid CUDA issues
    """
    oracle, order, wrong, paths, prob = results_tuple
    
    # Create directory if it doesn't exist
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Helper function to convert torch tensors to CPU
    def to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_cpu(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(to_cpu(item) for item in obj)
        else:
            return obj
    
    # Convert paths to CPU if it contains tensors
    paths_cpu = to_cpu(paths)
    
    # Save numpy arrays
    np.save(Path(results_dir) / f"{version}_{tag}_oracle.npy", oracle)
    np.save(Path(results_dir) / f"{version}_{tag}_order.npy", order)
    np.save(Path(results_dir) / f"{version}_{tag}_wrong.npy", wrong)
    np.save(Path(results_dir) / f"{version}_{tag}_prob.npy", prob)
    
    # Save paths with pickle (now guaranteed to be CPU-only)
    with open(Path(results_dir) / f"{version}_{tag}_paths.pkl", "wb") as f:
        pickle.dump(paths_cpu, f)
    
    print(f"Results saved successfully to: {results_dir}")

def save_trained_model(model, model_dir, tag, version):
    """Save the trained LassoNet model"""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(model_dir) / f"{version}_{tag}_model.pkl"
    
    try:
        # Move model to CPU before saving to ensure compatibility
        if hasattr(model, 'module_') and model.module_ is not None:
            model.module_ = model.module_.to(torch.device('cpu'))
        if hasattr(model, 'device'):
            model.device = torch.device('cpu')
        
        # Save the entire model using joblib (recommended for sklearn-compatible models)
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_trained_model(model_path):
    """Load a previously trained LassoNet model with CPU fallback"""
    try:
        # Determine device - always use CPU for loading to avoid CUDA errors
        device = torch.device('cpu')
        
        # Load with joblib
        model = joblib.load(model_path)
        
        # If model has internal torch tensors/modules, move them to CPU
        if hasattr(model, 'module_') and model.module_ is not None:
            model.module_ = model.module_.to(device)
        
        # Update model's device attribute if it exists
        if hasattr(model, 'device'):
            model.device = device
            
        print(f"Model loaded from: {model_path} (device: {device})")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def load_stability_results_safe(results_dir, tag, version):
    """Safely load stability selection results, handling CUDA tensors in pickle files"""
    try:
        # Load numpy arrays (these should be fine)
        oracle = np.load(Path(results_dir) / f"{version}_{tag}_oracle.npy")
        order = np.load(Path(results_dir) / f"{version}_{tag}_order.npy")
        wrong = np.load(Path(results_dir) / f"{version}_{tag}_wrong.npy")
        prob = np.load(Path(results_dir) / f"{version}_{tag}_prob.npy")
        
        # Load paths with pickle - this might contain torch tensors
        paths_file = Path(results_dir) / f"{version}_{tag}_paths.pkl"
        
        with open(paths_file, "rb") as f:
            # The torch.load patch should handle this, but add extra safety
            try:
                paths = pickle.load(f)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"Warning: CUDA error loading paths, attempting CPU remapping...")
                    # If the pickle contains torch objects, they should be handled by the patch
                    # But if not, we can try reloading
                    f.seek(0)
                    
                    # Force CPU device for any torch tensors in the pickle
                    import io
                    
                    class CPUUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            if module == 'torch.storage' and name == '_load_from_bytes':
                                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                            else:
                                return super().find_class(module, name)
                    
                    paths = CPUUnpickler(f).load()
                else:
                    raise
        
        print(f"Results loaded successfully from: {results_dir}")
        return oracle, order, wrong, paths, prob
        
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def check_model_exists(model_dir, tag, version):
    """Check if a trained model already exists"""
    model_path = Path(model_dir) / f"{version}_{tag}_model.pkl"
    return model_path.exists()

def slug_hidden(hidden):
    """Convert hidden layer tuple to string slug"""
    return "h" + "_".join(str(x) for x in hidden)

def run_stability_selection_and_save(model, tag, X_data, y_data, total_channel_names, nchannel, 
                                   train_new=True, save_model=True):
    """Run stability selection and save results with model saving option"""
    print(f"\n--- Running stability selection for {tag} ---")
    
    # Define directories
    results_dir = Path(txt_outdir) / f"{species}/{version}/Results/results-lassonet-stability-selection/{special_name}_{tag}_{nchannel}channel"
    model_dir = Path(txt_outdir) / f"{species}/{version}/Models/lassonet-models/{special_name}_{tag}_{nchannel}channel"
    
    stability_results = None
    
    if train_new:
        # Run stability selection
        stability_results = model.stability_selection(X_data, y_data)
        
        # Save results
        save_stability_results(stability_results, results_dir, tag, version)
        
        # Save trained model if requested
        if save_model:
            save_trained_model(model, model_dir, tag, version)
            
        print(f"Training completed. Results saved to: {results_dir}")
        
    else:
        print(f"Skipping training. Results should already exist at: {results_dir}")
    
    return stability_results, results_dir

def generate_plots_from_results(results_dir, tag, nchannel, total_channel_names, 
                               Normalized_TrainingData, version):
    """Generate plots from saved results (separate from training)"""
    plot_dir = Path(txt_outdir) / f"{species}/{version}/Figures/figures-LassoNet-stability-selection/{special_name}_{tag}_{nchannel}channel"
    
    try:
        # Create plots
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_Stability_Selection(results_dir, plot_dir, tag, nchannel, total_channel_names, Normalized_TrainingData, version)
        print(f"Plots saved to: {plot_dir}")
        return True
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

print("Starting stability selection experiments...")

# Set up device with optimized GPU handling
print("=== GPU Setup & Diagnostics ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"CUDA available - Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
    
    print(f"Selected device: {device}")
    print("Note: LassoNet handles device placement internally - keeping data as numpy arrays")
else:
    print("CUDA not available - using CPU")
    print(f"  PyTorch version: {torch.__version__}")
    print("  Possible reasons: No GPU, missing drivers, CPU-only PyTorch")
    device = torch.device("cpu")
    print(f"Using device: {device}")
print("=" * 35)

def LassoNet_Stability_Search(total_channel_names, nchannel):
    """Main function for LassoNet stability selection with flexible training/plotting"""
    print(f"Starting LassoNet Stability Selection - Device: {device}")
    print(f"Configuration: TRAIN_MODELS={TRAIN_MODELS}, SAVE_MODELS={SAVE_MODELS}, GENERATE_PLOTS={GENERATE_PLOTS}")
    
    success_count = 0
    total_count = len(LassoNet_hidden_layers)
    
    # Store results for batch plotting if needed
    experiment_results = []
    
    for i, hidden in enumerate([tuple(hidden) for hidden in LassoNet_hidden_layers], 1):
        print(f"\nExperiment {i}/{total_count}: {hidden}")
        
        hslug = slug_hidden(hidden)
        tag = f"LassoNetCV_{hslug}"
        
        # Define model directory for checking existing models
        model_dir = Path(txt_outdir) / f"{species}/{version}/Models/lassonet-models/{special_name}_{tag}_{nchannel}channel"
        results_dir = Path(txt_outdir) / f"{species}/{version}/Results/results-lassonet-stability-selection/{special_name}_{tag}_{nchannel}channel"
        
        try:
            # Check if model already exists and we don't want to force retrain
            model_exists = check_model_exists(model_dir, tag, version)
            results_exist = (results_dir / f"{version}_{tag}_oracle.npy").exists()
            
            should_train = TRAIN_MODELS and (FORCE_RETRAIN or not (model_exists and results_exist))
            
            if should_train:
                print(f"Training new model...")
                
                # Create model with custom architecture
                model_cv = LassoNetRegressorCV(
                    device=device, 
                    hidden_dims=hidden,
                    batch_size=LassoNet_batch_size,
                    lambda_start=1e-3, 
                    optim=partial(torch.optim.Adam, lr=lr0, betas=(Adam_beta0, Adam_beta1), eps=Adam_eps),
                    verbose=True
                )
                
                # Run stability selection and save results/model
                stability_results, results_path = run_stability_selection_and_save(
                    model_cv, tag, X_masked, y_masked, total_channel_names, nchannel,
                    train_new=True, save_model=SAVE_MODELS
                )
                
            elif results_exist:
                print(f"Using existing results from: {results_dir}")
                stability_results = None  # We'll load from files if needed
                results_path = results_dir
                
            else:
                print(f"No existing results found {results_dir}/{version}_{tag}_oracle.npy and training disabled. Skipping...")
                continue
            
            # Store experiment info for plotting
            experiment_results.append({
                'tag': tag,
                'results_dir': results_path,
                'nchannel': nchannel,
                'hidden': hidden
            })
            
            # Generate plots if requested
            if GENERATE_PLOTS and results_exist:
                print(f"Generating plots for {tag}...")
                plot_success = generate_plots_from_results(
                    results_path, tag, nchannel, total_channel_names, 
                    Normalized_TrainingData, version
                )
                if plot_success:
                    print(f"Plots generated successfully")
            
            success_count += 1
            print(f"Experiment {i}/{total_count} completed successfully")
            
        except Exception as e:
            print(f"Error in experiment {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Enhanced memory cleanup
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            import gc
            gc.collect()

    print(f"\nAll stability selection experiments completed!")
    print(f"   Successful: {success_count}/{total_count}")
    print(f"   Failed: {total_count - success_count}/{total_count}")
    
    if success_count < total_count:
        print(f"Some experiments failed - check error messages above")
    
    return experiment_results

def batch_generate_plots(experiment_results, total_channel_names, Normalized_TrainingData, version):
    """Generate plots for all experiments in batch (useful for plot-only runs)"""
    print(f"\nGenerating plots for {len(experiment_results)} experiments...")
    
    for i, exp in enumerate(experiment_results, 1):
        print(f"Generating plots {i}/{len(experiment_results)}: {exp['tag']}")
        try:
            generate_plots_from_results(
                exp['results_dir'], exp['tag'], exp['nchannel'], 
                total_channel_names, Normalized_TrainingData, version
            )
        except Exception as e:
            print(f"Error generating plots for {exp['tag']}: {str(e)}")
            import traceback
            traceback.print_exc()

def analyze_saved_model(model_dir, tag, version):
    """Load and analyze a saved model with better error handling"""
    model_path = Path(model_dir) / f"{version}_{tag}_model.pkl"
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None
    
    model = load_trained_model(model_path)
    if model is not None:
        print(f"Model analysis for {tag}:")
        print(f"  Hidden dimensions: {getattr(model, 'hidden_dims', 'Unknown')}")
        
        # Check device safely
        if hasattr(model, 'device'):
            print(f"  Device: {model.device}")
        elif hasattr(model, 'module_') and model.module_ is not None:
            try:
                device = next(model.module_.parameters()).device
                print(f"  Device: {device}")
            except:
                print(f"  Device: CPU (default)")
        else:
            print(f"  Device: CPU (default)")
    
    return model