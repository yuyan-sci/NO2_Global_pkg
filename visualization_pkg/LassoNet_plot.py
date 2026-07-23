import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from visualization_pkg.VIF_plot import _vif_statsmodels

def plot_Stability_Selection(results_dir, plots_dir, model_name, nchannel, total_channel_names, Normalized_TrainingData, version,
                             top_k=35, prob_threshold=0.5):
    os.makedirs(plots_dir, exist_ok=True)

    # ---- Load recordings ----
    order = np.load(os.path.join(results_dir, f'{version}_{model_name}_order.npy'))
    prob  = np.load(os.path.join(results_dir, f'{version}_{model_name}_prob.npy'))  # selection probability
    with open(os.path.join(results_dir, f'{version}_{model_name}_paths.pkl'), 'rb') as f:
        paths = pickle.load(f)
    wrong = np.load(os.path.join(results_dir, f'{version}_{model_name}_wrong.npy'))

    # ---- Path that matches prob length ----
    example_path = None
    for p in paths:
        if len(p) == prob.shape[0]:
            example_path = p
            break
    if example_path is None:
        raise RuntimeError("No path in paths.pkl matches prob.shape[0].")

    # λ and validation loss
    lambda_values = np.array([item.lambda_ for item in example_path])
    val_losses    = np.array([item.val_loss for item in example_path])
    log_lambda    = np.log10(lambda_values + 1e-10)

    # sort by λ for plotting/integration
    sort_idx      = np.argsort(log_lambda)
    log_lambda    = log_lambda[sort_idx]
    val_losses    = val_losses[sort_idx]
    prob          = prob[sort_idx, :]

    # ------------------------------ Helpers ------------------------------------
    def km_label(buf):
        try:
            n = int(buf)
            return f"{n/1000:g} km" if n >= 1000 else f"{n} m"
        except:
            return buf
    def pretty_name(s):
        """Convert feature names to pretty, readable format"""
        s_lower = s.lower()
        
        # 1. Handle buffer patterns - convert to km/m and add road type context
        buffer_match = re.search(r'buffer-(\d+)', s_lower)
        if buffer_match:
            distance = int(buffer_match.group(1))
            dist_str = f"{distance/1000:g} km" if distance >= 1000 else f"{distance} m"
            
            return f"B{dist_str}"
        
        # 2. Handle distance/dist patterns with road type context
        if s_lower.endswith('_distance') or s_lower.endswith('_dist'):
            if 'major_roads_new_dist' in s_lower:
                return "Distance (P,T)"
            elif 'major_roads_dist' in s_lower:
                return "Distance (P,S,T,M)"
            elif 'minor_roads_new_dist' in s_lower:
                return "Distance (T,R)"
            elif 'minor_roads_dist' in s_lower:
                return "Distance (T,R,U)"
            else:
                return "Distance"

        # 3. Handle log patterns with road type context
        if s_lower.startswith('log_'):
            if 'log_major_roads_new' in s_lower:
                return "Log (P,T)"
            elif 'log_major_roads' in s_lower:
                return "Log (P,S,T,M)"
            elif 'log_minor_roads_new' in s_lower:
                return "Log (T,R)"
            elif 'log_minor_roads' in s_lower:
                return "Log (T,R,U)"
            else:
                return "Log"
       
        # 4. Handle density patterns with road type context
        if (s_lower.endswith('_roads') or s_lower.endswith('_new') or s_lower.endswith('_density')) and not s_lower.startswith('log_'):
            if 'major_roads_new' in s_lower:
                return "Density (P,T)"
            elif 'major_roads' in s_lower:
                return "Density (P,S,T,M)"
            elif 'minor_roads_new' in s_lower:
                return "Density (T,R)"
            elif 'minor_roads' in s_lower:
                return "Density (T,R,U)"
            else:
                return "Density"
               
        # Handle specific feature names
        replacements = {
            'lai': 'LAI',
            'ndvi': 'NDVI',
            'grn': 'GRN',
            'isa': 'ISA',
            'no2': 'NO₂',
            'Prectot': 'Precip.',
        }
        
        # Check for exact matches first
        if s_lower in replacements:
            return replacements[s_lower]
        
        # Default: title case with underscores replaced by spaces
        return s.replace('_', ' ').title()
    # def pretty_name(s):
    #     s2 = s.replace('_', ' ')
    #     m = re.search(r'buffer-(\d+)', s2)
    #     if m:
    #         s2 = re.sub(r'buffer-\d+', f"({km_label(m.group(1))})", s2)
    #     return (s2.title()
    #             .replace('No2', 'NO₂')
    #             .replace('Lai','LAI').replace('Ndvi','NDVI')
    #             .replace('Isa','ISA').replace('Prectot','Precip.')
    #             .replace('Grn', 'GRN'))

    def slug(s):  # for file names
        return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')

    # groups (match on lowercase)
    group_specs = {
        "Major Roads":      ['major_roads', 'major_roads_new', 'log_major_roads', 'log_major_roads_new',
                             'major_roads_dist', 'major_roads_new_dist',
                             'major_roads_buffer-500', 'major_roads_buffer-1000', 'major_roads_buffer-1500', 'major_roads_buffer-2000', 'major_roads_buffer-2500', 
                             'major_roads_buffer-3000','major_roads_buffer-3500', 'major_roads_buffer-4000', 'major_roads_buffer-4500', 'major_roads_buffer-5000', 
                             'major_roads_buffer-5500', 'major_roads_buffer-6000', 'major_roads_buffer-6500', 'major_roads_buffer-7000',  'major_roads_buffer-7500', 
                             'major_roads_buffer-8000', 'major_roads_buffer-8500', 'major_roads_buffer-9000', 'major_roads_buffer-9500', 'major_roads_buffer-10000',
                             'major_roads_buffer-10500', 'major_roads_buffer-11000',
                             'major_roads_new_buffer-500', 'major_roads_new_buffer-1000', 'major_roads_new_buffer-1500', 'major_roads_new_buffer-2000', 'major_roads_new_buffer-2500', 
                             'major_roads_new_buffer-3000','major_roads_new_buffer-3500', 'major_roads_new_buffer-4000', 'major_roads_new_buffer-4500', 'major_roads_new_buffer-5000', 
                             'major_roads_new_buffer-5500', 'major_roads_new_buffer-6000', 'major_roads_new_buffer-6500', 'major_roads_new_buffer-7000',  'major_roads_new_buffer-7500', 
                             'major_roads_new_buffer-8000', 'major_roads_new_buffer-8500', 'major_roads_new_buffer-9000', 'major_roads_new_buffer-9500', 'major_roads_new_buffer-10000',
                             'major_roads_new_buffer-10500', 'major_roads_new_buffer-11000'],
        "Minor Roads":      ['minor_roads', 'log_minor_roads', 'minor_roads_new', 'log_minor_roads_new',
                             'minor_roads_dist', 'minor_roads_new_dist',
                             'minor_roads_buffer-500', 'minor_roads_buffer-1000', 'minor_roads_buffer-1500', 'minor_roads_buffer-2000', 'minor_roads_buffer-2500', 
                             'minor_roads_buffer-3000','minor_roads_buffer-3500', 'minor_roads_buffer-4000', 'minor_roads_buffer-4500', 'minor_roads_buffer-5000', 
                             'minor_roads_buffer-5500', 'minor_roads_buffer-6000', 'minor_roads_buffer-6500', 'minor_roads_buffer-7000',  'minor_roads_buffer-7500', 
                             'minor_roads_buffer-8000', 'minor_roads_buffer-8500', 'minor_roads_buffer-9000', 'minor_roads_buffer-9500', 'minor_roads_buffer-10000',
                             'minor_roads_buffer-10500', 'minor_roads_buffer-11000',
                             'minor_roads_new_buffer-500', 'minor_roads_new_buffer-1000', 'minor_roads_new_buffer-1500', 'minor_roads_new_buffer-2000', 'minor_roads_new_buffer-2500', 
                             'minor_roads_new_buffer-3000','minor_roads_new_buffer-3500', 'minor_roads_new_buffer-4000', 'minor_roads_new_buffer-4500', 'minor_roads_new_buffer-5000', 
                             'minor_roads_new_buffer-5500', 'minor_roads_new_buffer-6000', 'minor_roads_new_buffer-6500', 'minor_roads_new_buffer-7000',  'minor_roads_new_buffer-7500', 
                             'minor_roads_new_buffer-8000', 'minor_roads_new_buffer-8500', 'minor_roads_new_buffer-9000', 'minor_roads_new_buffer-9500', 'minor_roads_new_buffer-10000',
                             'minor_roads_new_buffer-10500', 'minor_roads_new_buffer-11000',],
        "Shrublands":       ['shrublands_density','shrublands_distance',
                             'shrublands_buffer-500','shrublands_buffer-1000','shrublands_buffer-1500', 
                             'shrublands_buffer-2000', 'shrublands_buffer-2500','shrublands_buffer-3000', 'shrublands_buffer-3500', 'shrublands_buffer-4000', 
                             'shrublands_buffer-4500','shrublands_buffer-5000', 'shrublands_buffer-5500', 'shrublands_buffer-6000', 'shrublands_buffer-6500', 
                             'shrublands_buffer-7000','shrublands_buffer-7500', 'shrublands_buffer-8000', 'shrublands_buffer-8500', 'shrublands_buffer-9000', 
                             'shrublands_buffer-9500', 'shrublands_buffer-10000', 'shrublands_buffer-10500', 'shrublands_buffer-11000'],
        "Croplands":        ['croplands_density', 'croplands_distance',
                             'croplands_buffer-500','croplands_buffer-1000','croplands_buffer-1500','croplands_buffer-2000',
                             'croplands_buffer-2500','croplands_buffer-3000','croplands_buffer-3500','croplands_buffer-4000','croplands_buffer-4500','croplands_buffer-5000',
                             'croplands_buffer-5500','croplands_buffer-6000','croplands_buffer-6500','croplands_buffer-7000','croplands_buffer-7500','croplands_buffer-8000',
                             'croplands_buffer-8500','croplands_buffer-9000','croplands_buffer-9500','croplands_buffer-10000','croplands_buffer-10500','croplands_buffer-11000'],
        "Forests":          ['lai', 'forests_density','forests_distance',
                             'forests_buffer-500','forests_buffer-1000','forests_buffer-1500','forests_buffer-2000', 'forests_buffer-2500','forests_buffer-3000',
                             'forests_buffer-3500', 'forests_buffer-4000', 'forests_buffer-4500', 'forests_buffer-5000', 'forests_buffer-5500', 'forests_buffer-6000',
                             'forests_buffer-6500', 'forests_buffer-7000', 'forests_buffer-7500', 'forests_buffer-8000', 'forests_buffer-8500', 'forests_buffer-9000',
                             'forests_buffer-9500', 'forests_buffer-10000', 'forests_buffer-10500', 'forests_buffer-11000'],
        "Water Bodies":     ['water_bodies_density','water_bodies_distance','water_bodies_buffer-500','water_bodies_buffer-1000','water_bodies_buffer-1500',
                             'water_bodies_buffer-2000','water_bodies_buffer-2500','water_bodies_buffer-3000','water_bodies_buffer-3500','water_bodies_buffer-4000',
                             'water_bodies_buffer-4500','water_bodies_buffer-5000','water_bodies_buffer-5500','water_bodies_buffer-6000','water_bodies_buffer-6500',
                             'water_bodies_buffer-7000','water_bodies_buffer-7500','water_bodies_buffer-8000','water_bodies_buffer-8500','water_bodies_buffer-9000',
                             'water_bodies_buffer-9500','water_bodies_buffer-10000','water_bodies_buffer-10500','water_bodies_buffer-11000'],
        "Urban Built-up":   ['urban_builtup_lands_density','urban_builtup_lands_distance','urban_builtup_lands_buffer-500','urban_builtup_lands_buffer-1000',
                             'urban_builtup_lands_buffer-1500','urban_builtup_lands_buffer-2000','urban_builtup_lands_buffer-2500','urban_builtup_lands_buffer-3000','urban_builtup_lands_buffer-3500',
                             'urban_builtup_lands_buffer-4000','urban_builtup_lands_buffer-4500','urban_builtup_lands_buffer-5000','urban_builtup_lands_buffer-5500','urban_builtup_lands_buffer-6000',
                             'urban_builtup_lands_buffer-6500','urban_builtup_lands_buffer-7000','urban_builtup_lands_buffer-7500','urban_builtup_lands_buffer-8000','urban_builtup_lands_buffer-8500',
                             'urban_builtup_lands_buffer-9000','urban_builtup_lands_buffer-9500','urban_builtup_lands_buffer-10000','urban_builtup_lands_buffer-10500','urban_builtup_lands_buffer-11000'],
        "Vegatation":       ['grn', 'ndvi'],
    }        
    feat_names_lower = [str(n).lower() for n in total_channel_names]

    # --------------------- 1) Importance via AUC over logλ ---------------------
    auc_scores = np.trapz(prob, x=log_lambda, axis=0)

    # ------------------------ 1A) Per-group lollipop plots ---------------------

    # Improved visualization: Combined subplot layout with top 5 features per group
    def create_combined_lollipop_plots(group_specs, auc_scores, feat_names_lower, total_channel_names, 
                                    plots_dir, version, model_name, top_n=5):
        """
        Create a combined figure with subplots for each feature group, showing top N features per group.
        """
        
        # Filter groups that have data
        valid_groups = {}
        for group, keys in group_specs.items():
            idxs = [i for i, nm in enumerate(feat_names_lower) if any(k in nm for k in keys)]
            if idxs:
                valid_groups[group] = idxs
        
        n_groups = len(valid_groups)
        if n_groups == 0:
            return
        
        # Calculate optimal subplot layout - force 2 rows, 4 columns
        nrows, ncols = 2, 4
        figsize = (11, 8)  # Even wider to accommodate rotated labels
        
        fig = plt.figure(figsize=figsize, dpi=300)
        gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.6)  # Increased horizontal spacing
        
        accent = '#264653'  # teal-charcoal
        colors = [accent] * 8
        
        for idx, (group, idxs) in enumerate(valid_groups.items()):
            row = idx // ncols
            col = idx % ncols
            ax = fig.add_subplot(gs[row, col])
            
            # Get top N features for this group
            auc_g = auc_scores[idxs]
            names_g = [pretty_name(total_channel_names[i]) for i in idxs]
            
            # Get top N features
            top_indices = np.argsort(auc_g)[-top_n:]  # Get top N indices
            if len(top_indices) < len(auc_g):
                auc_g = auc_g[top_indices]
                names_g = [names_g[i] for i in top_indices]
            
            # Sort by importance (ascending for better visual flow)
            order_g = np.argsort(auc_g)
            y = np.arange(len(auc_g))
            
            # Plot with group-specific color
            color = colors[idx]
            ax.hlines(y, 0, auc_g[order_g], linewidth=2.5, alpha=0.9, color=accent)
            ax.scatter(auc_g[order_g], y, s=40, color=accent, alpha=1.0, edgecolors='white', linewidth=0.6)
            
            # Formatting - rotate y-axis labels to 60 degrees with right edge at tick center
            ax.set_ylim(-0.5, len(auc_g) - 0.5)
            ax.set_yticks(y)
            ax.set_yticklabels([names_g[i] for i in order_g], fontsize=12, rotation=60, ha='right', va='top')
            ax.set_xlabel('AUC Importance', fontsize=14)
            ax.set_title(f'{group} (Top {min(top_n, len(idxs))})', fontsize=14, pad=8)
            
            # Grid
            # ax.grid(axis='x', which='major', linestyle=':', alpha=0.6)
            # ax.grid(axis='x', which='minor', linestyle=':', alpha=0.3)
            
            # Auto x-limits with tighter bounds - reduce empty space beyond 12.7
            gmin, gmax = float(np.nanmin(auc_g)), float(np.nanmax(auc_g))
            
            x_range = gmax - gmin
            
            # Add 25% padding on the max side (for top feature bars), 5% on min side
            padding_max = x_range * 0.25
            padding_min = x_range * 0.1
            
            xmin = gmin - padding_min
            xmax = gmax + padding_max
            ax.set_xlim(xmin, xmax)
                
            # Adjust tick spacing based on x-axis range for cleaner appearance
            x_range = xmax - xmin
            if x_range <= 0.1:
                ax.xaxis.set_major_locator(MultipleLocator(0.02))
                ax.xaxis.set_minor_locator(MultipleLocator(0.02))
            elif x_range > 0.1 and x_range <= 1.0:
                ax.xaxis.set_major_locator(MultipleLocator(0.05))
                ax.xaxis.set_minor_locator(MultipleLocator(0.05))
            elif x_range > 1.0 and r_range <= 2.0:
                ax.xaxis.set_major_locator(MultipleLocator(0.1))
                ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            else:
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # Remove the complex rotation logic - keep labels consistent
            ax.tick_params(axis='y', pad=2)  # Reduce padding between ticks and labels
        
        # Remove empty subplots
        for idx in range(n_groups, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            fig.add_subplot(gs[row, col]).axis('off')
        
        # Save
        plt.savefig(os.path.join(plots_dir, f'{version}_{model_name}_combined_group_lollipops_top{top_n}.png'),
                    bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)

    # Alternative: Horizontal bar chart version for better readability
    # def create_horizontal_bar_plots(group_specs, auc_scores, feat_names_lower, total_channel_names, 
    #                             plots_dir, version, model_name, top_n=5):
    #     """
    #     Create horizontal bar charts - often more readable than lollipop plots.
    #     """
        
    #     # Filter valid groups
    #     valid_groups = {}
    #     for group, keys in group_specs.items():
    #         idxs = [i for i, nm in enumerate(feat_names_lower) if any(k in nm for k in keys)]
    #         if idxs:
    #             valid_groups[group] = idxs
        
    #     n_groups = len(valid_groups)
    #     if n_groups == 0:
    #         return
        
    #     # Layout calculation - force 2 rows, 4 columns
    #     nrows, ncols = 2, 4
    #     figsize = (16, 8)  # Narrower width per plot
        
    #     fig = plt.figure(figsize=figsize, dpi=300)
    #     gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.35, wspace=0.25)
        
    #     colors = ['black', 'black', 'black', 'black',
    #               'black', 'black', 'black', 'black']
        
    #     for idx, (group, idxs) in enumerate(valid_groups.items()):
    #         row = idx // ncols
    #         col = idx % ncols
    #         ax = fig.add_subplot(gs[row, col])
            
    #         # Get data
    #         auc_g = auc_scores[idxs]
    #         names_g = [pretty_name(total_channel_names[i]) for i in idxs]
            
    #         # Get top N features
    #         top_indices = np.argsort(auc_g)[-top_n:]
    #         if len(top_indices) < len(auc_g):
    #             auc_g = auc_g[top_indices]
    #             names_g = [names_g[i] for i in top_indices]
            
    #         # Sort by importance (ascending)
    #         order_g = np.argsort(auc_g)
    #         y_pos = np.arange(len(auc_g))
            
    #         # Create horizontal bars
    #         color = colors[idx % len(colors)]
    #         bars = ax.barh(y_pos, auc_g[order_g], color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
            
    #         # Add value labels on bars
    #         for i, (bar, val) in enumerate(zip(bars, auc_g[order_g])):
    #             ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
    #                 va='center', ha='left', fontsize=9, fontweight='bold')
            
    #         # Formatting
    #         ax.set_yticks(y_pos)
    #         ax.set_yticklabels([names_g[i] for i in order_g], fontsize=10)
    #         ax.set_xlabel('AUC Importance', fontsize=11)
    #         ax.set_title(f'{group} (Top {min(top_n, len(idxs))})', fontsize=12, pad=10, fontweight='bold')
            
    #         # Grid and styling
    #         ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    #         ax.set_axisbelow(True)
            
    #         # Auto x-limits with tighter bounds - reduce empty space beyond 12.7
    #         gmax = float(np.nanmax(auc_g))
    #         xlim_max = min(12.7, gmax * 1.1)  # 10% padding, but max 12.7
    #         ax.set_xlim(0, xlim_max)
            
    #         # Remove top and right spines
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
        
    #     # Remove empty subplots
    #     for idx in range(n_groups, nrows * ncols):
    #         row = idx // ncols
    #         col = idx % ncols
    #         fig.add_subplot(gs[row, col]).axis('off')
        
    #     # Overall title and subtitle
    #     fig.suptitle(f'Top {top_n} Most Important Features by Category', fontsize=18, y=0.95, fontweight='bold')
    #     fig.text(0.5, 0.92, f'Model: {model_name}', ha='center', fontsize=12, style='italic')
        
    #     # Save
    #     plt.savefig(os.path.join(plots_dir, f'{version}_{model_name}_horizontal_bars_top{top_n}.png'),
    #                 bbox_inches='tight', pad_inches=0.1, dpi=300)
    #     plt.close(fig)

    # # Usage - replace your existing loop with:
    # # Combined lollipop plots
    create_combined_lollipop_plots(group_specs, auc_scores, feat_names_lower, total_channel_names, 
                                plots_dir, version, model_name, top_n=5)

    # # Horizontal bar charts (alternative visualization)
    # create_horizontal_bar_plots(group_specs, auc_scores, feat_names_lower, total_channel_names, 
    #                         plots_dir, version, model_name, top_n=5)
    
    # # ------------------------- 2) Heatmap: leaders + outside -------------------    
    # group_to_idxs = {}
    # for group, keys in group_specs.items():
    #     idxs = [i for i, nm in enumerate(feat_names_lower) if any(k in nm for k in keys)]
    #     if idxs:
    #         group_to_idxs[group] = np.array(idxs, dtype=int)

    # leaders, leader_groups = [], []
    # for group, idxs in group_to_idxs.items():
    #     best = idxs[np.argmax(auc_scores[idxs])]
    #     leaders.append(best); leader_groups.append(group)
    # leaders = np.asarray(leaders, dtype=int)

    # all_grouped = np.unique(np.concatenate(list(group_to_idxs.values()))) if group_to_idxs else np.array([], dtype=int)
    # all_idx = np.arange(len(total_channel_names))
    # outside_idx = np.setdiff1d(all_idx, all_grouped, assume_unique=False)

    # leader_order = np.argsort(auc_scores[leaders])[::-1] if leaders.size else np.array([], dtype=int)
    # leaders_sorted = leaders[leader_order]
    # leader_groups_sorted = [leader_groups[i] for i in leader_order]
    # outside_sorted = outside_idx[np.argsort(auc_scores[outside_idx])[::-1]] if outside_idx.size else outside_idx

    # # visible λ-range (keep your ≥1)
    # mask_vis = log_lambda >= 1
    # logx     = log_lambda[mask_vis]
    # prob_vis = prob[mask_vis, :]

    # # label overrides (outside features only)
    # label_map = {
    #     'GeoNO2': r"Geophysical NO$_2$",
    #     'GCHP_NO2': r"GCHP NO$_2$",
    #     'GCHP_O3': r"GCHP O$_3$",
    #     'GCHP_OH': r'GCHP OH',
    #     'GCHP_NO': r"GCHP NO",
    #     'GCHP_NH3': r"GCHP NH$_3$",
    #     'GCHP_HO2': r"GCHP HO$_2$",
    #     'GCHP_H2O2': r"GCHP H$_2$O$_2$",
    #     'GCHP_NO3': r"GCHP NO$_3$",
    #     'GCHP_N2O5': r"GCHP N$_2$O$_5$",
    #     'GCHP_CO': r"GCHP CO",
    #     'NO_emi': r"NO Emi",
    #     'Total_DM': 'Dry Matter Emi'
    # }
    # def make_label(name: str) -> str:
    #     return label_map.get(name, name)

    # heat_idx   = np.concatenate([leaders_sorted, outside_sorted]) if leaders.size else outside_sorted
    # row_labels = leader_groups_sorted + [make_label(total_channel_names[i]) for i in outside_sorted]

    # heat = prob_vis[:, heat_idx].T
    # auc_scores_plot = np.trapz(prob_vis, x=logx, axis=0)
    # # keep least-at-top / most-at-bottom (change to [::-1] if you want brightest at top)
    # order_rows = np.argsort(auc_scores_plot[heat_idx])
    # heat = heat[order_rows, :]
    # row_labels = [row_labels[i] for i in order_rows]

    # # --- AUTO LAYOUT based on number of rows & label length ---
    # n_rows = len(row_labels)
    # max_lab_len = max((len(s) for s in row_labels), default=10)

    # HM_FIG_W   = 12.5
    # HM_DPI     = 320

    # # font size: ~11pt for <=35 rows, shrink gradually beyond that, floor at 8pt
    # tick_fs  = int(max(8, min(11, 11 * 35 / max(n_rows, 35))))
    # label_fs = tick_fs
    # title_fs = tick_fs + 2
    # cbar_fs  = max(8, tick_fs - 1)

    # # row height chosen to keep ~1 line per tick w/ a little headroom
    # HM_ROW_H  = max(0.38, min(0.55, (tick_fs / 72) / 0.85))   # 72 pt = 1 inch
    # fig_h     = max(6.0, HM_ROW_H * n_rows)

    # # width ratios: widen text col a bit if labels are long
    # HM_TEXT_W = min(2.4, 1.6 + 0.02 * min(max_lab_len, 40))   # 1.6–2.4
    # HM_HEAT_W = 1.6                                           # more plot space
    # HM_CBAR_W = 0.18

    # fig = plt.figure(figsize=(HM_FIG_W, fig_h), dpi=HM_DPI, constrained_layout=False)
    # gs  = fig.add_gridspec(nrows=1, ncols=3,
    #                     width_ratios=[HM_TEXT_W, HM_HEAT_W, HM_CBAR_W],
    #                     wspace=0.04)

    # ax_lab = fig.add_subplot(gs[0, 0])   # label column
    # ax_hm  = fig.add_subplot(gs[0, 1])   # heatmap
    # cax    = fig.add_subplot(gs[0, 2])   # colorbar

    # # --- heatmap (use your "heat", "logx" computed earlier) ---
    # im = ax_hm.imshow(
    #     heat,
    #     aspect='auto',
    #     interpolation='nearest',
    #     extent=[logx.min(), logx.max(), -0.5, n_rows - 0.5],
    #     vmin=0, vmax=1,
    #     cmap='viridis'  # or 'viridis_r' if you want reversed
    # )
    # ax_hm.set_yticks([])
    # ax_hm.tick_params(axis='x', labelsize=tick_fs-1)
    # ax_hm.set_xlabel('log10(λ)', fontsize=label_fs)
    # ax_hm.set_title('Selection Probability vs log10(λ)', fontsize=title_fs)

    # # best-λ line if visible
    # best_i = int(np.argmin(val_losses))
    # best_loglam = float(log_lambda[best_i])
    # if best_loglam >= logx.min() and best_loglam <= logx.max():
    #     ax_hm.axvline(best_loglam, linestyle='--', linewidth=1.5, alpha=0.85)

    # # --- labels (left axis): one line per row; no overlap ---
    # ax_lab.set_xlim(0, 1)
    # ax_lab.set_ylim(-0.5, n_rows - 0.5)
    # ax_lab.invert_yaxis()
    # ax_lab.axis('off')
    # for i, lab in enumerate(row_labels):
    #     ax_lab.text(0.98, i, lab, ha='right', va='center', fontsize=tick_fs, clip_on=False)

    # # --- colorbar (a bit wider; pad label so it doesn’t clip) ---
    # cb = plt.colorbar(im, cax=cax)
    # cb.ax.tick_params(labelsize=cbar_fs)
    # cb.set_label('Selection Probability', fontsize=label_fs, labelpad=10)

    # fig.savefig(os.path.join(plots_dir, f'{version}_{model_name}_leaders_plus_OUTSIDE_heatmap_loglam_ge0.png'),
    #             bbox_inches='tight', pad_inches=0.05)
    # plt.close(fig)

    # # Define features to exclude from VIF analysis only (not initial heatmap)
    # EXCLUDE_FEATURES = {'lat', 'lon', 'GCHP_CO', 'GCHP_H2O2', 'GCHP_HO2', 
    #                 'GCHP_N2O5', 'GCHP_NH3', 'GCHP_NO', 'GCHP_NO3'}

    # def filter_excluded_features(indices, feature_names):
    #     """Filter out excluded features from indices array"""
    #     filtered_indices = []
    #     for idx in indices:
    #         if feature_names[idx] not in EXCLUDE_FEATURES:
    #             filtered_indices.append(idx)
    #     return np.array(filtered_indices, dtype=int)

    # # ------------------- NOW START VIF ANALYSIS WITH FILTERED FEATURES -------------------
    # print(f"\nNow filtering features for VIF analysis...")
    # print(f"Excluded features: {EXCLUDE_FEATURES}")

    # # Filter the heat_idx for VIF analysis - APPLY FILTERING HERE
    # heat_idx_filtered = filter_excluded_features(heat_idx, total_channel_names)
    # print(f"FILTERED - Features remaining for VIF: {len(heat_idx_filtered)} (removed {len(heat_idx) - len(heat_idx_filtered)})")

    # vif_threshold = 5
    # print('Normalized_TrainingData shape:', Normalized_TrainingData.shape)

    # # 1) Samples × features (center pixel) and drop rows with any NaNs across features
    # X_center = Normalized_TrainingData[:, :, 2, 2]           # (n_samples, n_features)
    # row_mask = ~np.any(np.isnan(X_center), axis=1)
    # X_full   = X_center[row_mask, :]
    # print('X_full shape:', X_full.shape)

    # # 2) VIFs for the FILTERED features
    # X_sel_filtered    = X_full[:, heat_idx_filtered]                    # FILTERED features
    # vif_vals_filtered = _vif_statsmodels(X_sel_filtered)               # in heat_idx_filtered order

    # # Create labels for filtered features
    # labels_filtered = []
    # for idx in heat_idx_filtered:
    #     # Check if this was a leader (group name) or outside feature (feature name)
    #     if idx in leaders_sorted:
    #         group_idx = np.where(leaders_sorted == idx)[0][0]
    #         labels_filtered.append(leader_groups_sorted[group_idx])
    #     else:
    #         labels_filtered.append(make_label(total_channel_names[idx]))

    # # 3) Sort by AUC for display order (filtered features)
    # auc_scores_filtered = auc_scores[heat_idx_filtered]
    # order_rows_filtered = np.argsort(auc_scores_filtered)  # least to most
    # labels_display_filtered = [labels_filtered[i] for i in order_rows_filtered]
    # vif_display_filtered = vif_vals_filtered[order_rows_filtered]
    # sel_indices_display_filtered = heat_idx_filtered[order_rows_filtered]

    # # 4) Keep/drop mask (display order) for the VIF-filtered heatmap
    # keep_mask = np.nan_to_num(vif_display_filtered, nan=np.inf) <= float(vif_threshold)

    # # 5) For plotting/CSV, sort **by VIF value** (ascending)
    # vif_for_sort = np.nan_to_num(vif_display_filtered, nan=np.inf)
    # order_by_vif = np.argsort(vif_for_sort)

    # labels_vif   = [labels_display_filtered[i] for i in order_by_vif]
    # featidx_vif  = sel_indices_display_filtered[order_by_vif]
    # vif_sorted   = vif_display_filtered[order_by_vif]

    # # 6) Save CSV (VIF-sorted order) - note this is FILTERED features only
    # lines = [f"row_by_vif,label,feature_name,feature_index,vif,keep(vif<={float(vif_threshold)}),excluded_from_vif_analysis"]
    # for i, (lab, fidx, v) in enumerate(zip(labels_vif, featidx_vif, vif_sorted), start=1):
    #     v_str = "" if np.isnan(v) else f"{float(v):.6f}"
    #     keep  = int((not np.isnan(v)) and (float(v) <= float(vif_threshold)))
    #     excluded = int(total_channel_names[int(fidx)] in EXCLUDE_FEATURES)
    #     lines.append(f"{i},{lab},{total_channel_names[int(fidx)]},{int(fidx)},{v_str},{keep},{excluded}")

    # with open(os.path.join(plots_dir, f'{version}_{model_name}_leaders_plus_OUTSIDE_filtered_VIF.csv'), 'w') as f:
    #     f.write("\n".join(lines))

    # print(f"VIF analysis complete. Features kept after VIF <= {vif_threshold}: {np.sum(keep_mask)}/{len(keep_mask)}")

    # # 7) VIF bar chart (VIF-sorted order) - using FILTERED features
    # VIF_FIG_W = HM_FIG_W; VIF_DPI = HM_DPI
    # vif_fig_h = max(6.0, HM_ROW_H * len(labels_vif))  # Adjust height for filtered features

    # fig = plt.figure(figsize=(VIF_FIG_W, vif_fig_h), dpi=VIF_DPI, constrained_layout=False)
    # gs  = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[HM_TEXT_W, HM_HEAT_W + HM_CBAR_W], wspace=0.06)
    # ax_lab = fig.add_subplot(gs[0, 0]); ax_bar = fig.add_subplot(gs[0, 1])

    # y = np.arange(len(labels_vif))
    # x_vals = np.nan_to_num(vif_sorted, nan=0.0)
    # xmax = float(min((x_vals.max() if x_vals.size else 0.0) * 1.05, 50.0))

    # ax_bar.barh(y, x_vals, height=0.8)
    # for thr in (5, 10, float(vif_threshold)):
    #     if thr <= max(5.0, xmax):
    #         ax_bar.axvline(thr, linestyle='--', linewidth=1.2, alpha=0.7)

    # ax_bar.set_ylim(-0.5, len(labels_vif) - 0.5)
    # ax_bar.set_yticks([])
    # ax_bar.set_xlabel('VIF', fontsize=label_fs)
    # ax_bar.set_xlim(0, max(5.0, xmax))
    # ax_bar.grid(axis='x', linestyle=':', alpha=0.4)
    # ax_bar.set_title('VIF for Group Leaders + Outside Features\n(Excluding Geographic & Selected Chemical Features)', fontsize=title_fs)

    # ax_lab.set_xlim(0, 1)
    # ax_lab.set_ylim(-0.5, len(labels_vif) - 0.5)
    # ax_lab.invert_yaxis()
    # ax_lab.axis('off')
    # for i, lab in enumerate(labels_vif[::-1]):
    #     ax_lab.text(0.98, i, lab, ha='right', va='center', fontsize=tick_fs, clip_on=False)

    # fig.savefig(os.path.join(plots_dir, f'{version}_{model_name}_leaders_plus_OUTSIDE_filtered_VIF.png'),
    #             bbox_inches='tight', pad_inches=0.05)
    # plt.close(fig)

    # # 8) Final filtered heatmap (VIF <= threshold, using filtered features)
    # if np.any(keep_mask):
    #     # Create heatmap data using filtered features and VIF threshold
    #     heat_f = prob_vis[:, heat_idx_filtered].T  # Use filtered features
    #     heat_f = heat_f[order_rows_filtered, :]     # Apply AUC-based ordering
    #     heat_f = heat_f[keep_mask, :]               # Apply VIF filtering
        
    #     labels_f = [labels_display_filtered[i] for i in np.where(keep_mask)[0]]
    #     n_rows_f = len(labels_f)

    #     # Adjust layout for final filtered heatmap
    #     tick_fs_f  = int(max(8, min(11, 11 * 35 / max(n_rows_f, 35))))
    #     label_fs_f = tick_fs_f
    #     title_fs_f = tick_fs_f + 2
    #     HM_ROW_H_f = max(0.38, min(0.55, (tick_fs_f / 72) / 0.85))
    #     fig_h_f    = max(6.0, HM_ROW_H_f * n_rows_f)

    #     fig2 = plt.figure(figsize=(HM_FIG_W, fig_h_f), dpi=HM_DPI, constrained_layout=False)
    #     gs2  = fig2.add_gridspec(nrows=1, ncols=3, width_ratios=[HM_TEXT_W, HM_HEAT_W, HM_CBAR_W], wspace=0.04)
    #     ax_lab2 = fig2.add_subplot(gs2[0, 0])
    #     ax_hm2 = fig2.add_subplot(gs2[0, 1])
    #     cax2 = fig2.add_subplot(gs2[0, 2])

    #     im2 = ax_hm2.imshow(heat_f, aspect='auto', interpolation='nearest',
    #                         extent=[logx.min(), logx.max(), -0.5, n_rows_f - 0.5],
    #                         vmin=0, vmax=1, cmap='viridis')
    #     ax_hm2.set_yticks([])
    #     ax_hm2.tick_params(axis='x', labelsize=tick_fs_f-1)
    #     ax_hm2.set_xlabel('log10(λ)', fontsize=label_fs_f)
    #     ax_hm2.set_title(f'Selection Probability vs log10(λ)\n(Filtered Features, VIF ≤ {vif_threshold})', fontsize=title_fs_f)

    #     # Add best lambda line if visible
    #     if best_loglam >= logx.min() and best_loglam <= logx.max():
    #         ax_hm2.axvline(best_loglam, linestyle='--', linewidth=1.5, alpha=0.85)

    #     # Labels
    #     ax_lab2.set_xlim(0, 1)
    #     ax_lab2.set_ylim(-0.5, n_rows_f - 0.5)
    #     ax_lab2.invert_yaxis()
    #     ax_lab2.axis('off')
    #     for i, lab in enumerate(labels_f):
    #         ax_lab2.text(0.98, i, lab, ha='right', va='center', fontsize=tick_fs_f, clip_on=False)

    #     # Colorbar
    #     cb2 = plt.colorbar(im2, cax=cax2)
    #     cb2.ax.tick_params(labelsize=max(8, tick_fs_f - 1))
    #     cb2.set_label('Selection Probability', fontsize=label_fs_f, labelpad=10)

    #     fig2.savefig(os.path.join(plots_dir, f'{version}_{model_name}_leaders_plus_OUTSIDE_filtered_heatmap_vif_le{int(vif_threshold)}.png'),
    #                 bbox_inches='tight', pad_inches=0.05)
    #     plt.close(fig2)
        
    #     print(f"Final filtered heatmap saved with {n_rows_f} features (VIF ≤ {vif_threshold})")
    # else:
    #     print(f"[VIF] All filtered features exceeded threshold ({vif_threshold}); no final filtered heatmap produced.")

    # print("\nHeatmap generation complete:")
    # print(f"1. All features heatmap: {len(heat_idx)} features")
    # print(f"2. Filtered features for VIF: {len(heat_idx_filtered)} features") 
    # print(f"3. Final VIF-filtered heatmap: {np.sum(keep_mask) if 'keep_mask' in locals() else 0} features")
    # print(f"4. Excluded from VIF analysis: {EXCLUDE_FEATURES}")


    # # ------------- 3) Validation loss + selected count (larger text, tighter plot) --------------
    # VL_FIG_W, VL_FIG_H, VL_DPI = 8.0, 4.6, 320
    # fig, ax1 = plt.subplots(figsize=(VL_FIG_W, VL_FIG_H), dpi=VL_DPI)
    # ax1.plot(log_lambda, val_losses, linewidth=2)
    # ax1.set_xlabel('log10(λ)', fontsize=12)
    # ax1.set_ylabel('Validation Loss', fontsize=12)
    # ax1.tick_params(axis='both', labelsize=11)
    # ax1.set_title(f'Validation Loss and Selected-Feature Count vs log10(λ)\n(threshold={prob_threshold:g})',
    #               fontsize=13)

    # selected_count = (prob >= prob_threshold).sum(axis=1)
    # ax2 = ax1.twinx()
    # ax2.plot(log_lambda, selected_count, '--', linewidth=2)
    # ax2.set_ylabel('Selected features (count)', fontsize=12)
    # ax2.tick_params(axis='both', labelsize=11)

    # ax1.axvline(best_loglam, linestyle=':', linewidth=1.5)
    # ax1.grid(axis='x', linestyle=':', alpha=0.4)

    # plt.tight_layout()
    # plt.savefig(os.path.join(plots_dir, f'{version}_{model_name}_val_loss_and_count.png'), dpi=VL_DPI)
    # plt.close(fig)

    # # --------------------- 4) Error vs number of selected features --------------
    # ER_FIG_W, ER_FIG_H, ER_DPI = 7.2, 4.2, 320
    # fig = plt.figure(figsize=(ER_FIG_W, ER_FIG_H), dpi=ER_DPI)
    # plt.plot(wrong)
    # plt.xlabel("Number of selected features", fontsize=12)
    # plt.ylabel("Error", fontsize=12)
    # plt.title("Error as a function of selected features", fontsize=13)
    # plt.tick_params(axis='both', labelsize=11)
    # plt.tight_layout()
    # plt.savefig(os.path.join(plots_dir, f'{version}_{model_name}_error.png'), dpi=ER_DPI)
    # plt.close(fig)

    # # --------------------- 5) Save ranked table -------------------
    # inv_rank_importance = np.zeros(nchannel)
    # for i, idx in enumerate(order):
    #     inv_rank_importance[idx] = nchannel - i

    # ranked_idx = np.argsort(auc_scores)[::-1]
    # out_lines = ["rank,feature,auc,order_inverse_rank,feature_index"]
    # for r, idx in enumerate(ranked_idx, start=1):
    #     out_lines.append(f"{r},{total_channel_names[idx]},{auc_scores[idx]:.6f},{inv_rank_importance[idx]:.6f},{idx}")
    # with open(os.path.join(plots_dir, f'{version}_{model_name}_feature_ranking_auc.csv'), 'w') as f:
    #     f.write("\n".join(out_lines))