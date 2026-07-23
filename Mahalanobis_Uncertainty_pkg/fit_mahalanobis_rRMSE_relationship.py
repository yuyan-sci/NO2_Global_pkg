"""Phase U4: fit the Mahalanobis-distance <-> rRMSE relationship.

Reads the BLISCO obs/final recordings and Mahalanobis-distance recordings
produced by Phases U2/U3, bins the data on log(1+Mahalanobis) for each
season and "All", computes rRMSE per bin, smooths with LOWESS, and saves a
single .npy file of LOWESS curves that downstream Phase U5/U6 uses to
convert Mahalanobis-distance maps into rRMSE maps.

Output file:
  {EST_ROOT}/{species}/{version}/Mahalanobis_Uncertainty/Mahalanobis_distance_LOWESS_values/
    {species}_nearby_sites-{N}{special_name}_Mahalanobis_distance_LOWESS_values.npy

The notebook version was `mahalabobis_distance_uncertainty_test.ipynb`. This
script is a direct translation (no plotting by default), parameterised so it
can be run once per channel scheme and reused for every target year.

Channel scheme note: BLISCO recordings are loaded by (version, typeName,
nchannel, special_name). The 26ch (2019-2023) and 27ch (2005-2018) runs live
in separate caches. Use `--year-range` to combine multiple years' BLISCO
recordings that share a channel scheme.
"""

import argparse
import os

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

# Sibling-package import when invoked from Mahalanobis_Uncertainty/
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_func.iostream import load_mahalanobis_distance_data, load_BLISCO_data
from data_func.utils import Get_typeName


EST_ROOT = '/path/to/NO2_DL_global/Training_Evaluation_Estimation/'

WINTER_MONTHS = ['Jan', 'Feb', 'Dec']
SPRING_MONTHS = ['Mar', 'Apr', 'May']
SUMMER_MONTHS = ['Jun', 'Jul', 'Aug']
AUTUMN_MONTHS = ['Sep', 'Oct', 'Nov']
ALL_MONTHS    = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# ---------------------------------------------------------------------------
# Concat helpers (verbatim logic from the notebook, stripped of prints)
# ---------------------------------------------------------------------------

def _concat_for_months(MONTHS_list, buffer_radius_list, startyear, endyear,
                       BLISCO_obs, BLISCO_final,
                       MD_eme, MD_emay, MD_aamay):
    years = [str(y) for y in range(startyear, endyear + 1)]
    obs_parts, fin_parts = [], []
    eme_parts, emay_parts, aamay_parts = [], [], []

    for iradius in buffer_radius_list:
        for y in years:
            obs_m  = [BLISCO_obs[iradius][y][m]   for m in MONTHS_list]
            fin_m  = [BLISCO_final[iradius][y][m] for m in MONTHS_list]
            eme_m  = [MD_eme[iradius][y][m]   for m in MONTHS_list]
            emay_m = [MD_emay[iradius][y][m]  for m in MONTHS_list]
            aam_m  = [MD_aamay[iradius][y][m] for m in MONTHS_list]

            obs_parts.append(obs_m)
            fin_parts.append(fin_m)
            eme_parts.append(np.concatenate(eme_m, axis=0))
            emay_parts.append(np.concatenate(emay_m, axis=0))
            aamay_parts.append(np.concatenate(aam_m, axis=0))

    obs   = np.concatenate([np.concatenate(p, axis=0) for p in obs_parts], axis=0)
    fin   = np.concatenate([np.concatenate(p, axis=0) for p in fin_parts], axis=0)
    eme   = np.concatenate(eme_parts,   axis=0)
    emay  = np.concatenate(emay_parts,  axis=0)
    aamay = np.concatenate(aamay_parts, axis=0)

    obs   = obs[~np.isnan(obs)]
    fin   = fin[~np.isnan(fin)]
    eme   = eme[~np.isnan(eme)]
    emay  = emay[~np.isnan(emay)]
    aamay = aamay[~np.isnan(aamay)]

    return obs, fin, eme, emay, aamay


def _concat_annual(buffer_radius_list, startyear, endyear,
                   BLISCO_obs, BLISCO_final,
                   MD_eme, MD_emay, MD_aamay):
    """Annual variant: month-average per site, then concat."""
    years = [str(y) for y in range(startyear, endyear + 1)]
    obs_parts, fin_parts = [], []
    eme_parts, emay_parts, aamay_parts = [], [], []

    for iradius in buffer_radius_list:
        for y in years:
            obs_arr = np.stack([BLISCO_obs[iradius][y][m]   for m in ALL_MONTHS], axis=1)
            fin_arr = np.stack([BLISCO_final[iradius][y][m] for m in ALL_MONTHS], axis=1)
            eme_arr = np.stack([MD_eme[iradius][y][m]   for m in ALL_MONTHS], axis=1)
            emay_arr = np.stack([MD_emay[iradius][y][m]  for m in ALL_MONTHS], axis=1)
            aamay_arr = np.stack([MD_aamay[iradius][y][m] for m in ALL_MONTHS], axis=1)

            obs_parts.append(np.nanmean(obs_arr, axis=1))
            fin_parts.append(np.nanmean(fin_arr, axis=1))
            eme_parts.append(np.nanmean(eme_arr, axis=1))
            emay_parts.append(np.nanmean(emay_arr, axis=1))
            aamay_parts.append(np.nanmean(aamay_arr, axis=1))

    obs   = np.concatenate(obs_parts,   axis=0)
    fin   = np.concatenate(fin_parts,   axis=0)
    eme   = np.concatenate(eme_parts,   axis=0)
    emay  = np.concatenate(emay_parts,  axis=0)
    aamay = np.concatenate(aamay_parts, axis=0)

    obs   = obs[~np.isnan(obs)]
    fin   = fin[~np.isnan(fin)]
    eme   = eme[~np.isnan(eme)]
    emay  = emay[~np.isnan(emay)]
    aamay = aamay[~np.isnan(aamay)]

    return obs, fin, eme, emay, aamay


# ---------------------------------------------------------------------------
# Bin + rRMSE + LOWESS
# ---------------------------------------------------------------------------

def _rRMSE_in_bin(obs, fin, x, lo, hi, low_pct, high_pct):
    idx = np.where((x >= lo) & (x < hi))[0]
    if len(idx) == 0:
        return np.nan
    obs_b = obs[idx]
    fin_b = fin[idx]
    resid = (fin_b - obs_b) / (obs_b + 1e-4)
    valid = ~np.isnan(obs_b) & ~np.isnan(fin_b)
    if valid.sum() == 0:
        return np.nan
    p_lo = np.percentile(resid[valid], low_pct)
    p_hi = np.percentile(resid[valid], high_pct)
    valid &= (resid >= p_lo) & (resid <= p_hi)
    obs_b = obs_b[valid]
    fin_b = fin_b[valid]
    if len(fin_b) == 0:
        return np.nan
    rmse = np.sqrt(np.mean((fin_b - obs_b) ** 2))
    mean_fin = np.mean(fin_b)
    return rmse / mean_fin if mean_fin != 0 else np.nan


def _season_curve(obs, fin, aamay, bins, low_pct, high_pct):
    x = np.log(aamay + 1.0)
    centers, rrmse = [], []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        centers.append(0.5 * (lo + hi))
        rrmse.append(_rRMSE_in_bin(obs, fin, x, lo, hi, low_pct, high_pct))
    return np.asarray(centers), np.asarray(rrmse)


def compute_and_save(startyear, endyear, buffer_radius_list,
                     species, version, typeName, nchannel, special_name,
                     width, height, nearby_sites_number,
                     start_bin, end_bin, number_bins,
                     low_percentile, high_percentile, lowess_frac,
                     plot_path=None):

    print(f'[U4] Loading BLISCO + Mahalanobis recordings '
          f'({startyear}-{endyear}, nchannel={nchannel}, special_name={special_name!r})')
    mdist = load_mahalanobis_distance_data(species, version, typeName,
                                           startyear, endyear, nchannel,
                                           special_name, width, height,
                                           nearby_sites_number)
    blisco = load_BLISCO_data(species, version, typeName,
                              startyear, endyear, nchannel,
                              special_name, width, height)

    BLISCO_obs   = blisco['BLISCO_obs_data_recording']
    BLISCO_final = blisco['BLISCO_final_data_recording']
    MD_eme   = mdist['EachMonth_EachYear_Martix_Mahalanobis_distance_recording']
    MD_emay  = mdist['EachMonth_AllYear_Martix_Mahalanobis_distance_recording']
    MD_aamay = mdist['AllMonth_AllYear_Martix_Mahalanobis_distance_recording']

    print('[U4] Concatenating by season...')
    seasons = {
        'WINTER': WINTER_MONTHS,
        'SPRING': SPRING_MONTHS,
        'SUMMER': SUMMER_MONTHS,
        'AUTUMN': AUTUMN_MONTHS,
    }
    season_data = {}
    for name, months in seasons.items():
        obs, fin, _eme, _emay, aam = _concat_for_months(
            months, buffer_radius_list, startyear, endyear,
            BLISCO_obs, BLISCO_final, MD_eme, MD_emay, MD_aamay,
        )
        season_data[name] = (obs, fin, aam)

    all_obs, all_fin, _eme, _emay, all_aam = _concat_annual(
        buffer_radius_list, startyear, endyear,
        BLISCO_obs, BLISCO_final, MD_eme, MD_emay, MD_aamay,
    )

    # Shared bin edges on log(1+MD) scale
    bins = start_bin + (end_bin - start_bin) * np.linspace(0, 1, number_bins + 1)
    print(f'[U4] bins edges: start={start_bin} end={end_bin} n={number_bins}')

    centers_ref, rrmse_by_season = None, {}
    for name, (obs, fin, aam) in season_data.items():
        centers, rrmse = _season_curve(obs, fin, aam, bins, low_percentile, high_percentile)
        centers_ref = centers
        rrmse_by_season[name] = rrmse

    centers_all, rrmse_all = _season_curve(all_obs, all_fin, all_aam, bins,
                                           low_percentile, high_percentile)
    assert np.allclose(centers_all, centers_ref)

    # LOWESS smoothing (same-domain, return_sorted=False preserves bin order)
    def _lowess(y):
        return lowess(y, centers_ref, frac=lowess_frac, return_sorted=False)

    WINTER_LOWESS = _lowess(rrmse_by_season['WINTER'])
    SPRING_LOWESS = _lowess(rrmse_by_season['SPRING'])
    SUMMER_LOWESS = _lowess(rrmse_by_season['SUMMER'])
    AUTUMN_LOWESS = _lowess(rrmse_by_season['AUTUMN'])
    ALL_LOWESS    = _lowess(rrmse_all)

    outdir = os.path.join(EST_ROOT, species, version,
                          'Mahalanobis_Uncertainty',
                          'Mahalanobis_distance_LOWESS_values')
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(
        outdir,
        f'{species}_nearby_sites-{nearby_sites_number}_{startyear}-{endyear}_{nchannel}Channel{special_name}_Mahalanobis_distance_LOWESS_values',
    )
    payload = {
        'Mahalanobis_distance_bin_centers': centers_ref,
        'WINTER_LOWESS_values': WINTER_LOWESS,
        'SPRING_LOWESS_values': SPRING_LOWESS,
        'SUMMER_LOWESS_values': SUMMER_LOWESS,
        'AUTUMN_LOWESS_values': AUTUMN_LOWESS,
        'ALL_LOWESS_values':    ALL_LOWESS,
    }
    # Save as .npz (portable, no pickle) for numpy-major-version-independent
    # loading from the container. The legacy .npy dict-dump is no longer
    # written; load_bins_LOWESS_values prefers .npz anyway.
    npz_path = base + '.npz'
    np.savez(npz_path, **{k: np.asarray(v) for k, v in payload.items()})
    print(f'[U4] Saved LOWESS values -> {npz_path}')

    if plot_path:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for label, color, raw, smooth in [
            ('Winter', 'blue',   rrmse_by_season['WINTER'], WINTER_LOWESS),
            ('Spring', 'green',  rrmse_by_season['SPRING'], SPRING_LOWESS),
            ('Summer', 'orange', rrmse_by_season['SUMMER'], SUMMER_LOWESS),
            ('Autumn', 'red',    rrmse_by_season['AUTUMN'], AUTUMN_LOWESS),
            ('Annual', 'purple', rrmse_all,                 ALL_LOWESS),
        ]:
            plt.scatter(centers_ref, raw, marker='o', label=label, c=color)
            plt.plot(centers_ref, smooth, linestyle='--', linewidth=2, color=color)
        plt.xlabel('log(1 + Mahalanobis distance)')
        plt.ylabel('rRMSE')
        plt.title(f'{species} rRMSE vs Mahalanobis distance')
        plt.legend(loc='upper left')
        plt.grid(alpha=0.3)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f'[U4] Saved diagnostic figure -> {plot_path}')

    return npz_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Phase U4: fit Mahalanobis-distance <-> rRMSE LOWESS curves.')
    # Year range. Use a single year for a per-year fit, or a range to pool
    # BLISCO data across years that share a channel scheme.
    p.add_argument('--year',       type=int, default=None,
                   help='Single target year. Overrides --startyear/--endyear if given.')
    p.add_argument('--startyear',  type=int, default=None)
    p.add_argument('--endyear',    type=int, default=None)

    p.add_argument('--version',      type=str, default='v5')
    p.add_argument('--special-name', type=str, default='')
    p.add_argument('--nchannel',     type=int, default=None,
                   help='26 or 27. If omitted, inferred: year<=2018 -> 27, else 26.')
    p.add_argument('--width',        type=int, default=5)
    p.add_argument('--height',       type=int, default=5)
    p.add_argument('--nearby-sites-number', type=int, default=30)

    # Bin configuration (defaults mirror the notebook's SECOND save call,
    # which is the curve file that actually persists on disk).
    p.add_argument('--start-bin',   type=float, default=0.3)
    p.add_argument('--end-bin',     type=float, default=2.0)
    p.add_argument('--number-bins', type=int,   default=30)

    p.add_argument('--low-percentile',  type=float, default=1.0)
    p.add_argument('--high-percentile', type=float, default=99.0)
    p.add_argument('--lowess-frac',     type=float, default=0.5)
    p.add_argument('--plot', action='store_true',
                   help='Save a diagnostic PNG next to the .npy output.')
    return p.parse_args()


def main():
    args = parse_args()

    if args.year is not None:
        startyear = endyear = args.year
    else:
        if args.startyear is None or args.endyear is None:
            raise SystemExit('Provide --year or both --startyear and --endyear.')
        startyear, endyear = args.startyear, args.endyear

    # Channel scheme auto-select (overridable)
    if args.nchannel is None:
        nchannel = 27 if endyear <= 2018 else 26
    else:
        nchannel = args.nchannel

    species = 'NO2'
    typeName = Get_typeName(bias=False, normalize_bias=False,
                            normalize_species=False, absolute_species=True,
                            log_species=False, species=species)

    buffer_radius_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                          110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

    plot_path = None
    if args.plot:
        outdir = os.path.join(EST_ROOT, species, args.version,
                              'Mahalanobis_Uncertainty',
                              'Mahalanobis_distance_LOWESS_values')
        plot_path = os.path.join(outdir,
            f'{species}_nearby_sites-{args.nearby_sites_number}_{startyear}-{endyear}_{nchannel}Channel{args.special_name}_Mahalanobis_distance_LOWESS_curves.png')

    compute_and_save(
        startyear=startyear, endyear=endyear,
        buffer_radius_list=buffer_radius_list,
        species=species, version=args.version, typeName=typeName,
        nchannel=nchannel, special_name=args.special_name,
        width=args.width, height=args.height,
        nearby_sites_number=args.nearby_sites_number,
        start_bin=args.start_bin, end_bin=args.end_bin, number_bins=args.number_bins,
        low_percentile=args.low_percentile, high_percentile=args.high_percentile,
        lowess_frac=args.lowess_frac,
        plot_path=plot_path,
    )


if __name__ == '__main__':
    main()
