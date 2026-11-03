"""Compare current missing-ocean face fluxes against zero-flux diagnostic.
Run from backend: PYTHONPATH="$PWD" python scripts/diagnose_boundary_flux.py
This does NOT modify app/subduction.py.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from app.config import DATA_DIR
from app.dyntopo import load_maximenko
from app.mld import load_mld, maximum_mld, regrid_maximum_mld, regrid_ocean_mask_by_overlap
from app.velocity import reconstruct_geostrophic_velocity
from app.subduction import geostrophic_subduction_rate

SEC_PER_YEAR = 365.25 * 24 * 3600

def stats(name, values, mask):
    x = np.abs(values[mask]) * SEC_PER_YEAR
    print(f'{name}: N={x.size:,}, median={np.median(x):.2f}, p95={np.percentile(x,95):.2f}, '
          f'p98={np.percentile(x,98):.2f}, max={np.max(x):.2f} m/year')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=DATA_DIR / 'Maximenko_dyntopo.nc')
    p.add_argument('--mld-input', type=Path, default=DATA_DIR / 'mld_dr003_ref10m_v2023.nc')
    p.add_argument('--output-dir', type=Path, default=Path('diagnostics/boundary_flux'))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_maximenko(str(args.input))
    mld_source = load_mld(str(args.mld_input))
    maximum = maximum_mld(mld_source.monthly_m)
    ocean = regrid_ocean_mask_by_overlap(mld_source.source_ocean, mld_source.longitude_deg,
        mld_source.latitude_deg, source.longitude_deg, source.latitude_deg)
    mld = regrid_maximum_mld(maximum.maximum_m, maximum.month_count,
        mld_source.longitude_deg, mld_source.latitude_deg,
        source.longitude_deg, source.latitude_deg, ocean)
    velocity = reconstruct_geostrophic_velocity(source.height_m, source.longitude_deg, source.latitude_deg)
    usable = mld.valid & velocity.valid
    common = (mld.maximum_m, velocity.east, velocity.north,
              source.longitude_deg, source.latitude_deg, usable)
    baseline = geostrophic_subduction_rate(*common, ocean)
    # Diagnostic ONLY: treat every invalid cell as impermeable, even if physically ocean.
    closed = geostrophic_subduction_rate(*common, usable)
    a = baseline.rate_m_per_s * SEC_PER_YEAR
    b = closed.rate_m_per_s * SEC_PER_YEAR
    delta = b - a
    stats('Original', baseline.rate_m_per_s, usable)
    stats('All invalid faces closed', closed.rate_m_per_s, usable)
    print(f'Cells changed (>1e-8 m/year): {np.count_nonzero(np.abs(delta[usable]) > 1e-8):,}')
    print(f'Max |change|: {np.nanmax(np.abs(delta)):.2f} m/year')
    print('Original missing-data faces:', baseline.conservation.missing_data_face_count)
    print('Original missing-boundary outward flux (m^3/s):', baseline.conservation.missing_data_outward_m3_per_s)
    print('Original closure residual (m^3/s):', baseline.conservation.closure_residual_m3_per_s)
    print('Closed closure residual (m^3/s):', closed.conservation.closure_residual_m3_per_s)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    limit = np.nanpercentile(np.abs(a[usable]), 98)
    diff_limit = max(np.nanpercentile(np.abs(delta[usable]), 98), 1e-12)
    extent = [source.longitude_deg[0], source.longitude_deg[-1], source.latitude_deg[0], source.latitude_deg[-1]]
    for ax, arr, title, lim in zip(axes, (a, b, delta),
        ('Original: missing-ocean faces use wet-side flux',
         'Diagnostic: all invalid faces closed',
         'Change: closed minus original'), (limit, limit, diff_limit)):
        im = ax.imshow(np.ma.masked_where(~usable, arr), origin='lower',
                       extent=extent, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim,
                       interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('Longitude (degrees east)')
        ax.set_ylabel('Latitude (degrees north)')
        fig.colorbar(im, ax=ax, label='m/year')
    png = args.output_dir / 'boundary_flux_comparison.png'
    fig.savefig(png, dpi=180)
    plt.close(fig)
    np.savez_compressed(args.output_dir / 'boundary_flux_comparison.npz',
        latitude=source.latitude_deg, longitude=source.longitude_deg, valid=usable,
        physical_ocean=ocean, original=a, all_closed=b, difference=delta)
    print('Saved:', png)

if __name__ == '__main__':
    main()
