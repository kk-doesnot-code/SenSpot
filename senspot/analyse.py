"""
SenSpot analysis module — called when --analyse flag is used.
Produces spatial maps, distribution plots, driver genes, and summary stats.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path


def run_analysis(
    adata,
    results: pd.DataFrame,
    output_dir: str,
    sample_name: str,
    threshold: float = 0.543,
    layer: str = None,
    verbose: bool = True,
):
    """
    Run full analysis on SenSpot results.

    Parameters
    ----------
    adata       : AnnData object (input data)
    results     : DataFrame with senescence_probability and sen_positive columns
    output_dir  : Directory to save figures and summary
    sample_name : Sample name for figure titles
    threshold   : Sen+ threshold used
    layer       : AnnData layer used (default .X)
    verbose     : Print progress
    """
    os.makedirs(output_dir, exist_ok=True)
    probs = results['senescence_probability'].values
    sen   = results['sen_positive'].values.astype(bool)
    pct   = sen.mean() * 100

    if verbose:
        print(f"\n  Analysing {sample_name}...")
        print(f"  Spots: {len(probs):,} | Sen+: {sen.sum():,} ({pct:.1f}%)")

    # ── 1. Summary stats ──────────────────────────────────────
    _write_summary(results, probs, sen, pct, threshold, sample_name,
                   output_dir, verbose)

    # ── 3. Spatial maps ───────────────────────────────────────
    coords = _get_coords(adata, results)
    if coords is not None:
        _plot_spatial_probability(coords, probs, threshold, sample_name, output_dir, adata)
        _plot_spatial_binary(coords, sen, pct, sample_name, output_dir, adata)
        if verbose: print(f"  Saved spatial_map.png + binary_map.png")
    else:
        if verbose: print(f"  No spatial coords found — skipping spatial maps")

    # ── 4. Top driver genes ───────────────────────────────────
    if sen.sum() >= 10 and (~sen).sum() >= 10:
        _plot_driver_genes(adata, results, sen, sample_name, output_dir, layer, verbose)

    if verbose:
        print(f"  Analysis saved to: {output_dir}/")


def _write_summary(results, probs, sen, pct, threshold,
                   sample_name, output_dir, verbose):
    """Write summary stats to text file."""
    from scipy.stats import gaussian_kde
    kde  = gaussian_kde(probs, bw_method=0.08)
    x    = np.linspace(0.4, 0.7, 200)
    peak = x[np.argmax(kde(x))]
    bimodal = _check_bimodal(probs)

    lines = [
        f"SenSpot Analysis Summary",
        f"=" * 40,
        f"Sample:           {sample_name}",
        f"Total spots:      {len(probs):,}",
        f"Threshold:        {threshold}",
        f"",
        f"--- Senescence calls ---",
        f"Sen+ spots:       {sen.sum():,} ({pct:.1f}%)",
        f"Sen- spots:       {(~sen).sum():,} ({100-pct:.1f}%)",
        f"",
        f"--- Probability distribution ---",
        f"Mean:             {probs.mean():.4f}",
        f"Median:           {np.median(probs):.4f}",
        f"Std:              {probs.std():.4f}",
        f"Min:              {probs.min():.4f}",
        f"Max:              {probs.max():.4f}",
        f"Distribution peak:{peak:.4f}",
        f"Bimodal signal:   {'Yes' if bimodal else 'No (near-baseline)'}",
    ]

    summary = "\n".join(lines)
    with open(f"{output_dir}/senspot_summary.txt", "w") as f:
        f.write(summary)
    if verbose:
        print(f"\n{summary}")


def _check_bimodal(probs, baseline=0.5, noise_std=0.04):
    """Check if distribution is meaningfully above noise baseline."""
    return probs.mean() > baseline + noise_std


def _get_coords(adata, results):
    """Extract spatial coordinates from AnnData or obs."""
    try:
        if 'spatial' in adata.obsm:
            coords = adata.obsm['spatial']
            common = adata.obs_names.intersection(results.index)
            idx    = [list(adata.obs_names).index(b) for b in common]
            return pd.DataFrame(
                {'x': coords[idx, 0], 'y': coords[idx, 1]},
                index=common)
        elif 'pxl_col' in adata.obs.columns:
            return adata.obs[['pxl_col','pxl_row']].rename(
                columns={'pxl_col':'x','pxl_row':'y'})
    except Exception:
        pass
    return None


def _get_tissue_image(adata):
    """Try to load lowres tissue image from AnnData uns."""
    try:
        lib = list(adata.uns['spatial'].keys())[0]
        img = adata.uns['spatial'][lib]['images'].get('lowres')
        sf  = adata.uns['spatial'][lib]['scalefactors']['tissue_lowres_scalef']
        spot_r = adata.uns['spatial'][lib]['scalefactors']['spot_diameter_fullres'] * sf / 2
        return img, sf, spot_r
    except Exception:
        return None, None, None




def _plot_spatial_probability(coords, probs, threshold, sample_name,
                               output_dir, adata=None, dark=False,
                               fname='spatial_map.png'):
    """Continuous probability spatial map — white or dark background."""
    img, sf, spot_r = _get_tissue_image(adata) if adata is not None else (None, None, None)

    fig, ax = plt.subplots(figsize=(7, 7))
    bg_color = 'black' if dark else 'white'
    fig.patch.set_facecolor(bg_color)

    x = coords['x'].values
    y = coords['y'].values

    if img is not None and dark:
        from PIL import Image as PILImage, ImageEnhance
        img_pil = PILImage.fromarray((img * 255).astype(np.uint8) if img.max() <= 1 else img)
        img_pil = ImageEnhance.Brightness(img_pil).enhance(0.45)
        ax.imshow(np.array(img_pil), origin='upper')
        s = spot_r * 1.8 if spot_r else 12
        y_plot = y
    else:
        s = spot_r * 1.8 if spot_r else 12
        y_plot = -y  # flip for white bg

    sc = ax.scatter(x, y_plot, c=probs, cmap='RdYlBu_r',
                    vmin=0.4, vmax=0.7, s=s, alpha=0.80,
                    linewidths=0, rasterized=True)
    ax.axis('off')
    title_color = 'white' if dark else 'black'
    ax.set_title(f'{sample_name} — Senescence probability',
                 fontsize=12, fontweight='bold', color=title_color)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label('Sen. probability', fontsize=10, color=title_color)
    cbar.ax.axhline(y=threshold, color=title_color, lw=1.5, linestyle='--')
    if dark:
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{fname}', dpi=200,
                bbox_inches='tight', facecolor=bg_color)
    plt.close()


def _plot_spatial_binary(coords, sen, pct, sample_name,
                          output_dir, adata=None, dark=False,
                          fname='binary_map.png'):
    """Binary sen+/sen- spatial map — white or dark background."""
    img, sf, spot_r = _get_tissue_image(adata) if adata is not None else (None, None, None)

    fig, ax = plt.subplots(figsize=(7, 7))
    bg_color = 'black' if dark else 'white'
    fig.patch.set_facecolor(bg_color)

    x = coords['x'].values
    y = coords['y'].values
    y_plot = y if (img is not None and dark) else -y
    s_pos  = (spot_r * 2.0) if spot_r else 14
    s_neg  = (spot_r * 1.2) if spot_r else 8
    sen_color = '#FF3333' if dark else '#C94040'

    if img is not None and dark:
        from PIL import Image as PILImage, ImageEnhance
        img_pil = PILImage.fromarray((img*255).astype(np.uint8) if img.max()<=1 else img)
        img_pil = ImageEnhance.Brightness(img_pil).enhance(0.45)
        ax.imshow(np.array(img_pil), origin='upper')

    ax.scatter(x[~sen], y_plot[~sen], s=s_neg, c='#AAAAAA',
               alpha=0.20 if dark else 0.35, linewidths=0, rasterized=True)
    ax.scatter(x[sen],  y_plot[sen],  s=s_pos, c=sen_color,
               alpha=0.85, linewidths=0, rasterized=True)
    ax.axis('off')

    title_color = 'white' if dark else 'black'
    ax.set_title(f'{sample_name} — Sen+ spots ({pct:.1f}%)',
                 fontsize=12, fontweight='bold', color=title_color)

    fc_color = '#222222' if dark else 'white'
    lc = 'white' if dark else 'black'
    ax.legend(handles=[
        mpatches.Patch(color=sen_color, label=f'Sen+ ({pct:.0f}%)'),
        mpatches.Patch(color='#AAAAAA', label=f'Sen− ({100-pct:.0f}%)', alpha=0.5)],
        loc='lower right', fontsize=10, framealpha=0.7,
        facecolor=fc_color, labelcolor=lc)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{fname}', dpi=200,
                bbox_inches='tight', facecolor=bg_color)
    plt.close()


def _plot_driver_genes(adata, results, sen, sample_name,
                        output_dir, layer, verbose):
    """Top LFC genes in sen+ vs sen- spots."""
    import scipy.sparse as sp

    try:
        common = adata.obs_names.intersection(results.index)
        adata_s = adata[common].copy()
        X = adata_s.X.toarray() if sp.issparse(adata_s.X) else adata_s.X
        df_expr = pd.DataFrame(X, columns=adata_s.var_names, index=common)

        # Align sen mask
        sen_aligned = results.loc[common, 'sen_positive'].values.astype(bool)
        if sen_aligned.sum() < 5 or (~sen_aligned).sum() < 5:
            return

        lfc_rows = []
        for gene in df_expr.columns:
            pos_vals = df_expr.loc[sen_aligned, gene].values
            neg_vals = df_expr.loc[~sen_aligned, gene].values
            if pos_vals.mean() == 0 and neg_vals.mean() == 0:
                continue
            lfc = pos_vals.mean() - neg_vals.mean()
            _, p = stats.ttest_ind(pos_vals, neg_vals)
            lfc_rows.append({'gene': gene, 'lfc': lfc, 'p': p})

        lfc_df = pd.DataFrame(lfc_rows)
        lfc_df = lfc_df[lfc_df['p'] < 0.05].copy()
        top_up   = lfc_df.nlargest(15, 'lfc')
        top_down = lfc_df.nsmallest(10, 'lfc')
        top_genes = pd.concat([top_up, top_down]).sort_values('lfc')

        # Save CSV
        lfc_df.sort_values('lfc', ascending=False).to_csv(
            f'{output_dir}/driver_genes.csv', index=False)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('white')
        colors = ['#C94040' if v > 0 else '#4A90C4' for v in top_genes['lfc']]
        ax.barh(top_genes['gene'], top_genes['lfc'],
                color=colors, alpha=0.8, edgecolor='black', linewidth=0.4)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('LFC (sen+ vs sen−)', fontsize=11)
        ax.set_title(f'{sample_name} — Top driver genes\n'
                     f'(log-normalised expression, p<0.05)',
                     fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_drivers.png', dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        if verbose:
            print(f"  Saved top_drivers.png + driver_genes.csv")

    except Exception as e:
        if verbose:
            print(f"  Driver gene analysis failed: {e}")
