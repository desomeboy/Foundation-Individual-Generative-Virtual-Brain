import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


def plot_enhanced_fc_correlation(x, y, patient_id, output_dir, fig_size=(6, 6), dpi=300, max_points=10000):
    """
    Create an enhanced FC correlation plot with:
    1. y=x reference line
    2. OLS regression line with confidence interval
    3. 2D KDE density background
    4. Statistical annotations
    5. Scientific styling
    
    Parameters:
    -----------
    x : array-like
        Model FC values (flattened without diagonal)
    y : array-like
        Empirical FC values (flattened without diagonal)
    patient_id : str/int
        Patient identifier
    output_dir : str
        Directory to save the plot
    fig_size : tuple
        Figure dimensions (width, height) in inches
    dpi : int
        Resolution for saved figure
    max_points : int
        Maximum points to display (for performance with large datasets)
    """
    
    # Create figure with scientific styling
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': dpi,
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '.3'
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Subsample points if needed for performance
    if len(x) > max_points:
        idx = np.random.choice(len(x), max_points, replace=False)
        x_plot, y_plot = x[idx], y[idx]
    else:
        x_plot, y_plot = x, y
    
    # 1. Plot 2D KDE density background
    sns.kdeplot(
        x=x_plot, y=y_plot,
        cmap="viridis",
        fill=True,
        alpha=0.7,
        levels=15,
        thresh=0.05,
        zorder=1,
        ax=ax
    )
    
    # 2. Plot scatter points with size based on density
    # Calculate point sizes inversely proportional to local density
    from scipy.stats import gaussian_kde
    xy = np.vstack([x_plot, y_plot])
    z = gaussian_kde(xy)(xy)
    idx_sort = z.argsort()  # Sort points by density for proper layering
    
    scatter = ax.scatter(
        x_plot[idx_sort], y_plot[idx_sort],
        c=z[idx_sort],
        s=12 * (1/(z[idx_sort] + 0.01)),  # Larger points in sparse regions
        cmap="plasma",
        alpha=0.8,
        edgecolor='w',
        linewidth=0.3,
        vmin=z.min(),
        vmax=z.max()*1.5,
        zorder=2
    )
    
    # 3. Add colorbar for density
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Point Density', fontsize=13)
    
    # 4. Add y=x reference line
    min_val = min(np.min(x), np.min(y))
    max_val = max(np.max(x), np.max(y))
    ax.plot([min_val, max_val], [min_val, max_val], 
            'k--', linewidth=2, alpha=0.8, label='y = x', zorder=3)
    
    # 5. Add OLS regression line with confidence interval
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r2_value = r_value**2
    
    # Generate regression line points
    x_reg = np.linspace(min_val, max_val, 100)
    y_reg = slope * x_reg + intercept
    
    # Calculate confidence intervals
    n = len(x)
    x_bar = np.mean(x)
    sxx = np.sum((x - x_bar)**2)
    s_err = np.sqrt(np.sum((y - (slope*x + intercept))**2)/(n-2))
    conf = 1.96 * s_err * np.sqrt(1/n + (x_reg - x_bar)**2/sxx)  # 95% CI
    
    # Plot regression line and confidence band
    ax.plot(x_reg, y_reg, 'r-', linewidth=2.5, label=f'Regression: y = {slope:.2f}x + {intercept:.2f}', zorder=4)
    ax.fill_between(x_reg, y_reg - conf, y_reg + conf, color='red', alpha=0.15, zorder=3)
    
    # 6. Add statistical annotations
    # stats_text = (f"Pearson r = {r_value:.3f} (p={p_value:.2e})\n"
    #               f"R² = {r2_value:.3f}\n"
    #               f"Slope = {slope:.3f} ± {std_err:.3f}")
    
    # # Position text in lower-right corner
    # ax.text(0.98, 0.05, stats_text,
    #         transform=ax.transAxes,
    #         ha='right', va='bottom',
    #         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.8'),
    #         fontsize=13,
    #         zorder=5)
    
    # 7. Set axis limits and labels
    ax.set_xlim(-0.75, 1.05)
    ax.set_ylim(-0.75, 1.05)
    ax.set_xlabel('Model FC', fontweight='bold')
    ax.set_ylabel('Empirical FC', fontweight='bold')
    
    # 8. Add grid for readability
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # 9. Add legend
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),  # x=0.5: 水平居中；y=0.02: 紧贴X轴上方，不遮挡
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        ncol=1,
        fancybox=False,
        shadow=False
    )
  

    # 10. Tight layout and save
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"patient_{patient_id}_enhanced_FC_correlation.pdf")
    plt.savefig(scatter_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    
    print(f"Enhanced FC correlation plot saved to {scatter_path}")
    return r_value, p_value, r2_value, slope

# def plot_enhanced_fc_correlation(x, y, patient_id, output_dir, fig_size=(6, 6), dpi=300, max_points=5000):
#     """
#     Create a scientifically rigorous FC correlation plot with:
#     - Minimalist design without titles or statistical annotations
#     - Optimized point sizes and density visualization
#     - Reference lines with clear visual hierarchy
#     - Publication-ready styling
    
#     Parameters:
#     -----------
#     x : array-like
#         Model FC values (flattened without diagonal)
#     y : array-like
#         Empirical FC values (flattened without diagonal)
#     patient_id : str/int
#         Patient identifier
#     output_dir : str
#         Directory to save the plot
#     fig_size : tuple
#         Figure dimensions (width, height) in inches
#     dpi : int
#         Resolution for saved figure
#     max_points : int
#         Maximum points to display (for performance)
#     """
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     from scipy import stats
#     import numpy as np
    
#     # Scientific styling parameters
#     plt.rcParams.update({
#         'font.family': 'sans-serif',
#         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
#         'font.size': 12,
#         'axes.labelsize': 14,
#         'axes.linewidth': 1.2,
#         'axes.edgecolor': '.3',
#         'axes.spines.top': False,
#         'axes.spines.right': False,
#         'xtick.major.width': 1.2,
#         'ytick.major.width': 1.2,
#         'figure.dpi': dpi
#     })
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=fig_size)
    
#     # Subsample points if needed
#     if len(x) > max_points:
#         idx = np.random.choice(len(x), max_points, replace=False)
#         x_plot, y_plot = x[idx], y[idx]
#     else:
#         x_plot, y_plot = x, y
    
#     # 1. Plot 2D KDE density background with subtle gradient
#     sns.kdeplot(
#         x=x_plot, y=y_plot,
#         cmap="Blues",  # Softer blue gradient
#         fill=True,
#         alpha=0.3,     # Much more transparent
#         levels=8,      # Fewer contour levels
#         thresh=0.05,
#         zorder=1,
#         ax=ax
#     )
    
#     # 2. Plot scatter points with fixed small size
#     ax.scatter(
#         x_plot, y_plot,
#         s=8,           # Significantly smaller points
#         color='#2C3E50',  # Deep blue-gray
#         alpha=0.4,     # Higher transparency
#         edgecolor='none',  # No borders
#         zorder=2
#     )
    
#     # 3. Add y=x reference line (subtle)
#     min_val = min(np.min(x), np.min(y), -0.75)
#     max_val = max(np.max(x), np.max(y), 1.05)
#     ax.plot([min_val, max_val], [min_val, max_val], 
#             'k--', linewidth=1.5, alpha=0.7, label='Identity (y=x)', zorder=3)
    
#     # 4. Add OLS regression line (prominent)
#     slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
#     x_reg = np.linspace(min_val, max_val, 100)
#     y_reg = slope * x_reg + intercept
    
#     # Plot regression line without confidence interval (simpler)
#     ax.plot(x_reg, y_reg, color='#E74C3C', linewidth=2.5, 
#             label=f'Regression (slope={slope:.2f})', zorder=4)
    
#     # 5. Set axis limits and labels
#     ax.set_xlim(-0.75, 1.05)
#     ax.set_ylim(-0.75, 1.05)
#     ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
#     ax.set_yticks([-0.5, 0.0, 0.5, 1.0])
#     ax.set_xlabel('Model FC', fontweight='bold', labelpad=8)
#     ax.set_ylabel('Empirical FC', fontweight='bold', labelpad=8)
    
#     # 6. Add subtle grid
#     ax.grid(True, linestyle=':', alpha=0.3, zorder=0)
    
#     # 7. Legend with minimal styling
#     legend = ax.legend(
#         loc='upper left',
#         frameon=True,
#         framealpha=0.92,
#         edgecolor='#444444',
#         facecolor='white',
#         fontsize=12
#     )
#     legend.get_frame().set_linewidth(0.8)
    
#     # 8. Adjust layout and save
#     plt.tight_layout(pad=1.5)
#     scatter_path = os.path.join(output_dir, f"patient_{patient_id}_enhanced_FC_correlation.pdf")
#     plt.savefig(scatter_path, bbox_inches='tight', dpi=dpi)
#     plt.close(fig)
    
#     print(f"Enhanced FC correlation plot saved to {scatter_path}")
#     return r_value, p_value,None,None
