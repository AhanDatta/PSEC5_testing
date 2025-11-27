import matplotlib.pyplot as plt 
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy import optimize

# --- Plotting Style Settings ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "mathtext.fontset": "cm"
})

# --- Data Definition ---
DIV_RATIO = 512

# Digital Band 111111 (63 decimal)
VTUNE_63 = np.array([0.070, 0.240, 0.391, 0.467, 0.624, 0.721, 0.905, 1.036, 1.120])
d_VTUNE_63 = 0.003 * np.ones(len(VTUNE_63))
REF_FREQ_63 = np.array([7.339, 7.400, 7.463, 7.505, 7.590, 7.648, 7.734, 7.798, 7.828])
d_REF_FREQ_63 = 0.015 * np.ones(len(REF_FREQ_63))
VCO_FREQ_63 = REF_FREQ_63 * DIV_RATIO
d_VCO_FREQ_63 = d_REF_FREQ_63 * DIV_RATIO

# Digital Band 000000 (0 decimal)
VTUNE_0 = np.array([0.071, 0.241, 0.322, 0.493, 0.595, 0.703, 0.875, 1.060, 1.173])
d_VTUNE_0 = 0.003 * np.ones(len(VTUNE_0))
REF_FREQ_0 = np.array([6.329, 6.369, 6.390, 6.441, 6.483, 6.515, 6.579, 6.634, 6.667])
d_REF_FREQ_0 = 0.015 * np.ones(len(REF_FREQ_0))
VCO_FREQ_0 = REF_FREQ_0 * DIV_RATIO
d_VCO_FREQ_0 = d_REF_FREQ_0 * DIV_RATIO

# Digital Band 100010 (34 decimal)
VTUNE_34 = np.array([0.070, 0.204, 0.384, 0.546, 0.648, 0.739, 0.865, 0.937, 1.041, 1.168])
d_VTUNE_34 = 0.003 * np.ones(len(VTUNE_34))
REF_FREQ_34 = np.array([6.681, 6.826, 6.920, 6.993, 7.030, 7.067, 7.117, 7.156, 7.194, 7.246])
d_REF_FREQ_34 = 0.015 * np.ones(len(REF_FREQ_34))
VCO_FREQ_34 = REF_FREQ_34 * DIV_RATIO
d_VCO_FREQ_34 = d_REF_FREQ_34 * DIV_RATIO

@dataclass
class FitResult:
    """Results from a peak fit"""
    params: NDArray[np.float_]
    errors: NDArray[np.float_]
    chisq: float
    dof: int
    
    @property
    def red_chisq(self) -> float:
        return self.chisq / self.dof if self.dof > 0 else np.nan

def residual(p: NDArray, func, xvar: NDArray, yvar: NDArray, err: NDArray) -> NDArray:
    """Calculate normalized residuals"""
    return (func(p, xvar) - yvar) / err

def linear_func(p: list[float], x: float) -> float:
    """
    Inputs: p, x 
    p[0] = slope
    p[1] = y-intercept
    x = input variable
    """
    return p[0] * x + p[1]

def fit_func(p0: list, func, xvar: NDArray, yvar: NDArray, err: NDArray) -> FitResult:
    """Fit a peak with the given function and return results"""
    try:
        fit = optimize.least_squares(residual, p0, args=(func, xvar, yvar, err), verbose=0)
    except Exception as error:
        print(f"Fit failed: {error}")
        return FitResult(np.array(p0), np.zeros_like(p0), np.nan, 0)
    
    pf = fit['x']
    
    try:
        cov = np.linalg.inv(fit['jac'].T.dot(fit['jac']))
    except:
        print('Fit did not converge')
        return FitResult(pf, np.zeros_like(pf), np.nan, 0)
    
    chisq = sum(residual(pf, func, xvar, yvar, err)**2)
    dof = len(xvar) - len(pf)
    pferr = np.sqrt(np.diagonal(cov))
    
    return FitResult(pf, pferr, chisq, dof)

def plot_multiband_vco(results_raw: dict, results_vco: dict, data_raw: dict, data_vco: dict, div_ratio: int):
    """
    Plots the reference frequency and VCO frequency vs voltage for multiple digital bands
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    colors = {'63': 'red', '0': 'blue', '34': 'green'}
    markers = {'63': 'o', '0': 's', '34': '^'}
    band_labels = {'63': '111111', '0': '000000', '34': '100010'}
    
    # ===== LEFT PLOT: Raw Reference Frequency =====
    for band_name, band_data in data_raw.items():
        xvar, yvar, xerr, yerr = band_data
        color = colors[band_name]
        marker = markers[band_name]
        
        # Plot raw data
        ax1.errorbar(xvar, yvar, xerr=xerr, yerr=yerr, 
                   fmt=marker, color=color, capsize=3, 
                   label=f'Band {band_labels[band_name]}', 
                   markersize=6, alpha=0.7)
        
        # Plot fit line
        res = results_raw[band_name]
        x_smooth = np.linspace(0, 1.2, 100)
        y_fit = linear_func(res.params, x_smooth)
        ax1.plot(x_smooth, y_fit, color=color, linestyle='-', linewidth=2, alpha=0.8)

    # Annotation boxes for raw data
    y_positions = [0.93, 0.24, 0.56]
    for i, band_name in enumerate(['63', '0', '34']):
        res = results_raw[band_name]
        slope, intercept = res.params
        d_slope, d_intercept = res.errors
        color = colors[band_name]
        
        fit_info = (
            r'$\mathbf{{Band\ {}}}$'.format(band_labels[band_name]) + '\n' +
            r'$m = {:.2f} \pm {:.2f}\ \mathrm{{MHz/V}}$'.format(slope, d_slope) + '\n' +
            r'$c = {:.2f} \pm {:.2f}\ \mathrm{{MHz}}$'.format(intercept, d_intercept) + '\n' +
            r'$\chi^2_\nu = {:.2f}$'.format(res.red_chisq)
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor=color, linewidth=2)
        ax1.text(0.02, y_positions[i], fit_info, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')

    ax1.set_xlabel(r'Analog Control Voltage, $V_{\mathrm{tune}}$ (V)')
    ax1.set_ylabel(r'Reference Clock Frequency (MHz)')
    ax1.set_title(f'Reference Clock Output\n(Division Ratio = {div_ratio})')
    ax1.minorticks_on()
    ax1.grid(which='major', linestyle='-', linewidth=0.7, color='black', alpha=0.4)
    ax1.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.3)
    ax1.legend(loc='lower right', fontsize=10)
    
    # ===== RIGHT PLOT: VCO Frequency =====
    for band_name, band_data in data_vco.items():
        xvar, yvar, xerr, yerr = band_data
        color = colors[band_name]
        marker = markers[band_name]
        
        # Plot VCO data
        ax2.errorbar(xvar, yvar, xerr=xerr, yerr=yerr, 
                   fmt=marker, color=color, capsize=3, 
                   label=f'Band {band_labels[band_name]}', 
                   markersize=6, alpha=0.7)
        
        # Plot fit line
        res = results_vco[band_name]
        x_smooth = np.linspace(0, 1.2, 100)
        y_fit = linear_func(res.params, x_smooth)
        ax2.plot(x_smooth, y_fit, color=color, linestyle='-', linewidth=2, alpha=0.8)

    # Annotation boxes for VCO data
    for i, band_name in enumerate(['63', '0', '34']):
        res = results_vco[band_name]
        slope, intercept = res.params
        d_slope, d_intercept = res.errors
        color = colors[band_name]
        
        fit_info = (
            r'$\mathbf{{Band\ {}}}$'.format(band_labels[band_name]) + '\n' +
            r'$m = {:.1f} \pm {:.1f}\ \mathrm{{MHz/V}}$'.format(slope, d_slope) + '\n' +
            r'$c = {:.1f} \pm {:.1f}\ \mathrm{{MHz}}$'.format(intercept, d_intercept) + '\n' +
            r'$\chi^2_\nu = {:.2f}$'.format(res.red_chisq)
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor=color, linewidth=2)
        ax2.text(0.02, y_positions[i], fit_info, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')

    ax2.set_xlabel(r'Analog Control Voltage, $V_{\mathrm{tune}}$ (V)')
    ax2.set_ylabel(r'VCO Frequency (MHz)')
    ax2.set_title(f'VCO Frequency\n(Reference × {div_ratio})')
    ax2.minorticks_on()
    ax2.grid(which='major', linestyle='-', linewidth=0.7, color='black', alpha=0.4)
    ax2.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Perform fits for RAW reference frequency data
    results_raw = {}
    
    print("=== RAW REFERENCE FREQUENCY FITS ===")
    print("\n=== Band 111111 (63) ===")
    results_raw['63'] = fit_func(
        p0=[0.5, 7],
        func=linear_func,
        xvar=VTUNE_63,
        yvar=REF_FREQ_63,
        err=d_REF_FREQ_63
    )
    
    print("\n=== Band 000000 (0) ===")
    results_raw['0'] = fit_func(
        p0=[0.3, 6],
        func=linear_func,
        xvar=VTUNE_0,
        yvar=REF_FREQ_0,
        err=d_REF_FREQ_0
    )
    
    print("\n=== Band 100010 (34) ===")
    results_raw['34'] = fit_func(
        p0=[0.4, 6.5],
        func=linear_func,
        xvar=VTUNE_34,
        yvar=REF_FREQ_34,
        err=d_REF_FREQ_34
    )
    
    # Perform fits for VCO frequency data
    results_vco = {}
    
    print("\n\n=== VCO FREQUENCY FITS ===")
    print("\n=== Band 111111 (63) ===")
    results_vco['63'] = fit_func(
        p0=[400, 3600],
        func=linear_func,
        xvar=VTUNE_63,
        yvar=VCO_FREQ_63,
        err=d_VCO_FREQ_63
    )
    
    print("\n=== Band 000000 (0) ===")
    results_vco['0'] = fit_func(
        p0=[300, 3200],
        func=linear_func,
        xvar=VTUNE_0,
        yvar=VCO_FREQ_0,
        err=d_VCO_FREQ_0
    )
    
    print("\n=== Band 100010 (34) ===")
    results_vco['34'] = fit_func(
        p0=[350, 3400],
        func=linear_func,
        xvar=VTUNE_34,
        yvar=VCO_FREQ_34,
        err=d_VCO_FREQ_34
    )
    
    # Prepare data dictionaries
    data_raw = {
        '63': (VTUNE_63, REF_FREQ_63, d_VTUNE_63, d_REF_FREQ_63),
        '0': (VTUNE_0, REF_FREQ_0, d_VTUNE_0, d_REF_FREQ_0),
        '34': (VTUNE_34, REF_FREQ_34, d_VTUNE_34, d_REF_FREQ_34)
    }
    
    data_vco = {
        '63': (VTUNE_63, VCO_FREQ_63, d_VTUNE_63, d_VCO_FREQ_63),
        '0': (VTUNE_0, VCO_FREQ_0, d_VTUNE_0, d_VCO_FREQ_0),
        '34': (VTUNE_34, VCO_FREQ_34, d_VTUNE_34, d_VCO_FREQ_34)
    }
    
    # Generate plot
    plot_multiband_vco(results_raw, results_vco, data_raw, data_vco, DIV_RATIO)