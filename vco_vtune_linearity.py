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
    "mathtext.fontset": "cm" # Uses Computer Modern for LaTeX math
})

# --- Data Definition ---
VTUNE = np.array([0.067, 0.202, 0.398, 0.474, 0.618, 0.727, 0.811, 0.919, 0.992, 1.213]) 
d_VTUNE = 0.002 * np.ones(len(VTUNE)) 
VCO_FREQ = np.array([56.0, 56.5, 57.0, 57.3, 57.8, 58.2, 58.6, 59.0, 59.3, 59.8]) 
d_VCO_FREQ = 0.2 * np.ones(len(VCO_FREQ)) 

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
    
    # Console Output
    print(f'\nConverged with chi-squared {chisq:.2f}')
    print(f'Number of degrees of freedom, dof = {dof}')
    print(f'Reduced chi-squared {chisq/dof:.2f}\n')
    print(f'{"Parameter #":<11} | {"Initial":<24} | {"Best fit":<24} | {"Uncertainty":<24}')
    for num in range(len(pf)):
        print(f'{num:<11} | {p0[num]:<24.3e} | {pf[num]:<24.3e} | {pferr[num]:<24.3e}')
    
    return FitResult(pf, pferr, chisq, dof)

def plot_vco_fit(res: FitResult, xvar: NDArray, yvar: NDArray, xerr: NDArray, yerr: NDArray, func):
    """
    Plots the raw data with error bars and the resulting fit line.
    Adds an annotation box with the fit parameters and chi-squared stats.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Plot Raw Data with Error Bars
    # xerr is included here. Note: If d_VTUNE is small, the bars may be smaller than the marker.
    ax.errorbar(xvar, yvar, xerr=xerr, yerr=yerr, fmt='ko', capsize=3, label='Raw Data', markersize=5)

    # 2. Plot Fit Line
    x_smooth = np.linspace(min(xvar), max(xvar), 100)
    y_fit = func(res.params, x_smooth)
    ax.plot(x_smooth, y_fit, 'r-', linewidth=1.5, label='Linear Fit')

    # 3. Create Annotation Text
    slope, intercept = res.params
    d_slope, d_intercept = res.errors
    
    fit_info = (
        r'$\mathbf{Fit\ Model:}\ y = mx + c$' + '\n'
        r'$m = ({:.3f} \pm {:.3f})\ \mathrm{{MHz/V}}$'.format(slope, d_slope) + '\n'
        r'$c = ({:.3f} \pm {:.3f})\ \mathrm{{MHz}}$'.format(intercept, d_intercept) + '\n'
        r'$\chi^2_\nu = {:.2f}$'.format(res.red_chisq)
    )

    # 4. Add Text Box to Plot
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.95, fit_info, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # 5. Labels
    ax.set_xlabel(r'VCO Control Voltage, $V_{tune}$ (V)')
    ax.set_ylabel(r'Reference Clock Frequency (MHz)') # Updated Label
    ax.set_title('VCO Tuning Characteristic')

    # 6. Grid Settings (Finer Sub-grid)
    ax.minorticks_on()
    # Major grid: distinct and slightly darker
    ax.grid(which='major', linestyle='-', linewidth=0.7, color='black', alpha=0.6)
    # Minor grid: dotted and lighter
    ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)

    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    init_guess = [4, 56]
    
    # Perform Fit
    fit_result: FitResult = fit_func(
        p0=init_guess, 
        func=linear_func, 
        xvar=VTUNE, 
        yvar=VCO_FREQ, 
        err=d_VCO_FREQ
    )
    
    # Generate Plot
    if not np.isnan(fit_result.chisq):
        plot_vco_fit(
            res=fit_result,
            xvar=VTUNE,
            yvar=VCO_FREQ,
            xerr=d_VTUNE, 
            yerr=d_VCO_FREQ,
            func=linear_func
        )