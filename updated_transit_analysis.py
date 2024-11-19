import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import glob
import os
import json
from astropy.stats import sigma_clip

def clean_data_file(file_path):
    """Clean and load transit data files while handling different formats"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Find where the actual data starts
        for i, line in enumerate(lines):
            if '|' in line and all(col in line for col in ['HJD', 'Relative_Flux']):
                header_row = i
                data_start = i + 3
                break
                
        df = pd.read_csv(file_path, skiprows=data_start, delim_whitespace=True,
                        names=['HJD', 'Relative_Flux', 'Relative_Flux_Uncertainty', 'Accepted'])
        
        df = df[df['Accepted'] == 1].copy()
        df = df[df['Relative_Flux'].notna()].copy()
        
        # Remove outliers using sigma clipping
        flux_clean = sigma_clip(df['Relative_Flux'], sigma=5)
        df = df[~flux_clean.mask].copy()
        
        return df
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def calculate_phase(time, epoch, period):
    """Calculate orbital phase from time series"""
    phase = ((time - epoch) / period) % 1
    phase[phase > 0.5] = phase[phase > 0.5] - 1
    return phase

def transit_model(phase, depth, duration, t0):
    """Simple trapezoid transit model"""
    flux = np.ones_like(phase)
    in_transit = np.abs(phase - t0) < duration/2
    flux[in_transit] = 1 - depth
    return flux

def fit_transit_model(df):
    """Fit transit model to data"""
    try:
        # Initial parameter guesses
        p0 = [0.01, 0.1, 0.0]  # depth, duration, t0
        bounds = ([0, 0.01, -0.1], [0.1, 0.5, 0.1])  # parameter bounds
        
        popt, pcov = curve_fit(transit_model, df['phase'], df['Relative_Flux'], 
                             p0=p0, bounds=bounds, sigma=df['Relative_Flux_Uncertainty'])
        
        return popt, pcov
    except:
        return None, None

def calculate_planet_parameters(transit_depth, period, stellar_mass=1.0, stellar_radius=1.0):
    """
    Calculate basic planetary parameters
    Assuming solar-like star if stellar parameters not provided
    """
    # Convert units to SI
    period_s = period * 24 * 3600  # days to seconds
    stellar_mass_kg = stellar_mass * 1.989e30  # solar masses to kg
    stellar_radius_m = stellar_radius * 6.957e8  # solar radii to meters
    G = 6.67430e-11  # gravitational constant
    
    # Calculate planet radius (in Jupiter radii)
    planet_radius = np.sqrt(transit_depth) * stellar_radius * 9.7315  # convert to Jupiter radii
    
    # Calculate semi-major axis
    semi_major_axis = np.cbrt((G * stellar_mass_kg * period_s**2) / (4 * np.pi**2))
    semi_major_axis_au = semi_major_axis / 1.496e11  # convert to AU
    
    return {
        'Planet Radius (Jupiter radii)': planet_radius,
        'Semi-major Axis (AU)': semi_major_axis_au,
        'Impact Parameter': None  # Would need more detailed fitting
    }

def plot_transit_curve(df, model_params=None, wavelength=None, save=False, output_dir='output'):
    """Plot transit light curve with error bars and model fit"""
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    plt.errorbar(df['phase'], df['Relative_Flux'],
                yerr=df['Relative_Flux_Uncertainty'],
                fmt='o', markersize=2, alpha=0.5, color='blue',
                ecolor='gray', elinewidth=1, capsize=2, label='Data')
    
    # Plot running mean
    bin_means, bin_edges, _ = stats.binned_statistic(
        df['phase'], df['Relative_Flux'], 
        statistic='mean', bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, bin_means, 'r-', linewidth=2, alpha=0.7, label='Running Mean')
    
    # Plot model if available
    if model_params is not None:
        phase_model = np.linspace(min(df['phase']), max(df['phase']), 1000)
        flux_model = transit_model(phase_model, *model_params)
        plt.plot(phase_model, flux_model, 'g-', linewidth=2, alpha=0.7, label='Model Fit')
    
    plt.xlabel('Orbital Phase')
    plt.ylabel('Relative Flux')
    title = 'Transit Light Curve'
    if wavelength:
        title += f' at {wavelength}'
    plt.title(title)
    
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 0.05)
    plt.legend()
    
    if save:
        plt.savefig(os.path.join(output_dir, f'transit_curve_{wavelength}.png'), 
                   dpi=300, bbox_inches='tight')
    plt.close()

def analyze_transit(df, period):
    """Calculate comprehensive transit parameters"""
    # Basic transit parameters
    out_of_transit = df[abs(df['phase']) > 0.02]['Relative_Flux']
    baseline = np.median(out_of_transit)
    
    in_transit = df[abs(df['phase']) < 0.02]['Relative_Flux']
    transit_minimum = np.median(in_transit)
    
    transit_depth = baseline - transit_minimum
    
    # Transit duration
    in_transit_mask = df['Relative_Flux'] < (baseline - transit_depth/2)
    if any(in_transit_mask):
        duration = np.ptp(df[in_transit_mask]['phase']) * period
    else:
        duration = np.nan
    
    # Fit transit model
    popt, pcov = fit_transit_model(df)
    
    # Calculate planet parameters
    planet_params = calculate_planet_parameters(transit_depth, period)
    
    results = {
        'Transit Depth': transit_depth,
        'Transit Duration (days)': duration,
        'Baseline Flux': baseline,
        'Points in Transit': sum(in_transit_mask),
        'Total Points': len(df),
        'RMS Scatter': np.std(out_of_transit),
        'Model Parameters': popt.tolist() if popt is not None else None,
        **planet_params
    }
    
    return results

def plot_phase_binned(df, bin_size=0.001, save=False, output_dir='output', wavelength=None):
    """Create phase-binned light curve"""
    plt.figure(figsize=(12, 6))
    
    # Create phase bins
    bins = np.arange(-0.05, 0.05 + bin_size, bin_size)
    binned_flux = []
    binned_err = []
    bin_centers = []
    
    for i in range(len(bins)-1):
        mask = (df['phase'] >= bins[i]) & (df['phase'] < bins[i+1])
        if np.sum(mask) > 0:
            binned_flux.append(np.mean(df[mask]['Relative_Flux']))
            binned_err.append(np.std(df[mask]['Relative_Flux']) / np.sqrt(np.sum(mask)))
            bin_centers.append((bins[i] + bins[i+1])/2)
    
    plt.errorbar(bin_centers, binned_flux, yerr=binned_err, 
                fmt='o', markersize=4, color='blue', capsize=2)
    
    plt.xlabel('Orbital Phase')
    plt.ylabel('Relative Flux')
    plt.title(f'Phase-binned Transit Light Curve{" at " + wavelength if wavelength else ""}')
    plt.grid(True, alpha=0.3)
    
    if save:
        plt.savefig(os.path.join(output_dir, f'binned_transit_{wavelength}.png'), 
                   dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # System parameters
    PERIOD = 3.52474859  # days
    EPOCH = 2452826.628521  # HJD
    
    # Setup output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all data files
    data_files = glob.glob('*PLC*.txt')
    all_results = {}
    
    for file_path in data_files:
        try:
            wavelength = file_path.split('_')[2].split('.')[0]
            print(f"Processing {wavelength}...")
            
            # Load and clean data
            df = clean_data_file(file_path)
            if df is None or len(df) == 0:
                continue
                
            # Calculate orbital phase
            df['phase'] = calculate_phase(df['HJD'], EPOCH, PERIOD)
            
            # Save processed data
            processed_file = os.path.join(output_dir, f'processed_{wavelength}.csv')
            df.to_csv(processed_file, index=False)
            
            # Analyze transit
            results = analyze_transit(df, PERIOD)
            all_results[wavelength] = results
            
            # Create plots
            plot_transit_curve(df, results.get('Model Parameters'), 
                             wavelength=wavelength, save=True, output_dir=output_dir)
            plot_phase_binned(df, wavelength=wavelength, save=True, output_dir=output_dir)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Save results
    if all_results:
        # Save summary table
        summary_df = pd.DataFrame(all_results).T
        summary_df.to_csv(os.path.join(output_dir, 'transit_summary.csv'))
        
        # Save detailed results as JSON
        with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
            json.dump(all_results, f, indent=4)
        
        # Plot transit depth vs wavelength if applicable
        plt.figure(figsize=(10, 6))
        wavelengths = []
        depths = []
        depth_errors = []
        
        for wave, results in all_results.items():
            try:
                if 'nm' in wave:
                    wavelengths.append(float(wave.replace('nm', '')))
                    depths.append(results['Transit Depth'])
                    depth_errors.append(results['RMS Scatter'])
            except ValueError:
                continue
                
        if wavelengths and depths:
            plt.errorbar(wavelengths, depths, yerr=depth_errors, 
                        fmt='o-', capsize=5)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Transit Depth')
            plt.title('Transit Depth vs Wavelength')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'depth_vs_wavelength.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    main()