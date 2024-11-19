import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import glob
import os

def clean_data_file(file_path):
    """Clean and load transit data files while handling different formats"""
    try:
        # Read the entire file first to find the data start
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Find where the actual data starts
        for i, line in enumerate(lines):
            if '|' in line and all(col in line for col in ['HJD', 'Relative_Flux']):
                header_row = i
                data_start = i + 3
                break
                
        # Read the data portion with pandas
        df = pd.read_csv(file_path, skiprows=data_start, delim_whitespace=True,
                        names=['HJD', 'Relative_Flux', 'Relative_Flux_Uncertainty', 'Accepted'])
        
        # Clean the data
        df = df[df['Accepted'] == 1].copy()
        df = df[df['Relative_Flux'].notna()].copy()
        
        return df
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def calculate_phase(time, epoch, period):
    """Calculate orbital phase from time series"""
    phase = ((time - epoch) / period) % 1
    phase[phase > 0.5] = phase[phase > 0.5] - 1
    return phase

def plot_transit_curve(df, wavelength=None, save=False, output_dir='output'):
    """Plot transit light curve with error bars"""
    plt.figure(figsize=(12, 6))
    
    # Plot points with error bars
    plt.errorbar(df['phase'], df['Relative_Flux'],
                yerr=df['Relative_Flux_Uncertainty'],
                fmt='o', markersize=2, alpha=0.5, color='blue',
                ecolor='gray', elinewidth=1, capsize=2)
    
    # Add labels and title
    plt.xlabel('Orbital Phase')
    plt.ylabel('Relative Flux')
    title = 'Transit Light Curve'
    if wavelength:
        title += f' at {wavelength}'
    plt.title(title)
    
    # Customize plot
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 0.05)  # Focus on the transit
    
    # Calculate and plot running mean
    bin_means, bin_edges, _ = stats.binned_statistic(
        df['phase'], df['Relative_Flux'], 
        statistic='mean', bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, bin_means, 'r-', linewidth=2, alpha=0.7)
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'transit_curve_{wavelength}.png'), 
                   dpi=300, bbox_inches='tight')
    plt.close()

def analyze_transit(df):
    """Calculate basic transit parameters"""
    # Find transit depth
    out_of_transit = df[abs(df['phase']) > 0.02]['Relative_Flux']
    baseline = np.median(out_of_transit)
    
    in_transit = df[abs(df['phase']) < 0.02]['Relative_Flux']
    transit_minimum = np.median(in_transit)
    
    transit_depth = baseline - transit_minimum
    
    # Estimate transit duration
    in_transit_mask = df['Relative_Flux'] < (baseline - transit_depth/2)
    if any(in_transit_mask):
        duration = np.ptp(df[in_transit_mask]['phase']) * 3.52474859  # Convert to days
    else:
        duration = np.nan
        
    results = {
        'Transit Depth': transit_depth,
        'Transit Duration (days)': duration,
        'Baseline Flux': baseline,
        'Points in Transit': sum(in_transit_mask),
        'Total Points': len(df)
    }
    
    return results

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
            
            # Plot transit curve
            plot_transit_curve(df, wavelength=wavelength, save=True, output_dir=output_dir)
            
            # Analyze transit parameters
            results = analyze_transit(df)
            all_results[wavelength] = results
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Create summary table
    if all_results:
        summary_df = pd.DataFrame(all_results).T
        summary_df.to_csv(os.path.join(output_dir, 'transit_summary.csv'))
        
        # Plot transit depth vs wavelength
        plt.figure(figsize=(10, 6))
        wavelengths = []
        depths = []
        
        for wave, results in all_results.items():
            try:
                if 'nm' in wave:
                    wavelengths.append(float(wave.replace('nm', '')))
                    depths.append(results['Transit Depth'])
            except ValueError:
                continue
                
        if wavelengths and depths:
            plt.plot(wavelengths, depths, 'o-')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Transit Depth')
            plt.title('Transit Depth vs Wavelength')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'depth_vs_wavelength.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    main()