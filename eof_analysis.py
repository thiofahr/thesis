import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xeofs as xe
import xarray as xr
from scipy.signal import correlate
from scipy.stats import linregress
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import TABLEAU_COLORS as tableau
from matplotlib.colors import Normalize, TwoSlopeNorm

class EOF():
    def __init__(self, ds, n_modes=10):
        self.ds      = ds.sortby('time')
        self.n_modes = n_modes
        self.dim     = 'time'
        self.eofs    = None 

    def fit(self):
        
        eofs = {}
        
        for var, var_data in self.ds.data_vars.items():                
            dat = var_data.fillna(0)
            
            # --- Perform EOF analysis
            eof    = xe.single.EOF(n_modes=self.n_modes, random_state=1, use_coslat=True).fit(dat, dim=self.dim)
            comps  = eof.components()
            pc     = eof.scores()
            evr    = eof.explained_variance_ratio() * 100

            # --- Calculate the derivatives
            pc1, pc2 = pc[:2]
            
            # Normalized the principal components
            sigma_pc1 = np.std(pc1)
            sigma_pc2 = np.std(pc2)

            pc1_normalized = pc1 / sigma_pc1
            pc2_normalized = pc2 / sigma_pc2

            # Calculation
            amp   = np.sqrt(pc1_normalized**2 + pc2_normalized**2)
            if var == 'KW':
                phase = np.arctan2(-pc2_normalized, pc1_normalized) % (2 * np.pi)
            elif var == "ER":
                phase = np.arctan2(pc2_normalized, pc1_normalized) % (2 * np.pi)
            # --- Store the results in a dictionary
            eofs[var] = xr.Dataset(
                {
                    'comps': comps, 
                    'scores': pc, 
                    'evr': evr,
                    'amp': amp,
                    'phase': phase,
                }
            )

        self.eofs = eofs 
        
    def get_eofs(self):
        if self.eofs is None:
            raise ValueError("EOF analysis not performed yet. Call fit() first.")
        return self.eofs



# --- Load Data ---
# Area of interest
eof_selection = {
    'time': slice('2023-01', '2024-12'),
    'lon' : coverage['lon']
}

eof_input = waves.sel(**eof_selection)

# --- EOF ---
eof_analysis = EOF(eof_input)
eof_analysis.fit()
eof_out = eof_analysis.get_eofs()

# --- Export the data --- 
for key, result in eof_out.items():
    for var, res in result.data_vars.items():
        if 'solver_kwargs' in res.attrs:
            del res.attrs['solver_kwargs']
    result.to_netcdf(f'{event}.EOF.{key}.nc')


# --- Phase Speed Analysis --- 
# Visualization parameter
colors  = list(tableau.keys()) # using Tableau color palette
lat_format = lambda v, _: f'{v:.0f}°N' if v > 0 else (f'{v:.0f}°S' if v < 0 else '0')
lon_format = lambda v, _: f'{abs(v-360):.0f}°W' if v > 180 else (f'{v:.0f}°E' if v < 180 else '180°')

elnino_period = dict(time=slice('2023-01', '2024-12'))

# --- CSV parameters --- 
columns = ['Wave', 'Start', 'Stop', 'Period', 'Phase-1', 'Phase-2', 'Phase-3', 
           'Phase-4', 'Phase-5', 'Phase-6', 'Phase-7', 'Phase-8', 'Mean', 'STD']
rows = []

# --- Calculate the Phase Speeds ---
# Parameters
n_phases   = 8
deg_to_m   = 111e3  # meters per degree at equator

# Loop over each variable in EOFs
for n, (key, eof) in enumerate(eof_out.items()):
    
    # Latitude band
    if key == 'KW':
        lat_band   = slice(-2, 2)
        name = "Kelvin Waves"
    elif key == 'ER':
        lat_band   = slice(-5, 5)
        name = "Equatorial Rossby Waves"
    else:
        raise('Wrong Variable!')

    print(f"\n{name}")
    
    # Phase Binning
    
    phase = eof['phase'].sel(**elnino_period)
    phase_bins = np.linspace(0, 2 * np.pi, n_phases + 1)

    # Create Phase Composites
    sla = eof_input[key].sel(lat=lat_band, **elnino_period)
    composites = []
    for p in range(n_phases):
        mask = (phase >= phase_bins[p]) & (phase < phase_bins[p + 1]) 
        comp = sla.where(mask).mean(dim='time')
        comp.attrs['phase'] = p + 1
        comp.attrs['days']  = mask.values.sum()
        composites.append(comp)

    # Compute Latitude-Averaged Profiles
    profile = [comp.mean(dim='lat') * 100 for comp in composites]  # Convert to cm

    # Estimate Phase Duration from Unwrapped Phase
    w_unwrapped = np.unwrap(phase.values)
    t = np.arange(len(phase))
    slope, _, _, _, _ = linregress(t, w_unwrapped)
    ndays = abs(np.round(2 * np.pi / slope))
    ndays = 70 if key == 'KW' else 365 
    delta_t = ndays/n_phases * 24 * 3600  # seconds
    print(f"Wave Period: {ndays:.0f} days")

    # Estimate Phase Speeds 
    lon = composites[0].lon
    if not np.all(np.diff(lon) > 0):
        raise ValueError("Longitude array is not increasing. Reorder the data before proceeding.")
        
    lon_res = np.mean(np.diff(lon))
    phase_speed = []
    print("Phase Speeds:")
    for i in range(0, n_phases):
        a = profile[(i-1) % n_phases].values    # phase before target 
        b = profile[(i+1) % n_phases].values    # phase after target
        corr = correlate(a, b, mode='same')   # correlate phases before and after of the target phase
        max_lag = lon.values[np.argmax(corr)]
        # distance
        delta_x = max_lag * lon_res * deg_to_m
        
        # speed
        v = abs(delta_x / delta_t)
        phase_speed.append(v)
        print(f"phase-{i+1}: {v:.2f} m/s  |  max lag: {max_lag:.0f}°  |  delta x: {delta_x:.0f} m")
    # Compute the standard deviation
    std_speed  = np.std(phase_speed)
    mean_speed = np.mean(phase_speed)
    print(f"mean: {mean_speed:.2f} ± {std_speed:.2f} m/s")

    # --- Visualization of Phase Composite ---
    # Amplitude Plots
    fig, ax = plt.subplots(nrows=n_phases, figsize=(10, 2 * n_phases), sharex=True, dpi=300)
    for j, (wave, comp) in enumerate(zip(profile, composites)):
        # plot
        fill = ax[j].fill_between(wave.lon, wave, interpolate=True, color=colors[n])
        # add information of days and speed
        at = AnchoredText(f"Phase-{j + 1}: {comp.attrs['days']} days, {phase_speed[j]:.2f} m/s", 
                          prop=dict(size=8), frameon=True, loc='upper left')
        at.patch.set_edgecolor("black")
        ax[j].add_artist(at)
        # configure the axes
        ax[j].set_xlim(150, 270)
        #ax[j].set_ylabel(f'Phase {j + 1}')
        ax[j].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax[j].yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.0f} cm' if v != 0 else '0'))
        ax[j].tick_params(axis='both', which='major', labelsize=8)
        if j != 0:
            ax[j].spines['top'].set_visible(False)
        if j != n_phases - 1:
            ax[j].spines['bottom'].set_visible(False)
    ax[0].set_title(f'{name}', fontweight='bold')
    ax[-1].set_xlabel('Longitude')
    ax[-1].xaxis.set_major_formatter(ticker.FuncFormatter(lon_format))
    ax[-1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.subplots_adjust(hspace=0.00)
    plt.show()

    savefig = True
    if savefig == True:
        fig.savefig(f'{event}.EOF.Amplitude.{key}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{event}.EOF.Amplitude.{key}.svg', dpi=300, bbox_inches='tight')

    # add data to csv 
    add = [key, eof_selection['time'].start, eof_selection['time'].stop, ndays] + [np.round(v, 5) for v in phase_speed] + [mean_speed, std_speed]
    rows.append(add)

import csv
# writing to csv files
filename = fr'{event}.EOF.Wave_Speeds.csv'
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(columns)
    csvwriter.writerows(rows)
