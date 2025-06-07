# Equatorial Wave Filtering

# --- Import libraries
import time
import sys
import numpy as np
from scipy import signal
import xarray as xr
import matplotlib.pyplot as plt
import dask.array as da
import netCDF4

# --- Create a class for wave filtering
class WaveFilter:
    def __init__(self, path, var, coverage, wave_name, n, units):
        self.path       = path
        self.var        = var
        self.coverage  = coverage
        self.wave_name  = wave_name
        self.long_name  = None
        self.n          = n          

        self.ds         = xr.open_dataset(path,chunks={'time': 'auto'})  # 使用 dask 分块加载
        self.data       = None
        self.units      = units

        self.filtered_data = None
        self.fftdata       = None

    def load_data(self):
        """Load and preprocess the data"""
        var = self.var      # data variable

        # adjust the dimension
        if "latitude" in self.ds.coords:
            self.ds = self.ds.rename({'latitude':'lat'})
        if "longitude" in self.ds.coords:
            self.ds = self.ds.rename({'longitude':'lon'})
        
        # load the data
        self.data = self.ds[var].sel(**self.coverage).sortby('lat').transpose('time', 'lat', 'lon')
        self.data = self.data.fillna(0)             # fill NaN value with 0 to avoid error
        self.data = self.data.chunk({'time': -1})   # chunk the data for parallel processing
        
    def detrend_data(self):
        """Detrend the data using dask for parallel processing."""
        ntim, nlat, nlon = self.data.shape
        spd = 1 # number of sample per day
        #data_rechunked = self.data.data.rechunk({0: -1})
        data_rechunked = self.data.data.rechunk({0: -1, 2: -1})
        if ntim >  365*spd/3:
            # FFT
            rf   = da.fft.rfft(data_rechunked, axis=0)
            freq = da.fft.rfftfreq(ntim * spd, d=1. / float(spd))
            rf[(freq <= 3. / 365) & (freq >= 1. / 365), :, :] = 0.0
            datain = da.fft.irfft(rf, axis=0, n=ntim)
 
        # detrend the data
        self.detrend = da.apply_along_axis(signal.detrend, 0, datain)    
        # apply window function 
        window = signal.windows.tukey(ntim,0.05,True)
        self.detrend = self.detrend * window[:, np.newaxis, np.newaxis]   

    def fft_transform(self):
        """Perform 2D FFT on the detrended data using dask."""
        self.wavenumber = -da.fft.fftfreq(self.data.shape[2]) * self.data.shape[2]   # shape: (lon,)
        self.frequency  = da.fft.fftfreq(self.data.shape[0], d=1./float(1))          # shape: (time,)

        self.knum_ori, self.freq_ori = da.meshgrid(self.wavenumber, self.frequency)  # shape: (time, lon)
        self.knum = self.knum_ori.copy()
        self.knum = da.where(self.freq_ori < 0, -self.knum_ori, self.knum_ori)       # shape: (time, lon)
        
        self.freq = da.abs(self.freq_ori)    # shape: (time, lon)
    
    def apply_filter(self):
        """Apply filter based on wave type."""
        if self.wave_name.lower() == "kw":
            self.tMin, self.tMax = 20, 180
            self.kmin, self.kmax = 1, 14
            self.hmin, self.hmax = None, None #0.025, 90
        elif self.wave_name.lower() == "er":
            self.tMin, self.tMax = 120, 450
            self.kmin, self.kmax = -10, -1
            self.hmin, self.hmax = None, None #0.003, 90
        
        self.fmin, self.fmax = 1 / self.tMax, 1 / self.tMin
        self.mask =  da.zeros((self.data.shape[0], self.data.shape[2]), dtype=bool)

        if self.kmin is not None:
            self.mask = self.mask | (self.knum < self.kmin)
        if self.kmax is not None:
            self.mask = self.mask | (self.kmax < self.knum)

        if self.fmin is not None:
            self.mask = self.mask | (self.freq < self.fmin)
        if self.fmax is not None:
            self.mask = self.mask | (self.fmax < self.freq)

        if self.wave_name.lower() == 'kw':
            self.apply_wave_filter(self.wave_name)
        elif self.wave_name.lower() == 'er':
            self.apply_wave_filter(self.wave_name)
            
        self.fftdata = da.fft.fft2(self.detrend, axes=(0, 2)) # shape: (time, lat, lon)
        self.mask    = da.repeat(self.mask[:, np.newaxis, :], self.data.shape[1], axis=1)
        self.fftdata = da.where(self.mask, 0.0, self.fftdata)

    def apply_wave_filter(self, wave_name):
        """Apply equtorial wave filter."""
        # parameters
        g    = 9.8
        beta = 2.28e-11
        a    = 6.37e6
        n    = self.n

        if self.wave_name.lower() == "kw":
            
            if self.hmin is not None:
                c      = da.sqrt(g * self.hmin)
                omega  = 2. * np.pi * self.freq / 24. / 3600. / da.sqrt(beta * c)
                k      = self.knum / a * da.sqrt(c / beta)
                self.mask = self.mask | (omega - k < 0)
            if self.hmax is not None:
                c      = da.sqrt(g * self.hmax)
                omega  = 2. * np.pi * self.freq / 24. / 3600. / da.sqrt(beta * c)
                k      = self.knum / a * da.sqrt(c / beta)
                self.mask = self.mask | (omega - k > 0)
    
        if self.wave_name.lower() == "er":
    
            if self.hmin is not None:
                c = da.sqrt(g * self.hmin)
                omega = 2. * np.pi * self.freq / 24. / 3600. / da.sqrt(beta * c)
                k = self.knum / a * da.sqrt(c / beta)
                self.mask = self.mask | (omega * (k ** 2 + (2 * n + 1)) + k < 0)
            if self.hmax is not None:
                c = da.sqrt(g * self.hmax)
                omega = 2. * np.pi * self.freq / 24. / 3600. / da.sqrt(beta * c)
                k = self.knum / a * da.sqrt(c / beta)
                self.mask = self.mask | (omega * (k ** 2 + (2 * n + 1)) + k > 0)
         
    def inverse_fft(self):
        """Perform inverse FFT to get the filtered data."""
        self.filtered_data = da.fft.ifft2(self.fftdata, axes=(0, 2)).real

    def create_output(self):
        """Create xarray DataArray for filtered data."""

        if self.wave_name == 'KW':
            self.long_name = 'Kelvin Waves'
        elif 'ER' in self.wave_name:
            self.long_name = 'Equatorial Rossby Waves'
        else:
            self.wave_name = None
        
        self.wave_data = xr.DataArray(self.filtered_data.compute(),
                                      coords = {'time': self.data.time,
                                                'lat' : self.data.lat,
                                                'lon' : self.data.lon},
                                      dims=['time', 'lat', 'lon'])
        self.wave_data.attrs.update({
            'name'           : self.wave_name,
            'long_name'      : self.long_name,
            'min_wavenumber' : self.kmin,
            'max_wavenumber' : self.kmax,
            'min_period'     : self.tMin,
            'max_period'     : self.tMax,
            'min_frequency'  : self.fmin,
            'max_frequency'  : self.fmax,
            'units'          : self.units,
        })
        
        self.ds.close()
        return self.wave_data
        
# --- Main program 
start=time.time()
print("BEGIN")
print(sys.version)

# Data Information
years = "2017-2024"
var = "sla"
path = f"/kaggle/input/sea-level-anomalies-noaa/sla.2017-2024.noaa.nc"

# Region of Interest
coverage = {
    'lat' : slice(-10, 10), 
    'lon' : slice(120, 280)
}

# Wave Types
EquWaves = {
    'KW'  : {'wave_name':'KW', 'n':0, 'data':[]}, 
    'ER' : {'wave_name':'ER', 'n':1, 'data':[]}
}

# Loop through each wave type and apply the filter
EWs = []
for i, key in enumerate(EquWaves.keys()):
    wave_name = EquWaves[key]['wave_name']
    n         = EquWaves[key]['n']
    print(f"{i+1}: {wave_name}, n = {n}")

    # create a wave filter object
    wave_filter = WaveFilter(path, var, coverage, wave_name, n, units='m')

    # filtering process
    wave_filter.load_data()  
    wave_filter.detrend_data()
    wave_filter.fft_transform()
    wave_filter.apply_filter()
    wave_filter.inverse_fft()

    # store the data
    EquWaves[key]['data'] = wave_filter.create_output()

    print(f"{i+1}/{len(EquWaves)} Completed")

# --- Create a dataset
# Extract the dimension
t    = EquWaves['KW']['data'].time
lat  = EquWaves['KW']['data'].lat
lon  = EquWaves['KW']['data'].lon

# Initialize the dataset
waves = xr.Dataset(
    data_vars={
        key: (['time', 'lat', 'lon'], EquWaves[key]['data'].values, EquWaves[key]['data'].attrs)
        for key in EquWaves if isinstance(EquWaves[key]['data'], xr.DataArray)
    },
    coords=dict(
        time=t,
        lat=lat,
        lon=lon,
    ),
    attrs=dict(
        description=f'Equatorial oceanic Kelvin and Rossby waves based on the sea level data of {years}',
        unit='m'
    )
)

# Export the data
comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in waves.data_vars}
waves.to_netcdf(
    f'EW.SLA.{years}.noaa.nc',
    engine='netcdf4',
    encoding=encoding
)
waves.close()

end = time.time()
print("Runtime: %8.1f seconds." % (end-start))
print('DONE!')
