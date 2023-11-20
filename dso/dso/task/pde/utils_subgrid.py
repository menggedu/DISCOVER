import numpy as np
import xarray as xr
from scipy.stats import pearsonr
import re



"""
data = q,u,v,ufull,vfull
coordinates = 
"""

class Subgrid_forcing:

    def __init__(self,dataset):
        '''
        dataset: xrarray dataset
        '''
        # preload necessary data
        
        self.m = dataset
        k, l = np.meshgrid(self.m.k, self.m.l)
    
        self.ik = 1j * k
        self.il = 1j * l

        self.nx = self.ik.shape[0]
        self.wv2 = self.ik**2 + self.il**2

    def fft(self, x):
        try:
            return self.m.fft(x)
        except:
            # Convert to data array
            dims = [dict(y='l',x='k').get(d,d) for d in self['q'].dims]
            coords = dict([(d, self[d]) for d in dims])
            return xr.DataArray(np.fft.rfftn(x, axes=(-2,-1)), dims=dims, coords=coords)

    def ifft(self, x):
        try:
            return self.m.ifft(x)
        except:
            return self['q']*0 + np.fft.irfftn(x, axes=(-2,-1))
    
    def is_real(self, arr):
        return len(set(arr.shape[-2:])) == 1
    
    def real(self, arr):
        arr = self[arr]
        if isinstance(arr, float): return arr
        if self.is_real(arr): return arr
        return self.ifft(arr)
    
    def compl(self, arr):
        arr = self[arr]
        if isinstance(arr, float): return arr
        if self.is_real(arr): return self.fft(arr)
        return arr
    
    # Spectral derivatrives
    def ddxh(self, f): return self.ik * self.compl(f)
    def ddyh(self, f): return self.il * self.compl(f)
    def divh(self, x, y): return self.ddxh(x) + self.ddyh(y)
    def curlh(self, x, y): return self.ddxh(y) - self.ddyh(x)
    def laplacianh(self, x): return self.wv2 * self.compl(x)
    def advectedh(self, x_):
        x = self.real(x_)
        return self.ddxh(x * self.m.ufull) + self.ddyh(x * self.m.vfull)

    # Real counterparts
    def ddx(self, f): return self.real(self.ddxh(f))
    def ddy(self, f): return self.real(self.ddyh(f))
    def laplacian(self, x): return self.real(self.laplacianh(x))
    def advected(self, x): return self.real(self.advectedh(x))
    def curl(self, x, y): return self.real(self.curlh(x,y))
    def div(self, x, y): return self.real(self.divh(x,y))
    
    def __getitem__(self, q):
        if isinstance(q, str):
            return getattr(self.m, q)
        elif any([isinstance(q, kls) for kls in [xr.DataArray, np.ndarray, int, float]]):
            return q
        else:
            raise KeyError(q)




ds_path = './dso/task/pde/data/eddy_forcing1_run=0_scipy.nc'
lev=0
ds = xr.open_dataset(ds_path)
# .isel(lev=lev)
# ds = netcdf.NetCDFFile(ds_path,'r')
extractor = Subgrid_forcing(ds)
# import pdb;pdb.set_trace()

# def make_subgrid_functions(ds, lev = 0):
def apply_spatial(func, x):
    r = func(x.reshape(ds.q.shape))
    if isinstance(r, xr.DataArray): r = r.data
    return r.reshape(x.shape)

ddx = lambda x: apply_spatial(extractor.ddx, x)
ddy = lambda x: apply_spatial(extractor.ddy, x)
laplacian = lambda x: apply_spatial(extractor.laplacian, x)
adv = lambda x: apply_spatial(extractor.advected, x)


test_ds_path = './dso/task/pde/data/eddy_forcing1_run=1_scipy.nc'
ds_test = xr.open_dataset(test_ds_path)
extractor_test =Subgrid_forcing(ds_test)

ddx_t = lambda x: apply_spatial(extractor_test.ddx, x)
ddy_t = lambda x: apply_spatial(extractor_test.ddy, x)
laplacian_t = lambda x: apply_spatial(extractor_test.laplacian, x)
adv_t = lambda x: apply_spatial(extractor_test.advected, x)


        