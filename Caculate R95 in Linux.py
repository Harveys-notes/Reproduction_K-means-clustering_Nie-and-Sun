import os 
import xarray as xr
import numpy as np
from scipy import stats
import netCDF4 as nc

'''Specific year of data: 2001-2019, Apr-Sep'''
time_ls = []
for year in range(2001, 2020):
    for mon in range(4, 10):
        time = str(year) + str(mon).rjust(2,'0')    #set the format 01, 02, 03...
        time_ls.append(time)
        print(time)
        print('--------------------------------------------------')
        
        
'''Read GPM data'''
precip_all = []
for each in time_ls:
    filename = '3B-HHR.MS.MRG.3IMERG.' + each + '.daily.V06B.nc'
    filepath = '/data1/GPM/datasets/daily/'
    precip_data = xr.open_dataset(filepath + filename)
    precip = np.asarray(precip_data.precipitationCal.loc[:,21:34.3,97:109])[:,::-1,:] # 97°–109°E, 21°–34.3°N
    precip_all.append(precip)
    print(each)
    print(precip[:,0,0])
    print('--------------------------------------------------')
latitude = precip_data.lat.loc[21:34.3][::-1]
longitude = precip_data.lon.loc[97:109]


'''Calculate R95p'''
precips_3D = np.concatenate(precip_all, axis=0)  #Dimension: (n*time,lat,lon)
print(precips_3D[0,:,:])
time_number = precips_3D.shape[0]
lat_number = precips_3D.shape[1]
lon_number = precips_3D.shape[2]
sum_grid = np.zeros((lat_number, lon_number))   #Store the sum of every grid, i.e., R95p
for lat in range(lat_number):
    for lon in range(lon_number):
        peR95p = stats.scoreatpercentile(precips_3D[:,lat,lon], 95)
        sum_grid[lat, lon] = sum(precips_3D[:,lat,lon][precips_3D[:,lat,lon]>peR95p])

        
'''Save R95p in NC file'''
print(sum_grid[:,0])
f_w = nc.Dataset('./R95p.nc','w',format = 'NETCDF4')   

f_w.createDimension('lat', sum_grid.shape[0])   
f_w.createDimension('lon', sum_grid.shape[1])  

lat = f_w.createVariable('lat', np.float64, ('lat'))  
lon = f_w.createVariable('lon', np.float64, ('lon'))
R95p = f_w.createVariable('R95p', np.float64, ('lat','lon'))

lat.units = 'degree'
lon.units = 'degree'
R95p.units = 'mm'

f_w.variables['lat'][:] = latitude
f_w.variables['lon'][:] = longitude
f_w.variables['R95p'][:,:] = sum_grid

f_w.close()