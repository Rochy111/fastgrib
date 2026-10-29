"""Sea-level pressure and 500 hPa heights over the North Atlantic, read straight
from NOAA's GFS archive: fastgrib reads the file's index and fetches only the
two fields it needs (about 1.8 MB of a 500 MB file). Drawn with fastplot."""
import os

import numpy as np

import fastgrib
import fastplot.crs as ccrs
import fastplot.feature as cfeature
import fastplot.pyplot as plt

GFS = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261027/00/atmos/gfs.t00z.pgrb2.0p25.f006'
grib = fastgrib.open(os.environ.get('GFS_FILE', GFS))

mslp = grib.read('PRMSL:mean sea level') / 100.0
z500 = grib.read('HGT:500 mb') / 10.0
lat, lon = grib.latlon()
lat, lon = lat[:, 0], lon[0]
data = ccrs.PlateCarree()

fig = plt.figure(figsize=(9, 8), dpi=150)
ax = fig.add_subplot(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-35,
                                                               central_latitude=50))
ax.set_extent([-80, 15, 22, 75], crs=data)

shaded = ax.contourf(lon, lat, mslp, levels=np.arange(972, 1045, 4), cmap='PuOr_r',
                     extend='both', transform=data)
heights = ax.contour(lon, lat, z500, levels=np.arange(480, 600, 6), colors='black',
                     linewidths=0.9, transform=data)
ax.clabel(heights, fmt='%d', fontsize=7)
ax.coastlines(linewidth=0.7, color='#222222')
ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#555555')
ax.gridlines(linewidth=0.3, color='#666666', linestyle='--')

cb = fig.colorbar(shaded, ax=ax, orientation='horizontal', shrink=0.75, pad=0.03, aspect=40)
cb.set_label('Sea-level pressure (hPa)')
ax.set_title('500 hPa height (dam) over sea-level pressure', loc='left', fontsize=11)
ax.set_title('GFS  valid 06 UTC 27 Oct 2026', loc='right', fontsize=9, color='#555555')

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images',
                         'remote_fields.png'), bbox_inches='tight')
