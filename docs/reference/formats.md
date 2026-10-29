# What it reads

## GRIB2

| | |
|---|---|
| Grids (3.x) | 0 lat-lon, 1 rotated lat-lon, 10 Mercator, 20 polar stereographic, 30 Lambert conformal, 40 Gaussian, 41 rotated Gaussian (regular, reduced or staggered); 50 to 53 spherical harmonics, including rotated and stretched; 90 space view (any position, rotated, orthographic); 130 listed points; NCEP's 32768 and 32769 rotated E- and B-grids; every scanning mode; spherical earths, and ellipsoidal ones through pyproj. Any other grid (101 unstructured, 204 curvilinear) reads as values without positions. |
| Products (4.x) | 4.0 to 4.15: analyses and forecasts, ensemble members and statistics, clusters, probabilities, percentiles, statistics over time and space; chemicals and aerosols 4.40 to 4.49; 4.60 and 4.61 re-forecasts. Described as wgrib2 describes them. |
| Packing (5.x) | 0 simple; 2 and 3 complex (spatial differencing, missing values); 4 IEEE (32 and 64 bits); 40 JPEG2000 (any Part-1 codestream, up to 31 bits); 41 PNG; 42 CCSDS (AEC); 50 and 51 spherical harmonics; 200 run-length (JMA) |
| Bitmaps | Present, absent, or given in an earlier message (254) |
| Parameters | NCEP's tables (1,757 parameters), then ecCodes' (WMO, ECMWF and NCEP local, legacy), with chemicals and aerosols named by constituent, size and wavelength (`aod`, `pm2p5`) |

## GRIB1

| | |
|---|---|
| Grids | 0, 1, 3, 4, 5, 10, 14 (including thinned), 50, 60, 70, 80 (spherical harmonics), 90 (space view), 203 and 205 (NCEP's E- and B-grids), and predefined grids with no grid description: WMO's 21 to 26 and 61 to 64, and 67 of NCEP's ON388 Table B grids (211, 212, 218, 221, 2, 3, 4 and more), which ecCodes and pygrib can't place |
| Packing | Simple; every form of second-order packing (ECMWF's general extended, with spatial differencing and boustrophedonic rows; WMO's row by row, constant width and general); spectral simple and complex |
| Other | Bitmaps; matrices of values at each point (wave spectra), each element its own field; NCEP's ensemble extension and ECMWF's local definition 1 |
| Parameters | NCEP's tables 2 and 128 to 141, ECMWF's (3,954 parameters) |

Remote GRIB1 files are read through their wgrib inventory (`.inv`), as NCEI
publishes for NARR.

## Not supported

These are left out on purpose:

- Grids whose points live outside the message (3.101's grid file, 3.204's
  coordinate fields) are read as values only.
- GRIB1's 201 and 202 E-grids carry no rotation centre and are not placed.
  NCEP has retired them.
- JPEG2000 codestreams with progression-order changes or packed packet
  headers (POC, PPM and PPT markers), which no GRIB writer uses.
- GRIB1 second-order packing of a matrix of values, which nothing writes
  and ecCodes doesn't read either.

Anything not covered raises a clear error rather than guessing.
