# fastgrib

[![tests](https://github.com/Rochy111/fastgrib/actions/workflows/tests.yml/badge.svg)](https://github.com/Rochy111/fastgrib/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-rochy111.github.io%2Ffastgrib-teal)](https://rochy111.github.io/fastgrib/)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![licence](https://img.shields.io/badge/licence-MIT-green)

GRIB2 and GRIB1 for Python. Open a file or a URL, pick fields, and get
numpy arrays that are **bit-for-bit what wgrib2 gives**, decoded on every
CPU core with numba or on an NVIDIA GPU. It also regrids fields, reads any
key under ecCodes' names, and writes GRIB2 back.

![500 hPa height over sea-level pressure, read from NOAA's GFS archive](docs/images/remote_fields.png)

**[Documentation](https://rochy111.github.io/fastgrib/)** ·
[Getting started](https://rochy111.github.io/fastgrib/getting-started/) ·
[Gallery](https://rochy111.github.io/fastgrib/gallery/)

## Install

```bash
pip install git+https://github.com/Rochy111/fastgrib@v0.2.0
fastgrib -warm               # compile the decoders once, 20 to 30 s
```

Only numpy and numba are required. The `all` extra adds GPU decoding (cupy),
xarray and NetCDF output, and pyproj. Python 3.9+, numpy 1 or 2, Linux,
Windows and macOS.

## Example

```python
import fastgrib

f = fastgrib.open('gfs.t00z.pgrb2.0p25.f006')    # a path, an http(s):// or s3:// URL, or bytes
print(f)                                         # the inventory, as wgrib2 -s lists it
t2m = f.read('TMP:2 m above ground')             # (ny, nx) float32, NaN where undefined
z500 = f.read(name='HGT', level=500)             # pygrib-style keywords
tp = f.read(short_name='tp')                     # or ecCodes' short names
lat, lon = f.latlon()                            # rows south to north
u, v = f.winds(level=850)                        # east and north, whatever the grid
ds = f.to_xarray()                               # every field as an xarray Dataset
f.write('subset.grib2', name='TMP')              # fields picked, as GRIB2
```

With a URL, fastgrib reads the index published beside the file and downloads
only the fields you read.

## Speed

Milliseconds per field, decoding every field of a file (i7-13700K, WSL2):

| File | pygrib | wgrib2, 1 thread | fastgrib, 1 core | fastgrib, all cores |
|---|---:|---:|---:|---:|
| GFS 0.25°, complex packing | 7.32 | 2.67 | 1.25 | 0.28 |
| HAFS, complex packing with bitmaps | 23.8 | 5.02 | 2.93 | 0.47 |
| ECMWF IFS 0.25°, simple packing | 3.55 | 1.43 | 0.87 | 0.25 |
| ECMWF open data, CCSDS | 11.9 | 10.1 | 4.28 | 0.52 |
| RAP 13 km, JPEG2000 | 10.1 | 8.76 | 4.75 | 0.98 |

On a GPU, an NBM field of 3.7 million points decodes in 0.1 ms if it stays
on the card. See the [speed page](https://rochy111.github.io/fastgrib/speed/).

## What's included

- **Reading:** GFS, GEFS, HRRR, NAM, RAP, NBM, MRMS, ECMWF and GRIB1 archives.
  Simple, complex, IEEE, JPEG2000, PNG, CCSDS and run-length packing.
  Damaged files give up every intact field.
- **Remote files:** HTTP and S3, fetching only the fields you need.
- **Regridding:** onto lat-lon grids, another file's grid, or points, with
  winds turned to east and north.
- **Unusual grids:** spherical harmonics synthesised into maps, satellite
  views, staggered and rotated E- and B-grids.
- **Writing:** subsets, edited metadata, and new values in any GRIB2 packing,
  readable by wgrib2 and ecCodes.
- **Keys:** any field's keys under ecCodes' names.
- **Command line:** `fastgrib FILE -match ':TMP:' -stats`, NetCDF, binary and
  CSV output.

## Accuracy

Values match wgrib2 3.1.3 bit for bit on GFS, GEFS, NBM, MRMS and ECMWF
data. Inventory lines match NOAA's `.idx` files line for line, and keys
match ecCodes' on about 17,000 checks. `pytest` repeats these checks
against stored wgrib2 and ecCodes output, without needing either installed.
Details are on the [accuracy page](https://rochy111.github.io/fastgrib/reference/accuracy/).

## Related

[fastplot](https://github.com/Rochy111/fastplot) draws maps from the arrays
fastgrib reads, with matplotlib's and cartopy's API, in milliseconds.

## Contributing and licence

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [changelog](CHANGELOG.md).

MIT. The ecCodes-derived tables are under the Apache License 2.0
([LICENSE-ecCodes](LICENSE-ecCodes)), and WMO's code table C-14 under the MIT
licence ([LICENSE-WMO-CCT](LICENSE-WMO-CCT)). See [NOTICE](NOTICE).
