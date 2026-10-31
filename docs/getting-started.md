# Getting started

## Installation

fastgrib needs Python 3.9 or newer and installs with only numpy and numba.

```bash
pip install git+https://github.com/Rochy111/fastgrib@v0.2.1     # the 0.2.1 release
pip install -U git+https://github.com/Rochy111/fastgrib         # the latest on main, or upgrade to it
```

The `all` extra adds GPU decoding (cupy, CUDA 12), xarray and NetCDF output,
and pyproj for ellipsoidal grids:

```bash
pip install "fastgrib[all] @ git+https://github.com/Rochy111/fastgrib"
```

From a clone, use `pip install -e .` or `pip install -e ".[all]"`. It is
tested on Linux, Windows and macOS with numpy 1 and 2. Without an NVIDIA GPU,
everything runs on the CPU.

## Compiling the decoders

fastgrib's decoders are compiled by numba for your exact CPU. `pip install`
does this for you, which adds about a minute to the install, so there is
nothing to run afterwards and the first field is already at full speed. (The
0.2.1 release predates this: after installing that version, run
`fastgrib -warm` once.)

Each new process spends about half a second on its first decode, starting
numba and loading the compiled decoders. After that, a 2345 x 1597 field
takes a few milliseconds. Keep this in mind when timing a handful of fields
in a fresh process.

If the compiled decoders stop matching, because numba was upgraded or the
install was copied to a different CPU, fastgrib notices at the first decode
and compiles them again in a low-priority background process, which takes
about a minute. Fields still read correctly while that runs, and the first
of each packing type takes a few seconds longer. An editable install
(`pip install -e`), a wheel from a GitHub release and any numba older than
0.68 get their decoders this way too.

In a container image built on one machine and run on others, the same thing
happens on the first run on each new CPU. To avoid that, run `fastgrib -warm`
in the container at start-up, which waits until it is done, or set
`NUMBA_CPU_NAME=generic` both when building and when running, at the cost of
some speed.

| Environment variable | What it does |
|---|---|
| `FASTGRIB_AUTOWARM` | `0` stops fastgrib compiling its decoders in the background when they don't match this machine |
| `FASTGRIB_BUILD_KERNELS` | `0` while installing skips compiling the decoders, for a faster install and a slower first run |

## Opening a file

```python
import fastgrib

f = fastgrib.open('gfs_subset.grib2')        # five fields cut from a GFS 0.25° file
print(f)
```

```text
1:0:d=2026102700:TMP:2 m above ground:6 hour fcst:
2:870962:d=2026102700:PRMSL:mean sea level:6 hour fcst:
3:1856691:d=2026102700:HGT:500 mb:6 hour fcst:
4:2670201:d=2026102700:UGRD:10 m above ground:6 hour fcst:
5:3248921:d=2026102700:VGRD:10 m above ground:6 hour fcst:
```

The inventory lines are exactly what `wgrib2 -s` prints, so a regular
expression you use with `wgrib2 -match` works here too.

## Reading fields

```python
t2m = f.read('TMP:2 m above ground')             # one field: (ny, nx) float32
winds = f.read(':(UGRD|VGRD):850 mb:')           # several: (n, ny, nx)
z500 = f.read(name='HGT', level=500)             # pygrib-style keywords
tp = f.read(short_name='tp')                     # ecCodes short names
lat, lon = f.latlon()                            # the grid's coordinates
```

Undefined points come back as NaN. Every grid is returned west to east and
south to north, whatever order the file stores it in.

To go through many fields without holding them all in memory:

```python
for rec, field in f.fields(':TMP:'):
    print(rec.level, rec.valid_time, field.mean())
```

## Command line

```bash
fastgrib gfs.t00z.pgrb2.0p25.f006 -match ':TMP:' -stats
fastgrib gfs.t00z.pgrb2.0p25.f006 -match ':HGT:[0-9]+ mb:' -o heights.nc
```

See [Command line](guide/cli.md) for every option.

## Next steps

- [Reading files](guide/reading.md) covers picking fields, records and grids in detail.
- [Remote files](guide/remote.md) shows how to read from NOAA's and ECMWF's servers.
- [Regridding and winds](guide/regridding.md) puts fields onto other grids.
- [Writing GRIB2](guide/writing.md) covers subsets, edits and encoding.
