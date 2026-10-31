# Command line

Installing fastgrib adds a `fastgrib` command. Without installing, use
`python -m fastgrib` instead.

```bash
fastgrib FILE|URL                                  # the inventory, as wgrib2 -s
fastgrib FILE -match ':TMP:' -not ':0\.' -v        # with units, long names, valid times
fastgrib FILE -select name=TMP level=500 -stats    # min, max, mean, undefined points
fastgrib FILE -grid                                # the grids the fields lie on
fastgrib FILE -match ':HGT:[0-9]+ mb:' -o z.nc     # NetCDF (needs xarray)
fastgrib FILE -bin out.bin                         # float32, west-east and south-north
fastgrib FILE -csv out.csv                         # lon,lat,value for each point
fastgrib -warm                                     # compile every decoder now (installing already does)
```

| Option | Meaning |
|---|---|
| `-match REGEX` | Fields whose inventory line matches, as `wgrib2 -match` |
| `-not REGEX` | Leave out fields that match |
| `-select key=value ...` | Pick by keyword, as `GribFile.select` |
| `-v` | Longer inventory lines: units, long names, valid times |
| `-stats` | Minimum, maximum, mean and number of undefined points |
| `-grid` | Describe the grids |
| `-o FILE.nc` | Write NetCDF through xarray |
| `-bin FILE` | Raw float32, undefined points 9.999e20, as `wgrib2 -order we:sn -no_header -bin` |
| `-csv FILE` | One `lon,lat,value` line per point of a field (pick one with `-match`) |
| `-gpu` | Decode on the GPU |
| `-time` | Report how long it took |
| `-warm` | Compile all decoders now and wait for it. Installing already does this, and fastgrib does it again by itself when they stop matching |
| `-version` | Print the version |

Listing, matching and selecting never decode anything, so numba isn't even
imported. An inventory of GFS 0.25° takes 0.27 s from start to finish, most
of it starting Python and numpy.
