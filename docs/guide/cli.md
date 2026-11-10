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
fastgrib URL -match ':TMP:' -grib tmp.grib2        # the fields picked, as GRIB2 (wgrib2 -grib)
fastgrib FILE -small_grib -90:-75 30:40 box.grib2   # cut to a box (wgrib2 -small_grib)
fastgrib URL -max_records 50 > part.idx            # part of a file with no index, its index saved
fastgrib URL -index part.idx -match ':HGT:'        # a remote file through an index given
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
| `-grib FILE` | Write the fields picked as GRIB2, each message standing alone, as `wgrib2 -grib`; from a URL only those fields are fetched |
| `-packing TYPE` | With `-grib` or `-small_grib`: pack the fields again as `simple`, `complex`, `ccsds`, `jpeg`, `png` or `ieee`, on every core (`wgrib2 -set_grib_type`) |
| `-small_grib W:E S:N FILE` | Write the fields picked cut to a box of longitudes W to E and latitudes S to N, as `wgrib2 -small_grib` |
| `-max_records N` | Only the first N fields: a remote file with no index fetches only those |
| `-byte_range START:STOP` | Only the messages starting in these bytes (`START:` for the rest) |
| `-index IDX` | A remote file's index, a path or URL, in place of the one beside it |
| `-gpu` | Decode on the GPU |
| `-time` | Report how long it took |
| `-warm` | Compile all decoders now and wait for it. Installing already does this, and fastgrib does it again by itself when they stop matching |
| `-version` | Print the version |

Listing, matching and selecting never decode anything, so numba isn't even
imported. An inventory of GFS 0.25° takes 0.27 s from start to finish, most
of it starting Python and numpy.
