# Writing GRIB2

fastgrib writes three kinds of output: copies of fields as they are, fields
with edited metadata, and new values packed into a field's layout. Messages
are plain `bytes`, so you can write them, send them, or open them again with
`fastgrib.open`.

## Subsets

```python
f.write('temperature.grib2', name='TMP')
f.write('out.grib2', append=True, level='500 mb')
```

Each field picked is written as its own GRIB2 message. A message holding
several fields is split, a bitmap given in an earlier message is made
explicit, and GRIB1 fields are converted to GRIB2, so this converts a GRIB1
file:

```python
fastgrib.open('narr.grb').write('narr.grib2')
```

Remote files can be written the same way, which makes a local subset of a
file on a server.

## A box cut out

```python
f.write('se_us.grib2', box=(-90, -75, 30, 40))       # west, east, south, north
```

Like wgrib2's `-small_grib`, this writes each field on the smaller grid that
covers the box. On a lat-lon or Gaussian grid that is the rows and columns
of the points inside it. A box across the 0° seam of a global grid, or across
the antimeridian, starts at its west edge. On a Mercator, polar
stereographic or Lambert grid it is the smallest rectangle holding every
point in the box. Each field is packed again as it was, with the same
packing and precision, so its values read back unchanged. The grid sections
are the ones wgrib2 3.1.3's `-small_grib` writes.

Cutting the south-east US out of all 173 fields of an HRRR surface file took
0.87 s. wgrib2 took 6.2 s with its default simple packing (twice the size),
and 27.7 s packing as complex, the way fastgrib does. From the command
line, with wgrib2's arguments:

```bash
fastgrib hrrr.grib2 -small_grib -90:-75 30:40 se_us.grib2
```

`read(box=...)`, `fields(box=...)` and `latlon(box=...)` cut the same box
out of arrays.

## Editing metadata

```python
m = f.message(rec, name='DPT', level='700 mb')
m = f.message(rec, fcst='0-6 hour max fcst', date=datetime(2030, 1, 2))
m = f.message(rec, param=(0, 0, 6))        # discipline, category, number
```

`message` changes the parameter, level (in wgrib2's wording, or as
`(type, value[, type2, value2])`), reference time and forecast time
(statistical periods become templates 4.8 or 4.11), leaving the packed values
untouched.

## Encoding new values

```python
m = f.encode(values, like=rec)                           # the field's own packing
m = f.encode(values, rec, packing='complex', decimals=1)
fastgrib.open(m).read(0)                                 # check it
```

`values` is laid out as `read()` returns it, with NaN for missing points
(written as a bitmap). By default the field is packed the way the original
was, to the original's precision, so a field's own values encode back to the
same bits.

| `packing=` | What it is |
|---|---|
| `'simple'` | Template 5.0 |
| `'complex'` | Second-order spatial differencing, its groups found as wgrib2's tightest packer finds them (merged where that saves bits, boundaries moved to the narrower side). 0.90x the size of NCEP's own HRRR files and 0.94x GFS's, within 2% of wgrib2's `-set_grib_type c3` at 150x its speed. |
| `'ieee'` | 32-bit floats, lossless for the float32 values `read()` returns |
| `'png'` | PNG (zlib) |
| `'jpeg'` | JPEG2000, lossless, up to 31 bits |
| `'ccsds'` | CCSDS 121.0, as in ECMWF's open data. About 0.77x GFS's size. |

Spherical-harmonic fields encode with `'simple'` (5.50) or `'complex'`
(5.51), keeping the field's own subset and Laplacian power.

Precision follows the field's own binary and decimal scale factors unless
you set `decimals` or `bits`.

### Speed

One HRRR field of 1.9 million points (2 m temperature) packs in 29 ms
(complex), 6 ms (IEEE), 25 ms (CCSDS), 44 ms (JPEG2000) and 243 ms (PNG,
which is mostly zlib).

## Packing a whole file again

```python
f.write('smaller.grib2', packing='ccsds')            # every field, CCSDS
f.write('t.grib2', name='TMP', packing='complex', decimals=1)
```

This is what wgrib2's `-set_grib_type` does. `write()` with `packing=`,
`decimals=` or `bits=` decodes each field and packs it again, as
`encode()` would on its own. The packers release Python's lock, so the
fields are packed side by side on every core while the next one decodes.
They are written in file order and the bytes are identical to packing each
field by itself. At the field's own precision the values read back
unchanged. `box=` combines with it.

The whole 143 MB HRRR surface file (173 fields of 1.9 million points),
packed again on 24 threads (i7-13700K, the best of 3), against wgrib2
3.1.3's `-set_grib_type` on the same machine:

| `packing=` | fastgrib | wgrib2 | Output, fastgrib / wgrib2 |
|---|---:|---:|---:|
| `'complex'` | 0.87 s | 131.6 s | 128.5 / 125.6 MiB |
| `'ccsds'` | 0.47 s | 8.7 s | 119 / 118 MiB |
| `'jpeg'` | 4.8 s | 27.2 s | 101 / 100 MiB |

Each worker thread decodes its own field, whole, into a row it keeps, then
packs it, so no thread waits on another. The packers keep their working
arrays between fields too: a fresh field-sized array is faulted in one page
at a time, and no number of threads speeds that up.

Many fields packed at once are held up by memory's bandwidth rather than by
the cores, so the packers keep their passes over memory few. Complex
packing makes the integers in chunks that stay in the cache and takes
their differences from there, never storing the integers. After each pass
of its group search, it moves the groups left together, so the next pass
walks along memory rather than jumping about it. Each of these leaves the
bytes the same. The whole HRRR file now packs at 4.9 ms a field on 16
threads, where before 4 threads were as fast as 24 (7.4 ms). `encode()` of
a single field, with memory to itself, makes its integers in one pass.

Complex packing searches for its groups the way wgrib2 does, starting from
blocks of 8 values rather than wgrib2's runs of equal values. That costs
0.4% in size and saves three quarters of the time. NCEP's own file was
143 MB.

From the command line: `fastgrib FILE -grib out.grib2 -packing ccsds`.

### Checked against wgrib2 and ecCodes

Everything fastgrib writes reads back the same in wgrib2 and ecCodes. Every
packing at every bit depth from 1 to 30, with and without missing points,
has been checked against libaec, OpenJPEG and libpng. That includes JPEG2000
above 25 bits, which ecCodes itself writes incorrectly because OpenJPEG's
encoder overflows there.

## The JPEG2000 codec

fastgrib has its own JPEG2000 encoder and decoder, written in numba, with
no C library needed. It decodes any Part-1 codestream the way OpenJPEG does:
every code-block style, all five progression orders, tiles, precincts,
quality layers, the 9/7 wavelet and regions of interest. It is checked
sample for sample against OpenJPEG.

It also handles cases OpenJPEG gets wrong. OpenJPEG crashes encoding 24-bit
terminated passes and loses samples above 25 bits, and Pillow's reader is
wrong above 16 bits. Damaged codestreams raise `ValueError`, which fuzz
testing confirms.
