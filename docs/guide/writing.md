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
| `'complex'` | Second-order spatial differencing, with groups merged where it saves space. About 0.8x the size of NCEP's own NBM files, 1.07x on GFS. |
| `'ieee'` | 32-bit floats, lossless for the float32 values `read()` returns |
| `'png'` | PNG (zlib) |
| `'jpeg'` | JPEG2000, lossless, up to 31 bits |
| `'ccsds'` | CCSDS 121.0, as in ECMWF's open data. About 0.77x GFS's size. |

Spherical-harmonic fields encode with `'simple'` (5.50) or `'complex'`
(5.51), keeping the field's own subset and Laplacian power.

Precision follows the field's own binary and decimal scale factors unless
you set `decimals` or `bits`.

### Speed

NBM's 3.7 million points pack in 32 to 70 ms (complex), 20 ms (IEEE), 34 ms
(CCSDS), 64 ms (JPEG2000) and 100 to 400 ms (PNG, which is mostly zlib).

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
