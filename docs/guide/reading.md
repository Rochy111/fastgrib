# Reading files

## Opening

```python
f = fastgrib.open('file.grib2')          # a path
f = fastgrib.open('https://...')         # a URL (see Remote files)
f = fastgrib.open(message_bytes)         # bytes already in memory
```

Opening only scans the file's section headers, so it is quick: listing a
GFS 0.25° file takes about 10 ms. GRIB1 files, and files mixing both
editions, open the same way. Use `f.close()` to release the file's memory
straight away, or `with fastgrib.open(...) as f:`.

## Picking fields

Every method that reads, regrids or writes takes the same ways of picking
fields.

**A regular expression** matched against each field's inventory line, as
`wgrib2 -match` does:

```python
f.read(':TMP:2 m above ground:')
f.read(':(UGRD|VGRD):[0-9]+ mb:')
```

**Keywords**, as pygrib's `select`. Each takes a value, a list of values or a
function:

```python
f.read(name='TMP', level=500)
f.read(name=['UGRD', 'VGRD'], level_type='isobaricInhPa', level=lambda v: v >= 200)
f.read(short_name='2t')                          # ecCodes' shortName
f.read(fcst_hours=6, member=0)
```

**A position or a Record**, such as `f.read(0)` or `f.read(f.records[10])`, or
a list of any of these.

One field comes back as `(ny, nx)` and several as `(n, ny, nx)`. To see what
matches without reading anything, use `f.find(pattern)` or
`f.select(**keywords)`, which return Records.

## Records

`f.records` holds one `Record` per field. Each one describes its field
without decoding it:

| Attribute | Example |
|---|---|
| `name`, `long_name`, `units` | `'TMP'`, `'Temperature'`, `'K'` |
| `short_name` | `'2t'` (ecCodes' shortName) |
| `level` | `'2 m above ground'` (wgrib2's wording) |
| `level_kind` | `'heightAboveGround'` (ecCodes' typeOfLevel) |
| `level_type`, `level_value` | `103`, `2.0` (GRIB2's code and value) |
| `ref_time`, `valid_time` | `datetime` objects |
| `fcst_hours` | `6` |
| `member`, `ensemble` | `3`, or `'ens mean'` for statistics |
| `grid` | the field's `Grid` |

`rec.as_dict()` gives all of them at once. Names come from NCEP's parameter
tables, then ecCodes' for anything NCEP doesn't define (ECMWF's local
parameters, for instance).

## Grids

```python
g = f.grid()              # .shape, .kind, .params, .earth
lat, lon = f.latlon()     # every point's latitude and longitude
```

Fields always come out west to east and south to north (wgrib2's
`-order we:sn`), whatever scanning order the file uses. Longitudes are in
[0, 360), as wgrib2 prints them.

## Many fields

`f.fields(pick)` decodes one field at a time into a fixed ring of buffers,
so memory stays flat however many fields you go through:

```python
for rec, field in f.fields(':UGRD:'):
    print(rec.level, field.max())
```

To reuse your own array between reads, pass `out=`:

```python
buf = np.empty(f.grid().shape, np.float32)
for rec in f.find(':TMP:'):
    f.read(rec, out=buf)
```

## GRIB1

A GRIB1 message reads as the GRIB2 field it would be. NCEP's parameters use
its ON388 tables, ECMWF's use ecCodes' with the level and statistic they
imply (`2t` at 2 m, `tp` accumulated from the start). Ensembles, predefined
grids that carry no grid description (WMO's and NCEP's Table B), spherical
harmonics, second-order packing and matrices of values (wave spectra) are
all handled.

GRIB1 messages are converted in batches as they are read: opening NARR's
56 MB analysis takes 0.03 s, and reading all 423 fields takes another
0.04 s.

## Damaged files

A file that is cut short, has bytes missing, has rubbish between messages
or a wrong length still gives up every intact field. Each damaged message
is skipped and described in `f.damaged`, with one warning. Corrupt packed
data raises `ValueError`. It never crashes or reads past the end of the
data, on the CPU or the GPU; this has been tested with tens of thousands of
deliberately corrupted messages.

Some unusual but valid layouts are handled too:

- **Several fields in one message** (wgrib2's `-submsg_uv` and others): each
  becomes its own field, numbered `5.1`, `5.2` as wgrib2 numbers them.
- **Bitmaps given in an earlier message** (indicator 254), including when
  only that field is fetched from a remote file.
- **Files larger than 4 GB**, every scanning order, and many threads reading
  at once.
