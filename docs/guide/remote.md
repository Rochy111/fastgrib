# Remote files

Give `fastgrib.open` an `http://`, `https://` or `s3://` address and it
works with the file where it is.

```python
url = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261027/00/atmos/gfs.t00z.pgrb2.1p00.f006'
f = fastgrib.open(url)
print(f)                                  # from the index: nothing downloaded yet
z500 = f.read(':HGT:500 mb:')             # about 0.05 s: one field fetched
everything = f.read()                     # 743 fields, 45 MB: about 1.2 s
```

## How it works

Operational centres publish an index next to each file: NOAA's `.idx`,
ECMWF's `.index`, or the `.inv` beside GRIB1 archives such as NCEI's NARR.
fastgrib reads that index first, so `print(f)`, `find` and `select` work
before any data is downloaded, with the same names and levels the fields
have once fetched.

When you read, it requests only the byte ranges of the fields you picked,
several at once. Connections are kept open between requests, since a new
one costs about a tenth of a second to set up (except through a proxy). A
field whose bitmap lives in an earlier message gets that message fetched
along with it.

If there is no index, the whole file is downloaded once, as several range
requests at a time when the server takes them (one connection to S3 often
slows to 10 MB/s; eight held 20-47 MB/s on a 40 MB/s line).

`at()`, `regrid()` onto a few points and `read(box=...)` go further: they
fetch only the part of each field the points need. That works in every
packing: the byte ranges of the values for simple and IEEE, a prefix of
the stream for complex, CCSDS and PNG, and the headers and needed
code-blocks for JPEG2000. One point from 24 HRRR JPEG2000 fields fetched
9% of their bytes, and three points of a simple-packed 0.25° field 13 KB
of its 1.2 MB. See
[Regridding](regridding.md#only-the-points-needed).

## Without an index: part of a file

To list or read only part of a file that has no index, say how much you want:

```python
f = fastgrib.open(url, max_records=20)                  # its first 20 fields
f = fastgrib.open(url, byte_range=(100_000_000, 140_000_000))   # the messages starting there
```

fastgrib then fetches 8 MB, then twice as much each time, each window as
several range requests at once, until it has the fields asked for. Each
message's length tells it where the next one starts. It never makes one
request per message, which would cost a round trip for each. The first 20
fields of a 540 MB GFS file took 1.2 s from S3. The records keep their
offsets in the whole file. Fields picked from part of a file are numbered
from 1.

Both options also work with an index (they pick among its entries) and on
local files.

## Keeping fetched fields on disk

```python
f = fastgrib.open(url, cache=True)        # or cache='/data/grib-cache'
```

With `cache=`, each message fetched is kept in a folder, one file per
message. Any later read of it, in this process or another one, comes from
disk instead. A script run again, or a pool of workers reading the same
fields, fetches each field once. The index is still fetched each time, so
new messages in a file that is still being written are seen. A damaged or
cut-short copy on disk is fetched again. `cache=True` uses
`FASTGRIB_REMOTE_CACHE`, or else `fastgrib/remote` in your user cache
folder. Nothing is ever deleted from it, so clear it when you like.

## Saving an index

A file with no index only has to be scanned once:

```python
f = fastgrib.open(url)                    # or max_records=..., byte_range=...
f.write_index('gfs.f006.idx')             # wgrib2 -s lines, as NOAA's .idx
g = fastgrib.open(url, index='gfs.f006.idx')   # later: nothing fetched until read
```

`index=` takes a path or a URL, in any of the formats above. The index of
part of a file serves that part: its last message is fetched up to its own
length, not to the end of the file. From the command line:

```bash
fastgrib URL -max_records 50 > part.idx
fastgrib URL -index part.idx -match ':TMP:' -grib tmp.grib2
```

## ECMWF open data

```python
url = ('https://data.ecmwf.int/forecasts/20261023/00z/ifs/0p25/oper/'
       '20261023000000-6h-oper-fc.grib2')
e = fastgrib.open(url)
t2 = e.read(short_name='2t')              # CCSDS-packed, decoded by fastgrib's own decoder
```

ECMWF's open-data server keeps only the last few days of runs, so a URL
like this one stops working after that.

## Notes

- Remote reads go through Python's standard library, so there's nothing
  extra to install. The usual proxy environment variables apply.
- `s3://bucket/key` is read through the bucket's public HTTPS address, so it
  works for public buckets such as NOAA's, without credentials.
- A remote file can be written to a local file with `f.write(...)`, like
  any other (see [Writing GRIB2](writing.md)).
