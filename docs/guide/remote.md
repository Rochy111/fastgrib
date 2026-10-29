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

If there is no index, the whole file is downloaded once.

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
