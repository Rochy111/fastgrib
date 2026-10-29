# Accuracy

## Values

fastgrib unpacks integers and applies GRIB's scaling exactly as wgrib2
does: `(R + X·2^E)·10^-D`, computed in double precision with wgrib2's
integer powers and rounded once to float32. So every value is bit-for-bit
wgrib2's, and NaN appears exactly where wgrib2 writes 9.999e20.

This has been checked against wgrib2 3.1.3 on:

- every field of GFS 1° and 0.5° (1,486 fields, 110 with undefined points);
- NBM CONUS on its Lambert grid, with probabilities, percentiles and
  ensemble statistics;
- GEFS members and statistics;
- JPEG2000, CCSDS and PNG re-encodings;
- MRMS (PNG);
- ECMWF open data (CCSDS, 191 fields);
- fields regridded onto every supported projection, including their
  latitudes and longitudes;
- every scanning mode.

Rotated and reduced grids' point positions match ecCodes' to 1e-5 degrees,
and their values match wgrib2's. IEEE and run-length values match wgrib2's.
GRIB1 values match ecCodes' (NARR's 423 fields and ecCodes' GRIB1 samples)
and wgrib2's for the same messages converted to GRIB2.

There is one deliberate difference. For constant GRIB1 fields with a
decimal scale factor, fastgrib applies GRIB's formula, while ecCodes and
g2clib don't. NARR's cloud water at 100 hPa comes out as 1e-12 kg/kg in
fastgrib and as 0.001 in ecCodes. wgrib2 with `-g2clib 0` agrees with
fastgrib.

## Inventory lines and names

Inventory lines match wgrib2's, and the `.idx` files NOAA publishes, line
for line. That was checked on whole GFS, GEFS, NBM, HRRR, RAP, NAM and RTMA
files. Where wgrib2 3.1.3 lacks a parameter name or level type, fastgrib
uses ecCodes' name and newer wgrib2's level wording.

`short_name` and `level_kind` match ecCodes' `shortName` and `typeOfLevel`
on every field tried (1,276 fields from GFS, GEFS, NBM and ECMWF).
[Keys](../guide/keys.md) match ecCodes' on about 17,000 checks.

## Testing it yourself

```bash
pytest
```

The tests repeat these checks on small files, with wgrib2's and ecCodes'
output stored beside each one, so they run without wgrib2, ecCodes or a
network connection. `FASTGRIB_BIG_TEST=1 pytest tests/test_big.py` also
writes, reads and removes a 4.3 GB file.
