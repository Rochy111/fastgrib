# Unusual grids

Most model output is on lat-lon, Gaussian, Lambert or polar stereographic
grids, and fastgrib handles those like any GRIB reader. This page covers
the less common ones, where fastgrib goes further than wgrib2 or ecCodes.

## Spherical harmonics

Global spectral models (ECMWF's IFS, ARPEGE) store some fields as
spherical-harmonic coefficients: GRIB2 templates 3.50 to 3.53 with packing
5.50 or 5.51, and GRIB1's equivalents.

```python
coeffs = f.read(0)                                    # the coefficients
t = f.regrid((np.arange(-90, 90.1, 0.5), np.arange(0, 360, 0.5)), 0)
point = f.at(51.5, 0.0, 0)
```

`read()` returns the coefficients exactly as ecCodes and pygrib do. `regrid()`
and `at()` go further and synthesise the field at any points, which those
tools don't. They use ECMWF's normalisation and agree with pyshtools to
1e-5 K. A T63 field onto a global 0.5° grid takes 12 ms. Rotated and
stretched fields (ARPEGE's kind) are synthesised in their own frame.
`encode()` packs coefficients back to the same bits. There is an example in
the [gallery](../gallery.md#a-field-from-spherical-harmonics).

## Satellite images

Space-view grids (template 3.90), such as Meteosat and Himawari images in
GRIB, are placed using the CGMS navigation on the field's own ellipsoid.
The points match wgrib2's, and PROJ's `geos` projection to 1e-10 degrees.
Points beyond the Earth's edge are NaN, where wgrib2 prints a sentinel and
ecCodes gives (0, 0). A full Meteosat disc of 13.8 million points takes
0.04 s.

Views that neither wgrib2 nor ecCodes can place (a satellite off the
equator or over a pole, a rotated image, orthographic views) are placed the
way PROJ's tilted perspective and orthographic projections do, to 1e-8
degrees. `regrid()` and `at()` work from these like any other grid.

## Staggered grids

Grids whose alternate rows are offset by half a step (scanning-mode flags
0x08, 0x04, 0x02 and 0x01) are placed as wgrib2 places them.

NCEP's rotated E-grid (3.32768) and B-grid (3.32769, NAM's native output,
which wgrib2 can't place) are placed as rotated grids about their centre.
NAM's points land where NCEP's own NLAT and ELON fields put them, to those
fields' packing precision.

## Points listed one by one

Template 3.130 lists each point's position. Its fields read as flat
`values` arrays with each point's latitude and longitude, and `at()` and
`regrid()` use the nearest point.

## Grids defined elsewhere

ICON's unstructured grid (3.101) and curvilinear grids (3.204) don't
include their points in the message. Their fields read as flat `values`,
and `keys()` names the grid file to use (`uuidOfHGrid`,
`numberOfGridUsed`).
