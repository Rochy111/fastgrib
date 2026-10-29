# Keys and sections

## ecCodes keys

```python
k = f.keys(0)
k['bitsPerValue'], k['typeOfLevel'], k['stepRange']
```

`f.keys(pick)` returns one field's keys under the names ecCodes, pygrib and
cfgrib use, read straight from the message's bytes. That covers the
identification keys (`discipline`, `parameterCategory`, `shortName`), the
level keys (`typeOfFirstFixedSurface`, `scaledValueOfFirstFixedSurface`,
`typeOfLevel`, `level`), grid keys (`Ni`, `Nj`,
`latitudeOfFirstGridPointInDegrees`, `LoVInDegrees`), time keys
(`forecastTime`, `typeOfStatisticalProcessing`, `lengthOfTimeRange`,
`stepRange`, `stepType`), ensemble keys (`perturbationNumber`) and packing
keys (`packingType`, `referenceValue`, `bitsPerValue`, `numberOfMissing`).

The values equal ecCodes' on every key of every test field, about 17,000
checks. The one exception is negative potential-vorticity levels, which
ecCodes reads as unsigned.

## Raw sections

For anything not covered by a named key, `f.sections(pick)` returns the
field's sections as a dict of section number to bytes, each whole with its
length and number first, so you can read any octet yourself. For a message
holding several fields, these are the sections in force for that field.
