# Gallery

The pictures are drawn with [fastplot](https://rochy111.github.io/fastplot/)
from fields read by fastgrib. The scripts are in `docs/examples/`.

## Fields from a remote file

fastgrib opens a GFS file on NOAA's AWS archive, reads the index published
beside it, and downloads only the two fields the map needs. To run it against
a local copy instead, set `GFS_FILE`.

![500 hPa height over sea-level pressure](images/remote_fields.png)

??? example "remote_fields.py"
    ```python
    --8<-- "remote_fields.py"
    ```

## A field from spherical harmonics

Global spectral models store some fields as spherical-harmonic coefficients
rather than on a grid. wgrib2, ecCodes and pygrib return the coefficients,
and fastgrib can also turn them back into a map. This one uses a T63
surface temperature field from fastgrib's test data.

![Surface temperature from spherical harmonics](images/spectral.png)

??? example "spectral.py"
    ```python
    --8<-- "spectral.py"
    ```

## How fast

Milliseconds to decode each field of five operational files.

![Decoding speed](images/speed_chart.png)

??? example "speed_chart.py"
    ```python
    --8<-- "speed_chart.py"
    ```
