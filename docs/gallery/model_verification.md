# GFS and ECMWF verification

The 500 hPa height forecasts of GFS and ECMWF's IFS from 18 daily runs, each scored against its own model's analysis at every 24 hours of lead time. Panel (a) is the latitude-weighted RMSE over 20-90°N for each run and on average; panel (b) is where GFS's day-5 errors were larger (red) or smaller (blue) than IFS's. The two models come from two centres' archives; fastgrib fetches about 450 fields, one per file, in about 20 seconds.

![GFS and ECMWF verification](../images/model_verification.png)

Data: GFS 1° from NOAA's archive and IFS 0.25° open data (CC BY 4.0) from ECMWF's Google Cloud mirror, 00 UTC runs from 1 to 18 October 2026.

```python title="docs/examples/model_verification.py"
--8<-- "model_verification.py"
```
