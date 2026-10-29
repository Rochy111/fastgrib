# Hovmöller diagram

The 250 hPa meridional wind averaged over 35-60°N, from 60 daily GFS analyses. Each analysis is a separate file in NOAA's archive, and fastgrib fetches the one field it needs from each; all 60 take about four seconds. Rossby wave packets show as bands of alternating colour sloping eastward with time, moving faster than the troughs and ridges inside them.

![Hovmöller diagram](../images/hovmoller.png)

Data: GFS 1° analyses, 00 UTC, 30 August to 28 October 2026.

```python title="docs/examples/hovmoller.py"
--8<-- "hovmoller.py"
```
