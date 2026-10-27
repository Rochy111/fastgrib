---
name: Bug report
about: Wrong values, a crash, or a file fastgrib cannot read
labels: bug
---

**What happened**
<!-- What you ran and what went wrong (the error, or the values you expected). -->

```python
import fastgrib
f = fastgrib.open('...')
```

**The file**
<!-- Where it comes from (model, product, URL if public), or attach a small
     file -- `f.write('one.grib2', which)` writes just the field concerned.
     If values differ from wgrib2: what `wgrib2 FILE -d N -order we:sn -bin out.bin` gives. -->

**Versions**
<!-- Paste the output of:
python -c "import sys, platform, fastgrib, numpy, numba; print(sys.version, platform.platform()); print('fastgrib', fastgrib.__version__, 'numpy', numpy.__version__, 'numba', numba.__version__)"
-->
