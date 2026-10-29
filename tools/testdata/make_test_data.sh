#!/bin/bash
# Build tests/data: small GRIB2 files and wgrib2's own reading of each --
# its inventory (NAME.inv, wgrib2 -s) and values (NAME.bin, float32 we:sn,
# 9.999e20 undefined) -- and ecCodes' names of some (NAME.ecc), so the
# tests need neither wgrib2, ecCodes nor the network.
#
#   tools/testdata/make_test_data.sh GFS_1DEG NBM_QMD ECMWF_MSG ECMWF_FC GEFS_DIR NAM_BGRD OUT_DIR
#
# GFS_1DEG   a GFS 1-degree pgrb2 file (e.g. gfs.t00z.pgrb2.1p00.f006)
# NBM_QMD    an NBM CONUS qmd file (e.g. blend.t00z.qmd.f132.co.grib2)
# ECMWF_MSG  one CCSDS-packed message of ECMWF's open data
# ECMWF_FC   an ECMWF open-data forecast file (e.g. ...-6h-oper-fc.grib2):
#            its local parameters and soil / MU-parcel levels
# GEFS_DIR   GEFS 0.5-degree pgrb2a files of one cycle and step, named
#            gep01, gec00, geavg, gespr (a member, the control, the mean
#            and the spread: product templates 4.1, 4.11, 4.2, 4.12)
# NAM_BGRD   a NAM native B-grid file (nam.tCCz.bgrdsfFF.tm00.grib2): its LAND,
#            NLAT and ELON fields
# Needs wgrib2 built with IPOLATES, OpenJPEG and libaec, gcc with libopenjp2's
# headers, and ecCodes'
# Python bindings (pip install eccodes eccodeslib).
set -e
tools=$(cd "$(dirname "$0")" && pwd)
gfs=$1; nbm=$2; ecm=$3; ecfc=$4; gefs=$(cd "$5" && pwd); nam=$(realpath "$6"); out=$7
mkdir -p "$out"; cd "$out"
# GFS: every 7th field and those with undefined points, a 41 x 31 crop
wgrib2 "$gfs" | awk -F: 'NR%7==1 || /SOILW|TSOIL|ICETK|WEASD|HGT:PV|UGRD:1829|LAND|ICEC/' \
  | wgrib2 -i "$gfs" -small_grib 250:290 20:50 gfs.grib2 > /dev/null
# other packings of a few of them
wgrib2 gfs.grib2 -for 1::5 -set_grib_type jpeg -grib_out gfs_jpeg.grib2 > /dev/null
wgrib2 gfs.grib2 -for 1::5 -set_grib_type aec -grib_out gfs_aec.grib2 > /dev/null
wgrib2 gfs.grib2 -for 1::5 -set_grib_type simple -grib_out gfs_simple.grib2 > /dev/null
# messages of several fields: wind pairs as wgrib2 -submsg_uv writes them
uv=$(mktemp)
wgrib2 "$gfs" \
  | grep -E ':(PRMSL:mean sea level|[UV]GRD:(10|250|500|850|1000) mb|SOILW:0-0.1 m below ground|[UV]GRD:10 m above ground):' \
  | wgrib2 -i "$gfs" -small_grib 250:290 20:50 "$uv" > /dev/null
wgrib2 "$uv" -submsg_uv gfs_uv.grib2 > /dev/null
rm -f "$uv"
# winds on Lambert and polar grids, along the grid's axes and east-north
uvsrc=$(mktemp)
wgrib2 "$gfs" -match ':(UGRD|VGRD):(850 mb|10 m above ground):' -grib "$uvsrc" > /dev/null
for w in grid earth; do
  wgrib2 "$uvsrc" -new_grid_winds $w -new_grid lambert:-97:30:60 -125:40:75000 20:30:75000     lcc_uv$w.grib2 > /dev/null
  wgrib2 "$uvsrc" -new_grid_winds $w -new_grid nps:-105:60 -150:30:180000 10:24:180000     nps_uv$w.grib2 > /dev/null
done
rm -f "$uvsrc"
# a field on each projection, and lat-lon scanned north to south
t2m=$(mktemp); wgrib2 "$gfs" -match ':TMP:2 m above ground:' -grib "$t2m" > /dev/null
ng() { wgrib2 "$t2m" -new_grid_winds earth -new_grid "$@" > /dev/null; }
ng nps:-105:60 -150:30:180000 10:24:180000 nps.grib2
ng sps:0:-71 -60:20:300000 -50:16:300000 sps.grib2
ng mercator:20 -180:30:130000:-120 -10:24:130000:40 merc.grib2
ng gaussian 0:192:1.875 88.542:94 gauss.grib2
ng lambert:-97:30:60 -125:40:75000 20:30:75000 lcc.grib2
ng latlon 0:72:5 90:37:-5 ll_ns.grib2
# ellipsoidal (WGS84) copies of three: geolocation through pyproj
for f in lcc nps merc; do
  python3 - "$f" <<'PY'
import sys
b = bytearray(open(sys.argv[1] + '.grib2', 'rb').read())
p = 16
while b[p + 4] != 3:                        # section 3
    p += int.from_bytes(b[p:p + 4], 'big')
b[p + 14] = 5                               # shape of the earth: WGS84
open(sys.argv[1] + '_wgs84.grib2', 'wb').write(bytes(b))
PY
done
# IEEE (32-bit), and from it 64-bit IEEE and run-length (5.200) files
wgrib2 gfs.grib2 | grep -E 'SOILW|TSOIL|WEASD|ICETK|PRMSL|TMP:2 m|HGT:500'   | wgrib2 -i gfs.grib2 -set_grib_type ieee -grib_out gfs_ieee.grib2 > /dev/null
wgrib2 gfs_ieee.grib2 -s > gfs_ieee.inv
python3 "$tools/make_more_packings.py" gfs_ieee.grib2 gfs_simple.grib2 .
# rotated and reduced grids (no wgrib2 -new_grid for them), from ecCodes' samples
python3 "$tools/make_grids_eccodes.py" .
# GRIB1: ecCodes' samples, and a few messages of a NARR analysis with its wgrib inventory
narr=https://www.ncei.noaa.gov/data/north-american-regional-reanalysis/access/3-hourly/200709/20070901/narr-a_221_20070901_0000_000
curl -sf -o narr.grb "$narr.grb" && curl -sf -o narr.inv "$narr.inv"
python3 "$tools/make_grib1_eccodes.py" . narr.grb narr.inv
python3 "$tools/make_grib1_second_order.py" .
python3 "$tools/make_grib1_predefined.py" .
python3 "$tools/make_grib1_matrix.py" .               # matrices of values (hand-made: ecCodes writes none)
python3 "$tools/make_spectral.py" .              # (needs pyshtools too)
rm -f narr.grb narr.inv
# PNG (this wgrib2 writes none): the simple-packed file's integers as PNGs
python3 "$tools/make_png_grib.py" gfs_simple.grib2 gfs_png.grib2
# every row / column scanning mode (0x10, 0x20 with 0x40, 0x80): the simple-
# packed fields' points reordered by wgrib2's rule
python3 "$tools/make_scan_grib.py" gfs_simple.grib2 gfs_scan.grib2
# bitmaps reused within a message (254): bitmapped neighbours joined
python3 "$tools/join_bitmaps.py" gfs.grib2 gfs_bm254.grib2
# NBM: probabilities, percentiles, ensemble statistics, a Lambert crop
wgrib2 "$nbm" -for 1::40 -ijsmall_grib 1000:1060 600:640 nbm.grib2 > /dev/null
cp "$ecm" ecmwf.grib2
# ECMWF's local parameters (tp, lcc, tcc, ro: no wgrib2 name), soil layers,
# the MU parcel's level, statistics with a time increment
wgrib2 "$ecfc" \
  | grep -E 'center=98 local_table=1 parmcat=(1 parm=193|6 parm=19[23]|0 parm=201)|:SOILTMP:|:CAPE:reserved|:TMP:2 m above ground:3-6 hour max|:GUST:|:PTYPE:' \
  | wgrib2 -i "$ecfc" -small_grib 270:285 30:42 ecmwf_local.grib2 > /dev/null
# GEFS: every 9th field and the statistics of each, a 41 x 31 crop
: > gefs.grib2
for m in gep01 gec00 geavg gespr; do
  wgrib2 "$gefs/$m" | awk -F: 'NR%9==1 || /TMAX|APCP|TCDC/' \
    | wgrib2 -i "$gefs/$m" -small_grib 250:290 20:50 gefs_$m.tmp > /dev/null
  cat gefs_$m.tmp >> gefs.grib2; rm gefs_$m.tmp
done
# regridding: wgrib2's -new_grid (IPOLATES) of grids it maps consistently, bilinear and nearest
# (the targets are the tests' REGRID cases)
for c in "lcc 235.169:63:0.348 25.528:39:0.348" "nps 200.27:50:0.556 20.015:39:0.556" \
         "gauss 232.5:81:3.095 -61.903:40:3.095" "ll_ns 230:78:3.25 -65:40:3.25"; do
  set -- $c
  for i in bilinear neighbor; do
    wgrib2 $1.grib2 -new_grid_interpolation $i -new_grid latlon $2 $3 rg.tmp > /dev/null
    wgrib2 rg.tmp -order we:sn -no_header -bin regrid_$1_$i.bin > /dev/null; rm rg.tmp
  done
done
# what fastgrib writes (new values packed three ways, metadata set), for wgrib2 to read below
python3 "$tools/make_written.py" .
python3 "$tools/make_other_grids.py" .        # space views, an unstructured grid
python3 "$tools/make_staggered.py" .          # staggered rotated grids, NCEP's E-grid
# NCEP's rotated B-grid (3.32769): NAM's native land mask, and NCEP's own latitudes and
# longitudes of its points (every 13th) to place it by
wgrib2 "$nam" -match ':LAND:surface:' -grib nam_bgrd.grib2 > /dev/null
wgrib2 "$nam" -match ':(NLAT|ELON):surface:' -order we:sn -no_header -bin nam_ll.bin > /dev/null
python3 -c "import numpy as np; v = np.fromfile('nam_ll.bin', np.float32).reshape(2, -1)[:, ::13]; np.savez_compressed('nam_bgrd_ll.npz', lat=v[0], lon=v[1])" && rm nam_ll.bin
# JPEG2000 codestreams of every Part-1 coding, by OpenJPEG (libopenjp2-7-dev)
gcc -O2 -o opj_tool "$tools/opj_tool.c" -I/usr/include/openjpeg-2.4 -lopenjp2
python3 "$tools/make_jpeg2000.py" ./opj_tool . && rm opj_tool
python3 "$tools/make_nbm_qmd.py" .          # NBM QMD fields: complex packing, missing values managed
for f in *.grib2; do
  b=${f%.grib2}
  [ "$b" = gfs_ieee64 ] || wgrib2 "$f" -s > "$b.inv"
  wgrib2 "$f" -order we:sn -no_header -bin "$b.bin" > /dev/null 2>&1 || true
done
cp gfs_ieee.bin gfs_ieee64.bin          # wgrib2 reads no 64-bit IEEE: the same values
python3 "$tools/eccodes_names.py" gfs.grib2 nbm.grib2 gefs.grib2 ecmwf.grib2 ecmwf_local.grib2 \
  gfs_uv.grib2 gfs_bm254.grib2 lcc.grib2
python3 "$tools/eccodes_keys.py" . gfs.grib2 nbm.grib2 gefs.grib2 ecmwf.grib2 lcc.grib2 nps.grib2 \
  merc.grib2 gauss.grib2 rot_ll.grib2 red_gg.grib2 gfs_ieee.grib2 gfs_jpeg.grib2 gfs_uv.grib2 \
  lcc_uvgrid.grib2 written.grib2 gefs_aero.grib2 sv.grib2 sv_more.grib2 ug.grb2 sh_more.grb2
for f in nps sps merc gauss lcc ll_ns lcc_wgs84 nps_wgs84 merc_wgs84 sv stag irr; do
  wgrib2 $f.grib2 -order we:sn -csv $f.csv > /dev/null
done
# csv -> compact lat/lon, and the big dumps (ECMWF's, NAM's B-grid) -> their hashes
python3 - <<'PY'
import csv, glob, hashlib, os
import numpy as np
for p in glob.glob("*.csv"):
    rows = list(csv.reader(open(p)))
    np.savez_compressed(p[:-4] + "_latlon.npz", lat=np.array([float(r[5]) for r in rows]),
                        lon=np.array([float(r[4]) for r in rows]))
    os.remove(p)
for big in ("ecmwf", "nam_bgrd"):
    open(big + ".sha256", "w").write(hashlib.sha256(open(big + ".bin", "rb").read()).hexdigest() + "\n")
    os.remove(big + ".bin")
PY
rm -f "$t2m"
ls -la
