"""A field's keys as ecCodes (pygrib, cfgrib) names them, read from its
message's octets: section 0's and 1's, the grid's (templates 3.0, 3.1,
3.10, 3.20, 3.30, 3.40, 3.41), the product's (4.0-4.15's level and
times, statistics, ensembles), the packing's, the bitmap's -- and what
fastgrib makes of them (shortName, typeOfLevel, stepRange ...)."""
import math
import struct

from . import _meta
from ._octets import s, u

_PACKING = {0: 'grid_simple', 2: 'grid_complex', 3: 'grid_complex_spatial_differencing',
            4: 'grid_ieee', 40: 'grid_jpeg', 41: 'grid_png', 42: 'grid_ccsds',
            50: 'spectral_simple', 51: 'spectral_complex', 200: 'grid_run_length'}
_STEP_TYPE = {0: 'avg', 1: 'accum', 2: 'max', 3: 'min', 4: 'diff', 5: 'rms', 6: 'sd'}
# where 4.n's statistics start (octet 35 + this): 4.8's at 0, its ensemble 4.11's at 3 ...
_STAT_AT = {8: 0, 9: 13, 10: 1, 11: 3, 12: 2, 13: 34, 14: 30, 15: None}
_ENS = (1, 11)
_MISSING32 = 0xFFFFFFFF


def _deg(b, p):
    return s(b, p, 4) * 1e-6


def keys(b, r, edition=2):
    """{ecCodes key: value} of record r's field; b holds its message."""
    from ._encode import sections
    secs = sections(b, r)
    out = {'editionNumber': edition, 'discipline': b[r.at + 6],      # (its own message's)
           'totalLength': 20 + sum(ln for k, (p, ln) in secs.items())}
    p = secs[1][0]
    out.update(centre=u(b, p + 5, 2), subCentre=u(b, p + 7, 2), tablesVersion=b[p + 9],
               localTablesVersion=b[p + 10], significanceOfReferenceTime=b[p + 11],
               dataDate=u(b, p + 12, 2) * 10000 + b[p + 14] * 100 + b[p + 15],
               dataTime=b[p + 16] * 100 + b[p + 17],
               productionStatusOfProcessedData=b[p + 19], typeOfProcessedData=b[p + 20])
    _grid_keys(b, secs[3][0], out)
    _product_keys(b, secs[4][0], out)
    p = secs[5][0]
    drt = u(b, p + 9, 2)
    out.update(numberOfValues=u(b, p + 5, 4), dataRepresentationTemplateNumber=drt,
               packingType=_PACKING.get(drt, 'template 5.%d' % drt))
    if drt in (0, 2, 3, 40, 41, 42, 50, 51):
        out.update(referenceValue=struct.unpack_from('>f', bytes(b[p + 11:p + 15]))[0],
                   binaryScaleFactor=s(b, p + 15, 2), decimalScaleFactor=s(b, p + 17, 2),
                   bitsPerValue=b[p + 19])
    elif drt == 4:
        out['precision'] = b[p + 11]            # (1: 32 bits a value, 2: 64)
    if drt == 50:
        out['realPartOf00'] = struct.unpack_from('>f', bytes(b[p + 20:p + 24]))[0]
    elif drt == 51:
        out.update(laplacianOperator=s(b, p + 20, 4) * 1e-6, JS=u(b, p + 24, 2),
                   KS=u(b, p + 26, 2), MS=u(b, p + 28, 2), TS=u(b, p + 30, 4),
                   unpackedSubsetPrecision=b[p + 34])
    out['bitmapPresent'] = int(r.bitmap >= 0)
    out['numberOfMissing'] = out['numberOfDataPoints'] - out['numberOfValues']
    out.update(shortName=r.short_name or 'unknown', name=r.long_name, units=r.units,
               typeOfLevel=r.level_kind, level=_level(r), uvRelativeToGrid=int(r.uv_relative))
    if r.valid_time is not None:
        out['validityDate'] = int(r.valid_time.strftime('%Y%m%d'))
        out['validityTime'] = r.valid_time.hour * 100 + r.valid_time.minute
    return out


def _level(r):
    """ecCodes' level: a whole number, in its unit for the kind of level
    (Pa below 1 hPa and for pressure above ground, 1e-9 for PV); the exact
    value is r.level_value."""
    v, kind = r.level_value, r.level_kind
    if v is None:
        return 0
    if kind in ('isobaricInPa', 'pressureFromGroundLayer'):
        v = v * 100.0
    elif kind == 'potentialVorticity':
        v = v * 1e9
    return int(math.floor(v + 0.5))


def _grid_keys(b, p, out):
    t = u(b, p + 12, 2)
    out.update(gridDefinitionTemplateNumber=t, numberOfDataPoints=u(b, p + 6, 4),
               shapeOfTheEarth=b[p + 14])
    g = {0: 'regular_ll', 1: 'rotated_ll', 10: 'mercator', 20: 'polar_stereographic',
         30: 'lambert', 40: 'regular_gg', 41: 'rotated_gg', 50: 'sh', 51: 'rotated_sh',
         52: 'stretched_sh', 53: 'stretched_rotated_sh', 90: 'space_view',
         101: 'unstructured_grid', 130: 'irregular_latlon'}.get(t)
    if g is None:
        return
    if t == 130:                                # each point's latitude, longitude listed
        out['gridType'] = g
        return
    if t == 90:                                 # space view
        out.update(Nx=u(b, p + 30, 4), Ny=u(b, p + 34, 4),
                   latitudeOfSubSatellitePointInDegrees=s(b, p + 38, 4) * 1e-6,
                   longitudeOfSubSatellitePointInDegrees=s(b, p + 42, 4) * 1e-6,
                   resolutionAndComponentFlags=b[p + 46], dx=u(b, p + 47, 4),
                   dy=u(b, p + 51, 4), XpInGridLengths=u(b, p + 55, 4) / 1e3,
                   YpInGridLengths=u(b, p + 59, 4) / 1e3, scanningMode=b[p + 63],
                   orientationOfTheGridInDegrees=s(b, p + 64, 4) * 1e-6,
                   NrInRadiusOfEarth=(None if u(b, p + 68, 4) == _MISSING32    # orthographic
                                      else u(b, p + 68, 4) * 1e-6), Xo=u(b, p + 72, 4),
                   Yo=u(b, p + 76, 4), gridType=g)
        out['Ni'], out['Nj'] = out['Nx'], out['Ny']
        return
    if t == 101:                                # unstructured: the grid file it is on
        out.update(numberOfGridUsed=u(b, p + 15, 3), numberOfGridInReference=b[p + 18],
                   uuidOfHGrid=bytes(b[p + 19:p + 35]).hex(), gridType=g)
        return
    if t in (50, 51, 52, 53):                   # spherical harmonics (rotated, stretched)
        out.update(J=u(b, p + 14, 4), K=u(b, p + 18, 4), M=u(b, p + 22, 4),
                   spectralType=b[p + 26], spectralMode=b[p + 27], gridType=g)
        q = p + 28
        if t in (51, 53):
            out.update(latitudeOfSouthernPoleInDegrees=s(b, q, 4) * 1e-6,
                       longitudeOfSouthernPoleInDegrees=u(b, q + 4, 4) * 1e-6,
                       angleOfRotationInDegrees=struct.unpack_from('>f', b, q + 8)[0])
            q += 12
        if t in (52, 53):
            out.update(latitudeOfStretchingPoleInDegrees=s(b, q, 4) * 1e-6,
                       longitudeOfStretchingPoleInDegrees=s(b, q + 4, 4) * 1e-6,
                       stretchingFactor=u(b, q + 8, 4) * 1e-6)
        return
    if t in (0, 1, 40, 41):
        ni, nj = u(b, p + 30, 4), u(b, p + 34, 4)
        reduced = ni == _MISSING32
        out.update(Ni=ni if not reduced else None, Nj=nj,
                   latitudeOfFirstGridPointInDegrees=_deg(b, p + 46),
                   longitudeOfFirstGridPointInDegrees=u(b, p + 50, 4) * 1e-6,
                   latitudeOfLastGridPointInDegrees=_deg(b, p + 55),
                   longitudeOfLastGridPointInDegrees=u(b, p + 59, 4) * 1e-6,
                   scanningMode=b[p + 71], resolutionAndComponentFlags=b[p + 54])
        di = u(b, p + 63, 4)
        out['iDirectionIncrementInDegrees'] = None if di == _MISSING32 else di * 1e-6
        out['Nx'], out['Ny'] = out['Ni'], nj
        if t in (40, 41):
            out['N'] = u(b, p + 67, 4)
        else:
            dj = u(b, p + 67, 4)
            out['jDirectionIncrementInDegrees'] = None if dj == _MISSING32 else dj * 1e-6
        if t in (1, 41):
            out.update(latitudeOfSouthernPoleInDegrees=_deg(b, p + 72),
                       longitudeOfSouthernPoleInDegrees=u(b, p + 76, 4) * 1e-6,
                       angleOfRotationInDegrees=struct.unpack_from('>f', bytes(b[p + 80:p + 84]))[0])
        if reduced:
            g = 'reduced_gg' if t == 40 else 'reduced_rotated_gg' if t == 41 else 'reduced_ll'
    elif t == 10:
        out.update(Ni=u(b, p + 30, 4), Nj=u(b, p + 34, 4),
                   latitudeOfFirstGridPointInDegrees=_deg(b, p + 38),
                   longitudeOfFirstGridPointInDegrees=u(b, p + 42, 4) * 1e-6,
                   resolutionAndComponentFlags=b[p + 46], LaDInDegrees=_deg(b, p + 47),
                   latitudeOfLastGridPointInDegrees=_deg(b, p + 51),
                   longitudeOfLastGridPointInDegrees=u(b, p + 55, 4) * 1e-6,
                   scanningMode=b[p + 59], DiInMetres=u(b, p + 64, 4) * 1e-3,
                   DjInMetres=u(b, p + 68, 4) * 1e-3)
        out.update(Nx=out['Ni'], Ny=out['Nj'], DxInMetres=out['DiInMetres'],
                   DyInMetres=out['DjInMetres'])
    else:                                       # 20, 30: polar stereographic, Lambert
        out.update(Nx=u(b, p + 30, 4), Ny=u(b, p + 34, 4),
                   latitudeOfFirstGridPointInDegrees=_deg(b, p + 38),
                   longitudeOfFirstGridPointInDegrees=u(b, p + 42, 4) * 1e-6,
                   resolutionAndComponentFlags=b[p + 46], LaDInDegrees=_deg(b, p + 47),
                   LoVInDegrees=u(b, p + 51, 4) * 1e-6, DxInMetres=u(b, p + 55, 4) * 1e-3,
                   DyInMetres=u(b, p + 59, 4) * 1e-3, projectionCentreFlag=b[p + 63],
                   scanningMode=b[p + 64])
        out['Ni'], out['Nj'] = out['Nx'], out['Ny']
        if t == 30:
            out.update(Latin1InDegrees=_deg(b, p + 65), Latin2InDegrees=_deg(b, p + 69),
                       latitudeOfSouthernPoleInDegrees=_deg(b, p + 73),
                       longitudeOfSouthernPoleInDegrees=u(b, p + 77, 4) * 1e-6)
    out['gridType'] = g


def _surface(b, p):
    t, sf, sv = b[p], b[p + 1], u(b, p + 2, 4)
    return t, (None if sf == 255 else _meta._sm(sf, 1)), \
        (None if sv == _MISSING32 else _meta._sm(sv, 4))


def _product_keys(b, p, out):
    tmpl = u(b, p + 7, 2)
    out.update(productDefinitionTemplateNumber=tmpl, parameterCategory=b[p + 9],
               parameterNumber=b[p + 10])
    if tmpl in _meta._EXTENDS:                  # a chemical's or aerosol's: read as 4.0-4.11's
        b, p, tmpl = _meta._normal(b, p, tmpl)
    if tmpl not in _meta._WITH_LEVEL:
        return
    out.update(typeOfGeneratingProcess=b[p + 11], generatingProcessIdentifier=b[p + 13],
               indicatorOfUnitOfTimeRange=b[p + 17], forecastTime=s(b, p + 18, 4))
    for side, q in (('First', p + 22), ('Second', p + 28)):
        t, sf, sv = _surface(b, q)
        out['typeOf%sFixedSurface' % side] = t
        out['scaleFactorOf%sFixedSurface' % side] = sf
        out['scaledValueOf%sFixedSurface' % side] = sv
    ft, unit = out['forecastTime'], out['indicatorOfUnitOfTimeRange']
    out['stepUnits'] = unit
    if tmpl in _ENS:
        out.update(typeOfEnsembleForecast=b[p + 34], perturbationNumber=b[p + 35],
                   numberOfForecastsInEnsemble=b[p + 36])
    elif tmpl in (2, 12):                       # derived from an ensemble: its kind, its size
        out.update(derivedForecast=b[p + 34], numberOfForecastsInEnsemble=b[p + 35])
    at = _STAT_AT.get(tmpl)
    if at is None:
        out['stepType'], out['stepRange'] = 'instant', str(ft)
        return
    q = p + 34 + at                             # the statistic: its end, ranges, the first's
    n = b[q + 7]
    stat, lunit, length = b[q + 12], b[q + 14], u(b, q + 15, 4)
    out.update(numberOfTimeRange=n, typeOfStatisticalProcessing=stat,
               indicatorOfUnitForTimeRange=lunit, lengthOfTimeRange=length)
    out['stepType'] = _STEP_TYPE.get(stat, 'stat %d' % stat)
    out['stepRange'] = '%d-%d' % (ft, ft + length) if lunit == unit else '%d' % ft
