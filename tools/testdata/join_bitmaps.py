"""Test data: pairs of consecutive bitmapped messages with the same bitmap, joined into
one message each, the second field's bitmap section saying 254 ('the one
before'): argv in.grib2 out.grib2."""
import struct, sys
b = open(sys.argv[1], 'rb').read()
msgs, pos = [], 0
while (pos := b.find(b'GRIB', pos)) >= 0:
    n = struct.unpack('>Q', b[pos + 8:pos + 16])[0]
    msgs.append(b[pos:pos + n]); pos += n

def sections(m):
    out, p = {}, 16
    while p < len(m) - 4:
        ln, num = struct.unpack('>IB', m[p:p + 5])
        out[num] = m[p:p + ln]; p += ln
    return out

out, k, joined = [], 0, 0
while k < len(msgs):
    a = sections(msgs[k])
    if k + 1 < len(msgs) and a[6][5] == 0:
        c = sections(msgs[k + 1])
        if c[6][5] == 0 and c[6] == a[6] and c[3] == a[3]:
            body = a[1] + a.get(2, b'') + a[3] + a[4] + a[5] + a[6] + a[7] + \
                c[4] + c[5] + struct.pack('>IBB', 6, 6, 254) + c[7]
            n = 16 + len(body) + 4
            out.append(msgs[k][:8] + struct.pack('>Q', n) + body + b'7777')
            k += 2; joined += 1
            continue
    out.append(msgs[k]); k += 1
open(sys.argv[2], 'wb').write(b''.join(out))
print('joined', joined, 'pairs;', len(out), 'messages')
