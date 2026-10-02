"""Minimal protobuf reader for Android Studio profiler payloads.

Only the wire types used by the profiler schema are supported (varint,
64-bit, length-delimited, 32-bit). Yields ``(field_number, wire_type, value,
value_pos, value_len)`` tuples; callers interpret values (nested messages
arrive as ``bytes``).
"""


#     Copyright 2026 András Oravecz <info@oandras.hu>
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

def read_varint(buf, pos):
    out = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        out |= (b & 0x7F) << shift
        if not (b & 0x80):
            return out, pos
        shift += 7


def walk(buf, start=0, end=None):
    """Yield (field_no, wire_type, value, value_pos, value_len)."""
    if end is None:
        end = len(buf)
    pos = start
    while pos < end:
        key, pos = read_varint(buf, pos)
        field, wt = key >> 3, key & 7
        if wt == 0:
            v, pos = read_varint(buf, pos)
            yield field, wt, v, pos, 0
        elif wt == 1:
            yield field, wt, buf[pos:pos + 8], pos, 8
            pos += 8
        elif wt == 2:
            ln, pos = read_varint(buf, pos)
            yield field, wt, buf[pos:pos + ln], pos, ln
            pos += ln
        elif wt == 5:
            yield field, wt, buf[pos:pos + 4], pos, 4
            pos += 4
        else:
            raise ValueError(f'bad wire type {wt} at {pos}')


def fm(blob):
    """Decode a message into {field_no: [values]}; never raises."""
    if not isinstance(blob, (bytes, bytearray)) or not blob:
        return {}
    out = {}
    try:
        for f, wt, v, _, _ in walk(blob):
            out.setdefault(f, []).append(v)
    except (IndexError, KeyError, ValueError):
        return out
    return out


def sub(blob, num, default=None):
    """First length-delimited value of a field, or *default*."""
    v = [x for x in fm(blob).get(num, []) if isinstance(x, bytes)]
    return v[0] if v else default


def sv(d, num, default=0):
    """First scalar value of a field, or *default*."""
    v = d.get(num)
    return v[0] if v else default


def ss(d, num):
    """First value of a field decoded as UTF-8, or ''."""
    v = d.get(num)
    return v[0].decode('utf-8', 'replace') if v else ''
