"""Find class-level properties without @JvmField.

Walks .kt files under --src, blanking comments/strings, then reports
val/var declarations at brace depth 0 of class/object bodies (incl. primary
constructor params) that lack @JvmField. The depth-0 rule skips locals in fun
bodies and in property-initializer lambdas (apply/run/with).

Excluded (cannot or should not carry it): open/abstract/override/lateinit/const,
delegated properties, custom accessors (incl. same-line get()/set() and
private set), interface/annotation members, value-class-typed properties,
private members of companion/object singletons.

usage:
    python3 scripts/find_no_jvmfield.py --src <kotlin-src-root> [--show-private]
"""
import argparse
import os
import re
import sys


ROOT = ''
SHOW_PRIVATE = False
VALUE_CLASS_RE = None
SKIP_MODS = {'open', 'abstract', 'override', 'lateinit', 'const'}

SKIP_MODS = {'open', 'abstract', 'override', 'lateinit', 'const'}
PROP_RE = re.compile(
    r'(?m)^[ \t]*'
    r'(?:@[\w:]+(?:\([^)]*\))?[ \t]+)*'
    r'(?:(?:public|internal|protected|private|open|abstract|override|lateinit|const|inline)[ \t]+)*'
    r'(?:val|var)[ \t]+(?P<name>\w+)'
)
ANN_RE = re.compile(r'@[\w:]+(?:\([^)]*\))?')
MOD_RE = re.compile(r'\b(public|internal|protected|private|open|abstract|override|lateinit|const|inline)\b')
CTX_RE = re.compile(
    r'(?P<ctx>class|object|interface|enum\s+class|annotation\s+class|companion\s+object)\s*'
    r'(?P<name>\w+)?')
CTOR_PROP_RE = re.compile(
    r'\s*(?P<ann>(?:@[\w:]+(?:\([^)]*\))?\s+)*)'
    r'(?P<mods>(?:public|internal|protected|private|open|abstract|override|lateinit|const|inline)\s+)*'
    r'(?P<kind>val|var)\s+(?P<name>\w+)')


def strip_comments_strings(src):
    """Replace comments and string literals with spaces (keep positions)."""
    out = []
    i, n = 0, len(src)
    while i < n:
        if src.startswith('/*', i):
            j = src.find('*/', i + 2)
            j = n if j == -1 else j + 2
            out.append(re.sub(r'[^\n]', ' ', src[i:j]))
            i = j
        elif src.startswith('//', i):
            j = src.find('\n', i)
            j = n if j == -1 else j
            out.append(' ' * (j - i))
            i = j
        elif src.startswith('"""', i):
            j = src.find('"""', i + 3)
            j = n if j == -1 else j + 3
            out.append(re.sub(r'[^\n]', ' ', src[i:j]))
            i = j
        elif src[i] == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == '\\' else 1
            j = min(n, j + 1)
            out.append(re.sub(r'[^\n]', ' ', src[i:j]))
            i = j
        elif src[i] == '\'' and i + 2 < n and src[i + 2] == '\'':
            out.append('   ')
            i += 3
        else:
            out.append(src[i])
            i += 1
    return ''.join(out)


def match_brace(src, i):
    """Given index of '{', return index just past matching '}'."""
    depth = 0
    while i < len(src):
        if src[i] == '{':
            depth += 1
        elif src[i] == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(src)


def find_matching(src, start, open_c='(', close_c=')'):
    depth = 0
    for i in range(start, len(src)):
        if src[i] == open_c:
            depth += 1
        elif src[i] == close_c:
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_top_commas(s):
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in '([{<':
            depth += 1
        elif ch in ')]}>':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append(''.join(cur))
    return parts


def parse_prop(decl_text, line_no):
    m = re.match(
        r'\s*(?P<ann>(?:@[\w:]+(?:\([^)]*\))?\s+)*)'
        r'(?P<mods>(?:public|internal|protected|private|open|abstract|override|lateinit|const|inline)\s+)*'
        r'(?P<kind>val|var)\s+(?P<name>\w+)', decl_text)
    if not m:
        return None
    mods = (m.group('mods') or '').split()
    return {
        'name': m.group('name'),
        'mods': mods,
        'has_jvmfield': 'JvmField' in (m.group('ann') or ''),
        'line': line_no,
    }


def collect_value_classes():
    """Collect source-declared `value class` names (comments/strings blanked)."""
    names = set()
    for dp, _, fns in os.walk(ROOT):
        for fn in sorted(fns):
            if fn.endswith('.kt'):
                with open(os.path.join(dp, fn), encoding='utf-8') as f:
                    clean = strip_comments_strings(f.read())
                for m in re.finditer(r'\bvalue\s+class\s+(\w+)', clean):
                    names.add(m.group(1))
    return names


def scan_file(path):
    with open(path, encoding='utf-8') as f:
        src = f.read()
    clean = strip_comments_strings(src)
    srclines = src.split('\n')
    results = []

    for m in re.finditer(
            r'(?P<ctx>class|object|interface|enum\s+class|annotation\s+class|companion\s+object)\b\s*'
            r'(?P<name>\w+)?', clean):
        # Skip class literals (Foo::class): not declarations. Their "body"
        # would be the following function body, leaking locals as properties.
        if clean[max(0, m.start() - 2):m.start()] == '::':
            continue
        # Skip backtick-quoted identifiers (e.g. SVGAttr.`class`): not
        # declarations either, for the same reason.
        if (m.start() > 0 and clean[m.start() - 1] == '`') or (
                m.end() < len(clean) and clean[m.end()] == '`'):
            continue
        ctx = re.sub(r'\s+', ' ', m.group('ctx')).strip()
        name = m.group('name')
        if ctx == 'object' and not name:
            # anonymous `object : Type` expression, not a declaration
            rest = clean[m.end():].lstrip()
            if rest.startswith(':'):
                continue
        is_iface = ctx.startswith('interface') or ctx.startswith('annotation')
        # constructor params (if any parens precede the body brace)
        pstart = clean.find('(', m.end())
        brace = clean.find('{', m.end())
        if 0 <= pstart < brace or (pstart >= 0 and brace == -1):
            pend = find_matching(clean, pstart)
            if pend != -1:
                for part in split_top_commas(clean[pstart + 1:pend]):
                    pm = CTOR_PROP_RE.match(part)
                    if pm:
                        results.append({
                            'ctx': ctx, 'ctor': True,
                            'name': pm.group('name'),
                            'mods': (pm.group('mods') or '').split(),
                            'has_jvmfield': 'JvmField' in (pm.group('ann') or ''),
                            'delegated': False,
                            'is_value_type': bool(
                                VALUE_CLASS_RE and VALUE_CLASS_RE.search(part)),
                            'file': os.path.relpath(path),
                            'line': src[:m.end()].count('\n') + 1,
                        })
                # A body brace must follow the header itself: if a `fun`
                # declaration (or anything function-like) comes first, this
                # class has no body (e.g. bodyless data class) — do not treat
                # the next function body as a class body.
                between = clean[pend:brace] if brace != -1 else clean[pend:]
                if re.search(r'\bfun\b', between):
                    continue
                body_start = brace
            else:
                continue
        else:
            body_start = brace
        if body_start == -1:
            continue
        body_end = match_brace(clean, body_start)
        body = clean[body_start + 1:body_end - 1]
        base_line = src[:body_start].count('\n') + 1
        for lm in PROP_RE.finditer(body):
            # Only brace-depth-0 declarations are class members: this skips
            # locals in fun bodies and in property-initializer lambdas
            # (apply/run/with). Nested-type members sit deeper too; nested
            # types are scanned separately via their own class match.
            if body.count('{', 0, lm.start()) != body.count('}', 0, lm.start()):
                continue
            decl = lm.group(0)
            line_no = base_line + body[:lm.start()].count('\n')
            # annotations on previous lines
            ann_extra = []
            ln = line_no - 1
            while ln >= 1:
                prev = srclines[ln - 1].strip()
                if prev.startswith('@'):
                    ann_extra.append(prev)
                    ln -= 1
                elif prev == '':
                    break
                else:
                    break
            info = parse_prop(decl + ' ', line_no)
            if info is None:
                continue
            if 'JvmField' in ' '.join(ann_extra):
                info['has_jvmfield'] = True
            # custom accessor: a get/set before the initializer '=' on the same
            # line (e.g. `val x: Boolean get() = ...`), or a get/set/private-set
            # block starting on one of the following lines
            after = body[lm.end():].split('\n', 1)[0]
            info['delegated'] = ' by ' in (' ' + after + ' ')
            info['is_value_type'] = bool(
                VALUE_CLASS_RE and VALUE_CLASS_RE.search(after))
            head = after.split('=', 1)[0]
            info['custom_acc'] = bool(re.search(r'\b(get|set)\b', head))
            if not info['custom_acc']:
                for rline in body[lm.end():].split('\n')[1:6]:
                    s = rline.strip()
                    if not s:
                        continue
                    if re.match(r'(private\s+)?(get|set)\b', s):
                        info['custom_acc'] = True
                    break
            info['ctx'] = ctx
            info['ctor'] = False
            info['file'] = os.path.relpath(path)
            results.append(info)
    return results


def main(argv=None):
    global ROOT, SHOW_PRIVATE, VALUE_CLASS_RE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True,
                        help='Kotlin source root to scan')
    parser.add_argument('--show-private', action='store_true',
                        help='also list private members')
    args = parser.parse_args(argv)
    ROOT = args.src
    SHOW_PRIVATE = args.show_private
    value_classes = collect_value_classes()
    if value_classes:
        VALUE_CLASS_RE = re.compile(
            r'\b(?:' + '|'.join(sorted(value_classes)) + r')\b')
    all_props = []
    for dp, _, fns in os.walk(ROOT):
        for fn in sorted(fns):
            if fn.endswith('.kt'):
                all_props.extend(scan_file(os.path.join(dp, fn)))
    shown = []
    for p in all_props:
        if p['has_jvmfield']:
            continue
        if p.get('delegated') or p.get('custom_acc'):
            continue
        ctx0 = p['ctx'].split()[0] if not p['ctor'] else 'ctor'
        if ctx0 in ('interface', 'annotation'):
            continue
        if not SHOW_PRIVATE and 'private' in p['mods']:
            continue
        if SKIP_MODS & set(p['mods']):
            continue
        if ctx0 in ('companion', 'object') and 'private' in p['mods']:
            # codebase precedent: private companion/object members stay plain
            continue
        if p.get('is_value_type'):
            # value-class-typed properties stay plain
            continue
        shown.append(p)
    print(f'total properties scanned: {len(all_props)}, missing @JvmField: {len(shown)}')
    for p in shown:
        where = 'ctor' if p['ctor'] else p['ctx'].split()[0]
        print(f"{p['file']}:{p['line']}: [{where}] {' '.join(p['mods'])} {p['name']}")


if __name__ == '__main__':
    main()
