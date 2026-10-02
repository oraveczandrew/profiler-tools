"""Tests for scripts/find_no_jvmfield.py.

Feeds synthetic Kotlin sources through scan_file and asserts which
declarations are reported as missing @JvmField: real class members are,
locals (fun bodies, apply/run initializers), custom accessors, delegated
and value-class-typed properties are not.
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

import contextlib
import io
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
import find_no_jvmfield as fnj


def scan(src, value_classes=()):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'T.kt')
        with open(p, 'w', encoding='utf-8') as f:
            f.write(src)
        old = fnj.VALUE_CLASS_RE
        fnj.VALUE_CLASS_RE = (
            re.compile(r'\b(?:' + '|'.join(value_classes) + r')\b') if value_classes else None)
        try:
            return fnj.scan_file(p)
        finally:
            fnj.VALUE_CLASS_RE = old


def by_name(results):
    return {r['name']: r for r in results}


class ScanTest(unittest.TestCase):
    def test_plain_property_reported(self):
        rs = by_name(scan('internal class A {\n var x: Int = 0\n}\n'))
        self.assertIn('x', rs)
        self.assertFalse(rs['x']['has_jvmfield'])
        self.assertFalse(rs['x'].get('custom_acc'))
        self.assertFalse(rs['x'].get('delegated'))

    def test_jvmfield_same_line_and_prev_line(self):
        rs = by_name(scan('internal class A {\n'
                          ' @JvmField var a: Int = 0\n'
                          ' @JvmField\n var b: Int = 0\n'
                          ' @JvmSynthetic\n @JvmField\n internal var c: Int = 0\n'
                          '}\n'))
        self.assertTrue(rs['a']['has_jvmfield'])
        self.assertTrue(rs['b']['has_jvmfield'])
        self.assertTrue(rs['c']['has_jvmfield'])

    def test_excluded_modifiers(self):
        rs = by_name(scan('internal class A {\n'
                          ' open val a: Int = 0\n'
                          ' override val b: Int = 0\n'
                          ' lateinit var c: String\n'
                          ' const val D: Int = 0\n'
                          '}\n'))
        for n in ('a', 'b', 'c', 'D'):
            self.assertIn(n, rs)  # scanned...
            self.assertTrue(  # ...but excludable via SKIP_MODS
                set(rs[n]['mods']) & fnj.SKIP_MODS)

    def test_delegated_excluded(self):
        rs = by_name(scan('internal class A {\n val x: Int by lazy { 1 }\n}\n'))
        self.assertTrue(rs['x']['delegated'])

    def test_same_line_getter_is_custom_accessor(self):
        rs = by_name(scan('internal class A {\n'
                          ' val repeatIndefinite: Boolean get() = repeatCount == 1\n'
                          '}\n'))
        self.assertTrue(rs['repeatIndefinite']['custom_acc'])

    def test_next_line_private_set_is_custom_accessor(self):
        rs = by_name(scan('internal class A {\n'
                          ' var version: Long = 0L\n private set\n'
                          '}\n'))
        self.assertTrue(rs['version']['custom_acc'])

    def test_next_line_getter_block_is_custom_accessor(self):
        rs = by_name(scan('internal class A {\n'
                          ' val x: Int\n get() = 1\n'
                          '}\n'))
        self.assertTrue(rs['x']['custom_acc'])

    def test_fun_body_locals_ignored(self):
        rs = by_name(scan('internal class A {\n'
                          ' var x: Int = 0\n'
                          ' fun f(): Int {\n val local = 1\n return local\n }\n'
                          '}\n'))
        self.assertIn('x', rs)
        self.assertNotIn('local', rs)

    def test_initializer_lambda_locals_ignored(self):
        rs = by_name(scan('internal class A {\n'
                          ' @JvmField val cache: Map<String, String> = buildMap {\n'
                          ' val key = "k"\n put(key, key)\n }\n'
                          '}\n'))
        self.assertIn('cache', rs)
        self.assertNotIn('key', rs)

    def test_backtick_class_no_phantom(self):
        rs = scan('internal class A {\n'
                  ' fun f(): Int {\n when (1) {\n 1 -> parse(`class`)\n }\n return 0\n }\n'
                  ' fun parse(s: String): Int = 0\n'
                  '}\n')
        self.assertEqual([r['name'] for r in rs], [])

    def test_class_literal_no_phantom(self):
        rs = scan('internal class A {\n'
                  ' val k: String = Foo::class.simpleName ?: ""\n'
                  '}\n')
        self.assertEqual([r['name'] for r in rs], ['k'])

    def test_ctor_params(self):
        rs = by_name(scan('internal class A(internal val keep: String, private val drop: Int) {\n}\n'))
        self.assertTrue(rs['keep']['ctor'])
        self.assertTrue(rs['drop']['ctor'])

    def test_value_class_typed_excluded(self):
        rs = by_name(scan('internal class A {\n'
                          ' val INVALID = Result(0, -1)\n'
                          ' val other: Int = 0\n'
                          '}\n', value_classes=('Result',)))
        self.assertTrue(rs['INVALID']['is_value_type'])
        self.assertFalse(rs['other']['is_value_type'])

    def test_nested_class_scanned_once(self):
        rs = scan('internal class Outer {\n'
                  ' var x: Int = 0\n'
                  ' class Inner {\n var y: Int = 0\n }\n'
                  '}\n')
        self.assertEqual(sorted(r['name'] for r in rs), ['x', 'y'])


def run_main(root):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fnj.main(['--src', root])
    return buf.getvalue()


class MainFilterTest(unittest.TestCase):
    def test_end_to_end_filters(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'A.kt'), 'w', encoding='utf-8') as f:
                f.write('internal class A {\n'
                        ' var keep: Int = 0\n'
                        ' private var hidden: Int = 0\n'
                        ' @JvmField var fine: Int = 0\n'
                        ' open val skip: Int = 0\n'
                        ' val comp: Boolean get() = true\n'
                        '}\n'
                        'interface I {\n var no: Int\n}\n')
            out = run_main(d)
        self.assertIn('missing @JvmField: 1', out)
        self.assertIn('keep', out)
        self.assertNotIn('hidden', out)
        self.assertNotIn('fine', out)
        self.assertNotIn('skip', out)
        self.assertNotIn('comp', out)
        self.assertNotIn('no', out)


if __name__ == '__main__':
    unittest.main()
