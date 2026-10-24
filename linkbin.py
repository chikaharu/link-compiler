#!/usr/bin/env python3
"""
linkbin v0.2 -- direct binary-output Link.

    .link (AST) ──linkbin──▶ x86-64 machine-code byte sequence ──▶ ELF executable

No Rust, no C compiler, no linker in the path: the compiler itself emits
the byte sequence (ELF64 header + machine code), writes it as a file,
and marks it executable.

Scope of this backend (deliberately small, integer dialect):
  - 64-bit signed integers only (no floats, no vectors yet)
  - let / set / for / while / if / block
  - + - * / mod, comparisons, unary minus
  - (print "label" expr ...)  : strings are written raw,
                                integers via a built-in decimal printer
                                (hand-assembled, ends with a newline)

Usage:
    python3 linkbin.py examples/demo_bytes.link -o out/demo_bytes [--run] [--hex]
"""
import argparse
import hashlib
import os
import struct
import subprocess
import sys

BASE = 0x400000          # load address
HDR = 64 + 56            # ELF header + one program header

# ---------------------------------------------------------------- parser
# (same S-expression AST syntax as linkc, with proper string escapes)

class Str(str):
    pass

_ESC = {'n': '\n', 't': '\t', '0': '\0'}

def tokenize(src):
    toks, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
        elif c == ';':
            while i < n and src[i] != '\n':
                i += 1
        elif c in '()':
            toks.append(c); i += 1
        elif c == '"':
            j = i + 1; buf = []
            while j < n and src[j] != '"':
                if src[j] == '\\' and j + 1 < n:
                    buf.append(_ESC.get(src[j + 1], src[j + 1])); j += 2
                else:
                    buf.append(src[j]); j += 1
            if j >= n:
                raise SyntaxError('unterminated string')
            toks.append(Str(''.join(buf))); i = j + 1
        else:
            j = i
            while j < n and not src[j].isspace() and src[j] not in '();"':
                j += 1
            toks.append(src[i:j]); i = j
    return toks

def parse(src):
    toks = tokenize(src); pos = 0
    def one():
        nonlocal pos
        if pos >= len(toks):
            raise SyntaxError('unexpected end of input')
        t = toks[pos]; pos += 1
        if isinstance(t, Str):
            return t
        if t == '(':
            out = []
            while pos < len(toks) and toks[pos] != ')':
                out.append(one())
            if pos >= len(toks):
                raise SyntaxError('missing )')
            pos += 1
            return out
        if t == ')':
            raise SyntaxError('unexpected )')
        try:
            return int(t)
        except ValueError:
            pass
        try:
            f = float(t)
        except ValueError:
            return t
        if f != int(f):
            raise SyntaxError(f'linkbin is the integer dialect; got {t}')
        return int(f)
    forms = []
    while pos < len(toks):
        forms.append(one())
    if len(forms) == 1 and isinstance(forms[0], list) and forms[0] and forms[0][0] == 'program':
        return forms[0][1:]
    return forms

# ---------------------------------------------------------------- assembler

class Asm:
    def __init__(self):
        self.buf = bytearray()
        self.labels = {}
        self.fixups = []        # (pos, kind, key) kind: rel32 | slot | bufend | str
        self.nlabels = 0

    def new_label(self, hint='L'):
        self.nlabels += 1
        return f'{hint}{self.nlabels}'

    def label(self, name):
        self.labels[name] = len(self.buf)

    def emit(self, *bs):
        for b in bs:
            if isinstance(b, int):
                self.buf.append(b)
            else:
                self.buf.extend(b)

    def imm64(self, v):
        self.emit(struct.pack('<q', v))

    def rel32(self, name):
        self.fixups.append((len(self.buf), 'rel32', name))
        self.emit(0, 0, 0, 0)

    def slot_ref(self, idx):
        """Emit an 8-byte placeholder for a variable slot address."""
        self.fixups.append((len(self.buf), 'slot', idx))
        self.emit(0, 0, 0, 0, 0, 0, 0, 0)

    def bufend_ref(self):
        self.fixups.append((len(self.buf), 'bufend', None))
        self.emit(0, 0, 0, 0, 0, 0, 0, 0)

    def str_ref(self, idx):
        self.fixups.append((len(self.buf), 'str', idx))
        self.emit(0, 0, 0, 0, 0, 0, 0, 0)

    def inbuf_ref(self):
        self.fixups.append((len(self.buf), 'inbuf', None))
        self.emit(0, 0, 0, 0, 0, 0, 0, 0)

    # -- instruction helpers (result convention: value in rax, scratch rcx/rdx)
    def mov_rax_imm(self, v):
        self.emit(0x48, 0xB8); self.imm64(v)

    def mov_rax_slot(self, idx):
        self.emit(0x48, 0xA1); self.slot_ref(idx)

    def mov_slot_rax(self, idx):
        self.emit(0x48, 0xA3); self.slot_ref(idx)

    def push_rax(self): self.emit(0x50)
    def pop_rax(self):  self.emit(0x58)
    def pop_rcx(self):  self.emit(0x59)

# ---------------------------------------------------------------- codegen

INTERNALS = ('__inpos', '__inlen', '__ineof', '__rval', '__rsign', '__rseen')

class Gen:
    def __init__(self):
        self.a = Asm()
        self.slots = {}         # name -> slot index
        self.strings = []       # string literals to place in the data area
        for name in INTERNALS:  # reserve stdin-reader state slots first
            self.slot(name)

    def slot(self, name):
        if name not in self.slots:
            self.slots[name] = len(self.slots)
        return self.slots[name]

    def err(self, m):
        raise TypeError(m)

    # ---- expressions: leave value in rax
    def expr(self, node):
        a = self.a
        if isinstance(node, int):
            a.mov_rax_imm(node); return
        if isinstance(node, str) and not isinstance(node, Str):
            a.mov_rax_slot(self.slot(node)); return
        if not isinstance(node, list) or not node:
            self.err(f'bad expression: {node!r}')
        op, args = node[0], node[1:]
        if op == 'read-int':
            a.emit(0xE8); a.rel32('read_int')          # call read_int -> rax
            return
        if op == 'read-eof?':
            a.mov_rax_slot(self.slot('__ineof'))
            return
        if op in ('+', '-', '*', '/', 'mod', '<', '<=', '>', '>=', '==', '!='):
            if op == '-' and len(args) == 1:
                self.expr(args[0]); a.emit(0x48, 0xF7, 0xD8)  # neg rax
                return
            self.expr(args[0]); a.push_rax()
            self.expr(args[1]); a.emit(0x48, 0x89, 0xC1)      # mov rcx, rax
            a.pop_rax()
            if op == '+':   a.emit(0x48, 0x01, 0xC8)                       # add rax, rcx
            elif op == '-': a.emit(0x48, 0x29, 0xC8)                       # sub rax, rcx
            elif op == '*': a.emit(0x48, 0x0F, 0xAF, 0xC1)                 # imul rax, rcx
            elif op == '/': a.emit(0x48, 0x99, 0x48, 0xF7, 0xF9)           # cqo; idiv rcx
            elif op == 'mod':
                a.emit(0x48, 0x99, 0x48, 0xF7, 0xF9, 0x48, 0x89, 0xD0)     # ...; mov rax, rdx
            else:
                a.emit(0x48, 0x39, 0xC8)                                   # cmp rax, rcx
                cc = {'<': 0x9C, '<=': 0x9E, '>': 0x9F, '>=': 0x9D, '==': 0x94, '!=': 0x95}[op]
                a.emit(0x0F, cc, 0xC0)                                     # setcc al
                a.emit(0x48, 0x0F, 0xB6, 0xC0)                             # movzx rax, al
            # fold extra operands for + and * (e.g. (+ a b c))
            for extra in args[2:]:
                a.push_rax(); self.expr(extra)
                a.emit(0x48, 0x89, 0xC1); a.pop_rax()
                a.emit(0x48, 0x01, 0xC8) if op == '+' else a.emit(0x48, 0x0F, 0xAF, 0xC1)
            return
        if op == 'if':
            lelse, lend = a.new_label('else'), a.new_label('endif')
            self.expr(args[0]); a.emit(0x48, 0x85, 0xC0)                   # test rax, rax
            a.emit(0x0F, 0x84); a.rel32(lelse)                              # jz else
            self.expr(args[1]); a.emit(0xE9); a.rel32(lend)                 # jmp end
            a.label(lelse); self.expr(args[2]); a.label(lend)
            return
        self.err(f'unknown operator in linkbin: {op}')

    # ---- built-in: print_int (rax = signed value, writes decimal + '\n')
    def emit_print_int_routine(self):
        a = self.a
        a.label('print_int')
        a.emit(0x53)                                   # push rbx
        a.emit(0x31, 0xF6)                             # xor esi, esi  (sign flag)
        a.emit(0x48, 0x85, 0xC0)                       # test rax, rax
        lpos = a.new_label('pos'); a.emit(0x0F, 0x89); a.rel32(lpos)   # jns pos
        a.emit(0x48, 0xF7, 0xD8)                       # neg rax
        a.emit(0xBE, 1, 0, 0, 0)                       # mov esi, 1
        a.label(lpos)
        a.emit(0x48, 0xBB); a.bufend_ref()             # mov rbx, buf_end
        a.emit(0x48, 0xFF, 0xCB)                       # dec rbx
        a.emit(0xC6, 0x03, 0x0A)                       # mov byte [rbx], '\n'
        ldigit = a.new_label('digit'); a.label(ldigit)
        a.emit(0x48, 0xB9); a.imm64(10)                # mov rcx, 10
        a.emit(0x48, 0x99)                             # cqo
        a.emit(0x48, 0xF7, 0xF9)                       # idiv rcx
        a.emit(0x48, 0x83, 0xC2, 0x30)                 # add rdx, '0'
        a.emit(0x48, 0xFF, 0xCB)                       # dec rbx
        a.emit(0x88, 0x13)                             # mov [rbx], dl
        a.emit(0x48, 0x85, 0xC0)                       # test rax, rax
        a.emit(0x0F, 0x85); a.rel32(ldigit)             # jnz digit
        a.emit(0x48, 0x85, 0xF6)                       # test rsi, rsi
        lnosign = a.new_label('nosign'); a.emit(0x0F, 0x84); a.rel32(lnosign)
        a.emit(0x48, 0xFF, 0xCB)                       # dec rbx
        a.emit(0xC6, 0x03, 0x2D)                       # mov byte [rbx], '-'
        a.label(lnosign)
        a.emit(0x48, 0xB8); a.imm64(1)                 # mov rax, 1 (sys_write)
        a.emit(0xBF, 1, 0, 0, 0)                       # mov edi, 1
        a.emit(0x48, 0x89, 0xDE)                       # mov rsi, rbx
        a.emit(0x48, 0xBA); a.bufend_ref()             # mov rdx, buf_end
        a.emit(0x48, 0x29, 0xDA)                       # sub rdx, rbx
        a.emit(0x0F, 0x05)                             # syscall
        a.emit(0x5B)                                   # pop rbx
        a.emit(0xC3)                                   # ret

    # ---- built-in stdin reader: input arrives as data, not as code
    # next_char -> rax = byte (0..255) or -1 at EOF. Buffered read(0, buf, 4096).
    def emit_next_char_routine(self):
        a = self.a
        S = lambda n: self.slot(n)
        a.label('next_char')
        # if inpos < inlen -> have
        a.mov_rax_slot(S('__inpos')); a.push_rax()
        a.mov_rax_slot(S('__inlen')); a.emit(0x48, 0x89, 0xC1)   # mov rcx, rax
        a.pop_rax()
        a.emit(0x48, 0x39, 0xC8)                                  # cmp rax, rcx
        lhave = a.new_label('have'); a.emit(0x0F, 0x8C); a.rel32(lhave)  # jl have
        # refill: sys_read(0, inbuf, 4096)
        a.emit(0x48, 0xB8); a.imm64(0)                            # mov rax, 0
        a.emit(0xBF, 0, 0, 0, 0)                                  # mov edi, 0
        a.emit(0x48, 0xBE); a.inbuf_ref()                         # mov rsi, inbuf
        a.emit(0x48, 0xBA); a.imm64(4096)                         # mov rdx, 4096
        a.emit(0x0F, 0x05)                                        # syscall
        a.emit(0x48, 0x83, 0xF8, 0x00)                            # cmp rax, 0
        leof = a.new_label('ceof'); a.emit(0x0F, 0x8E); a.rel32(leof)    # jle eof
        a.mov_slot_rax(S('__inlen'))
        a.emit(0x31, 0xC0)                                        # xor eax, eax
        a.mov_slot_rax(S('__inpos'))
        a.label(lhave)
        a.mov_rax_slot(S('__inpos')); a.push_rax()
        a.emit(0x48, 0xFF, 0xC0)                                  # inc rax
        a.mov_slot_rax(S('__inpos'))
        a.pop_rax()
        a.emit(0x48, 0xB9); a.inbuf_ref()                         # mov rcx, inbuf
        a.emit(0x48, 0x0F, 0xB6, 0x04, 0x01)                      # movzx rax, byte [rcx+rax]
        a.emit(0xC3)                                              # ret
        a.label(leof)
        a.mov_rax_imm(-1)
        a.emit(0xC3)                                              # ret

    # read_int -> rax = next signed decimal integer from stdin.
    # Whitespace-separated; '-' sign supported. At end of input it returns 0
    # and sets the flag readable as (read-eof?). A number that ends exactly
    # at EOF is still returned; the flag is set by the *next* call.
    def emit_read_int_routine(self):
        a = self.a
        S = lambda n: self.slot(n)
        def set_slot(name, v):
            a.mov_rax_imm(v); a.mov_slot_rax(S(name))
        a.label('read_int')
        set_slot('__rsign', 1); set_slot('__rseen', 0); set_slot('__rval', 0)
        lskip = a.new_label('skip'); a.label(lskip)
        a.emit(0xE8); a.rel32('next_char')
        a.emit(0x48, 0x83, 0xF8, 0xFF)                            # cmp rax, -1
        leof = a.new_label('rieof'); a.emit(0x0F, 0x84); a.rel32(leof)
        for ws in (32, 9, 10, 13):                                # space \t \n \r
            a.emit(0x48, 0x83, 0xF8, ws)
            a.emit(0x0F, 0x84); a.rel32(lskip)
        a.emit(0x48, 0x83, 0xF8, 0x2D)                            # cmp rax, '-'
        ldig = a.new_label('digit'); a.emit(0x0F, 0x85); a.rel32(ldig)
        set_slot('__rsign', -1)
        a.emit(0xE8); a.rel32('next_char')
        a.emit(0x48, 0x83, 0xF8, 0xFF)
        a.emit(0x0F, 0x84); a.rel32(leof)
        a.label(ldig)
        a.emit(0x48, 0x83, 0xF8, 0x30)                            # cmp rax, '0'
        ldone = a.new_label('done'); a.emit(0x0F, 0x8C); a.rel32(ldone)
        a.emit(0x48, 0x83, 0xF8, 0x39)                            # cmp rax, '9'
        a.emit(0x0F, 0x8F); a.rel32(ldone)
        a.emit(0x48, 0x83, 0xE8, 0x30)                            # sub rax, '0'
        a.push_rax()
        a.mov_rax_slot(S('__rval'))
        a.emit(0x48, 0xB9); a.imm64(10)                           # mov rcx, 10
        a.emit(0x48, 0x0F, 0xAF, 0xC1)                            # imul rax, rcx
        a.pop_rcx(); a.emit(0x48, 0x01, 0xC8)                     # add rax, rcx
        a.mov_slot_rax(S('__rval'))
        set_slot('__rseen', 1)
        a.emit(0xE8); a.rel32('next_char')
        a.emit(0x48, 0x83, 0xF8, 0xFF)
        lfin = a.new_label('finish'); a.emit(0x0F, 0x84); a.rel32(lfin)
        a.emit(0xE9); a.rel32(ldig)                               # jmp digit
        a.label(ldone)                                            # terminator char: unget it
        a.mov_rax_slot(S('__inpos')); a.emit(0x48, 0xFF, 0xC8)   # dec rax
        a.mov_slot_rax(S('__inpos'))
        a.label(lfin)
        a.mov_rax_slot(S('__rseen'))
        a.emit(0x48, 0x83, 0xF8, 0x00)                            # cmp rax, 0
        a.emit(0x0F, 0x84); a.rel32(leof)                         # no digits -> EOF path
        a.mov_rax_slot(S('__rval'))
        a.push_rax()
        a.mov_rax_slot(S('__rsign')); a.emit(0x48, 0x89, 0xC1)    # mov rcx, rax
        a.pop_rax()
        a.emit(0x48, 0x0F, 0xAF, 0xC1)                            # imul rax, rcx
        a.emit(0xC3)                                              # ret
        a.label(leof)
        set_slot('__ineof', 1)
        a.emit(0x31, 0xC0)                                        # xor eax, eax
        a.emit(0xC3)                                              # ret

    def write_str(self, s):
        a = self.a
        idx = len(self.strings); self.strings.append(s)
        data = s.encode()
        a.emit(0x48, 0xB8); a.imm64(1)                 # mov rax, 1
        a.emit(0xBF, 1, 0, 0, 0)                       # mov edi, 1
        a.emit(0x48, 0xBE); a.str_ref(idx)             # mov rsi, str addr
        a.emit(0x48, 0xBA); a.imm64(len(data))         # mov rdx, len
        a.emit(0x0F, 0x05)                             # syscall

    # ---- statements
    def stmt(self, node):
        a = self.a
        if not (isinstance(node, list) and node and isinstance(node[0], str)):
            self.expr(node); return
        op = node[0]
        if op in ('let', 'set'):
            self.expr(node[2]); a.mov_slot_rax(self.slot(node[1])); return
        if op == 'print':
            for arg in node[1:]:
                if isinstance(arg, Str):
                    self.write_str(arg)
                else:
                    self.expr(arg); a.emit(0xE8); a.rel32('print_int')
            return
        if op == 'for':
            var = node[1]
            if len(node) >= 5 and not isinstance(node[3], list):
                lo, hi, body = node[2], node[3], node[4:]
            else:
                lo, hi, body = 0, node[2], node[3:]
            end_slot = self.slot(a.new_label('__end'))
            self.expr(lo); a.mov_slot_rax(self.slot(var))
            self.expr(hi); a.mov_slot_rax(end_slot)
            ltop, lend = a.new_label('fortop'), a.new_label('forend')
            a.label(ltop)
            a.mov_rax_slot(self.slot(var))
            a.push_rax()
            a.mov_rax_slot(end_slot); a.emit(0x48, 0x89, 0xC1); a.pop_rax()
            a.emit(0x48, 0x39, 0xC8)                   # cmp rax, rcx
            a.emit(0x0F, 0x8D); a.rel32(lend)           # jge end
            for s in body:
                self.stmt(s)
            a.mov_rax_slot(self.slot(var))
            a.emit(0x48, 0xFF, 0xC0)                   # inc rax
            a.mov_slot_rax(self.slot(var))
            a.emit(0xE9); a.rel32(ltop)                # jmp top
            a.label(lend); return
        if op == 'while':
            ltop, lend = a.new_label('wtop'), a.new_label('wend')
            a.label(ltop)
            self.expr(node[1]); a.emit(0x48, 0x85, 0xC0)
            a.emit(0x0F, 0x84); a.rel32(lend)
            for s in node[2:]:
                self.stmt(s)
            a.emit(0xE9); a.rel32(ltop)
            a.label(lend); return
        if op == 'if':
            lelse, lend = a.new_label('selse'), a.new_label('send')
            self.expr(node[1]); a.emit(0x48, 0x85, 0xC0)
            a.emit(0x0F, 0x84); a.rel32(lelse)
            self.stmt(node[2])
            if len(node) > 3:
                a.emit(0xE9); a.rel32(lend)
                a.label(lelse); self.stmt(node[3]); a.label(lend)
            else:
                a.label(lelse)
            return
        if op == 'block':
            for s in node[1:]:
                self.stmt(s)
            return
        self.expr(node)

    def compile(self, stmts):
        a = self.a
        for s in stmts:
            self.stmt(s)
        # exit(0): mov rax, 60; xor edi, edi; syscall
        a.emit(0x48, 0xB8); a.imm64(60)
        a.emit(0x31, 0xFF)
        a.emit(0x0F, 0x05)
        self.emit_print_int_routine()
        self.emit_next_char_routine()
        self.emit_read_int_routine()
        return self.link()

    def link(self):
        a = self.a
        code = bytes(a.buf)
        nslots = len(self.slots)
        data_vaddr = BASE + HDR + len(code)
        buf_addr = data_vaddr + nslots * 8
        buf_end = buf_addr + 32
        str_addr = buf_end
        str_offsets = []
        for s in self.strings:
            str_offsets.append(str_addr)
            str_addr += len(s.encode())
        inbuf_addr = str_addr
        data_size = (inbuf_addr + 4096) - data_vaddr

        patched = bytearray(code)
        for pos, kind, key in a.fixups:
            if kind == 'rel32':
                target = BASE + HDR + a.labels[key]
                rel = target - (BASE + HDR + pos + 4)
                struct.pack_into('<i', patched, pos, rel)
            elif kind == 'slot':
                struct.pack_into('<Q', patched, pos, data_vaddr + key * 8)
            elif kind == 'bufend':
                struct.pack_into('<Q', patched, pos, buf_end)
            elif kind == 'str':
                struct.pack_into('<Q', patched, pos, str_offsets[key])
            elif kind == 'inbuf':
                struct.pack_into('<Q', patched, pos, inbuf_addr)

        data = bytearray(data_size)
        off = nslots * 8 + 32
        for s in self.strings:
            b = s.encode()
            data[off:off + len(b)] = b
            off += len(b)

        total = HDR + len(patched) + len(data)
        ehdr = struct.pack(
            '<16sHHIQQQIHHHHHH',
            b'\x7fELF' + bytes([2, 1, 1, 0]) + bytes(8),
            2, 0x3E, 1, BASE + HDR, 64, 0, 0, 64, 56, 1, 0, 0, 0)
        phdr = struct.pack('<IIQQQQQQ', 1, 7, 0, BASE, BASE, total, total, 0x1000)
        return ehdr + phdr + bytes(patched) + bytes(data)

# ---------------------------------------------------------------- driver

def hexdump(data):
    lines = []
    for off in range(0, len(data), 16):
        chunk = data[off:off + 16]
        hexs = ' '.join(f'{b:02x}' for b in chunk)
        asc = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        lines.append(f'{off:08x}  {hexs:<47}  |{asc}|')
    return '\n'.join(lines)

def main():
    ap = argparse.ArgumentParser(description='linkbin: AST -> machine-code bytes -> ELF')
    ap.add_argument('source')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--hex', action='store_true', help='print the full byte sequence')
    ap.add_argument('--boundary', action='store_true',
                    help='run linkir boundary reduction (integer mode) before codegen; '
                         'programs it cannot prove equivalent compile as-is')
    a = ap.parse_args()

    with open(a.source) as f:
        stmts = parse(f.read())
    if a.boundary:
        try:
            from linkir import Lower, count_nodes
            from linkemit import emit_program, intify_program, check_int_literals
            import linkc as _linkc
            check_int_literals(stmts)
            lowered = Lower(mode='boundary', integer=True).program(stmts)
            raw = Lower(mode='none', integer=True).program(stmts)
            def localize(n):
                if isinstance(n, _linkc.Str):
                    return Str(str(n))
                if isinstance(n, list):
                    return [localize(x) for x in n]
                return n
            stmts = [localize(s) for s in intify_program(emit_program(lowered, stmts))]
        except Exception as e:
            print(f'[linkbin] boundary reduction not applicable ({e}); compiling as-is',
                  file=sys.stderr)
        else:
            print(f'[linkbin] boundary reduction: {count_nodes(raw)} -> '
                  f'{count_nodes(lowered)} 1x1 Links')
    blob = Gen().compile(stmts)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'wb') as f:
        f.write(blob)
    os.chmod(a.out, 0o755)
    hex_path = a.out + '.hex'
    with open(hex_path, 'w') as f:
        f.write(hexdump(blob) + '\n')

    print(f'[linkbin] AST -> byte sequence : {len(blob)} bytes, '
          f'sha256 {hashlib.sha256(blob).hexdigest()[:16]}...')
    print(f'[linkbin] bytes -> ELF file    : {a.out} (hex dump: {hex_path})')
    if a.hex:
        print(hexdump(blob))
    if a.run:
        print('[linkbin] run:')
        return subprocess.run([os.path.abspath(a.out)]).returncode
    return 0

if __name__ == '__main__':
    sys.exit(main())
