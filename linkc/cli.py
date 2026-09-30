"""CLI commands for compiler and LinkVM."""
import argparse
import json
import sys
from pathlib import Path
from .syntax import LinkError
from .compiler import compile_source, optimize, lir
from .vm import run
from .bridge import dump_links
from .rust_backend import generate_rust, compile_rust


def compile_main(argv=None):
    parser = argparse.ArgumentParser(prog='linkc', description='Compile .link source to LinkVM .lbc')
    parser.add_argument('input')
    parser.add_argument('-o', '--output')
    parser.add_argument('--dump-lir', action='store_true')
    parser.add_argument('--dump-links', action='store_true')
    parser.add_argument('--emit-rust', metavar='PATH', help='emit a standalone Rust source file')
    parser.add_argument('--rust-bin', metavar='PATH', help='compile generated Rust to native binary (requires rustc)')
    try:
        args = parser.parse_args(argv)
        original = compile_source(Path(args.input).read_text(encoding='utf-8'))
        output = optimize(original)
        if args.dump_lir:
            print('BEFORE OPTIMIZATION\n' + lir(original) + '\nAFTER OPTIMIZATION\n' + lir(output))
        if args.dump_links:
            print(dump_links(output))
        path = Path(args.output or str(Path(args.input).with_suffix('.lbc')))
        path.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        print(f'Wrote {path}', file=sys.stderr)
        if args.emit_rust:
            Path(args.emit_rust).write_text(generate_rust(output), encoding='utf-8')
            print(f'Wrote Rust source {args.emit_rust}', file=sys.stderr)
        if args.rust_bin:
            bin_path = compile_rust(output, args.rust_bin, source_path=args.emit_rust)
            print(f'Wrote Rust binary {bin_path}', file=sys.stderr)
        return 0
    except (LinkError, OSError, ValueError, RuntimeError) as e:
        print(f'linkc: {e}', file=sys.stderr)
        return 1


def vm_main(argv=None):
    parser = argparse.ArgumentParser(prog='linkvm', description='Execute LinkVM .lbc')
    parser.add_argument('input')
    try:
        args = parser.parse_args(argv)
        program = json.loads(Path(args.input).read_text(encoding='utf-8'))
        answer = run(program)
        print(json.dumps(answer, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (LinkError, OSError, ValueError, KeyError, IndexError, TypeError) as e:
        print(f'linkvm: {e}', file=sys.stderr)
        return 1


def links_main(argv=None):
    parser = argparse.ArgumentParser(prog='linkc links', description='Inspect typed source-to-bytecode Link')
    parser.add_argument('input')
    try:
        args = parser.parse_args(argv)
        program = json.loads(Path(args.input).read_text(encoding='utf-8'))
        print(dump_links(program))
        return 0
    except (LinkError, OSError, ValueError, KeyError, TypeError) as e:
        print(f'linkc links: {e}', file=sys.stderr)
        return 1


def rust_main(argv=None):
    """Compile an existing .lbc file into Rust or a native Rust binary."""
    parser = argparse.ArgumentParser(prog='linkc rust', description='Lower bytecode through the typed opcode-to-Rust Link')
    parser.add_argument('input')
    parser.add_argument('--emit-rust', metavar='PATH')
    parser.add_argument('--rust-bin', metavar='PATH')
    try:
        args = parser.parse_args(argv)
        if not args.emit_rust and not args.rust_bin:
            parser.error('provide --emit-rust and/or --rust-bin')
        program = json.loads(Path(args.input).read_text(encoding='utf-8'))
        if args.emit_rust:
            Path(args.emit_rust).write_text(generate_rust(program), encoding='utf-8')
            print(f'Wrote Rust source {args.emit_rust}', file=sys.stderr)
        if args.rust_bin:
            compiled = compile_rust(program, args.rust_bin, source_path=args.emit_rust)
            print(f'Wrote Rust binary {compiled}', file=sys.stderr)
        return 0
    except (LinkError, RuntimeError, OSError, ValueError, TypeError, KeyError) as e:
        print(f'linkc rust: {e}', file=sys.stderr)
        return 1
