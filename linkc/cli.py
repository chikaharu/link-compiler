"""CLI commands for compiler and LinkVM."""
import argparse,json,sys
from pathlib import Path
from .syntax import LinkError
from .compiler import compile_source,optimize,lir
from .vm import run
from .bridge import dump_links
from .rust_backend import generate_rust,compile_rust
def compile_main(argv=None):
    p=argparse.ArgumentParser(prog='linkc'); p.add_argument('input'); p.add_argument('-o','--output'); p.add_argument('--dump-lir',action='store_true'); p.add_argument('--dump-links',action='store_true'); p.add_argument('--emit-rust'); p.add_argument('--rust-bin')
    try:
        a=p.parse_args(argv); original=compile_source(Path(a.input).read_text(encoding='utf-8')); out=optimize(original)
        if a.dump_lir:print('BEFORE OPTIMIZATION\n'+lir(original)+'\nAFTER OPTIMIZATION\n'+lir(out))
        if a.dump_links:print(dump_links(out))
        path=Path(a.output or str(Path(a.input).with_suffix('.lbc'))); path.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        if a.emit_rust:Path(a.emit_rust).write_text(generate_rust(out),encoding='utf-8')
        if a.rust_bin:compile_rust(out,a.rust_bin,source_path=a.emit_rust)
        return 0
    except (LinkError,OSError,ValueError,RuntimeError) as e: print(f'linkc: {e}',file=sys.stderr); return 1
def vm_main(argv=None):
    p=argparse.ArgumentParser(prog='linkvm'); p.add_argument('input')
    try:
        a=p.parse_args(argv); print(json.dumps(run(json.loads(Path(a.input).read_text())),indent=2,allow_nan=False)); return 0
    except Exception as e: print(f'linkvm: {e}',file=sys.stderr); return 1
def links_main(argv=None):
    p=argparse.ArgumentParser(prog='linkc links'); p.add_argument('input')
    try:a=p.parse_args(argv); print(dump_links(json.loads(Path(a.input).read_text()))); return 0
    except Exception as e:print(f'linkc links: {e}',file=sys.stderr); return 1
def rust_main(argv=None):
    p=argparse.ArgumentParser(prog='linkc rust'); p.add_argument('input'); p.add_argument('--emit-rust'); p.add_argument('--rust-bin')
    try:
        a=p.parse_args(argv)
        if not a.emit_rust and not a.rust_bin:p.error('provide --emit-rust and/or --rust-bin')
        program=json.loads(Path(a.input).read_text())
        if a.emit_rust:Path(a.emit_rust).write_text(generate_rust(program),encoding='utf-8')
        if a.rust_bin:compile_rust(program,a.rust_bin,source_path=a.emit_rust)
        return 0
    except Exception as e:print(f'linkc rust: {e}',file=sys.stderr); return 1
