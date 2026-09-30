import sys
from .cli import compile_main, vm_main, links_main, rust_main

if len(sys.argv) > 1 and sys.argv[1] == 'vm':
    sys.exit(vm_main(sys.argv[2:]))
if len(sys.argv) > 1 and sys.argv[1] == 'rust':
    sys.exit(rust_main(sys.argv[2:]))
if len(sys.argv) > 1 and sys.argv[1] == 'links':
    sys.exit(links_main(sys.argv[2:]))
sys.exit(compile_main())
