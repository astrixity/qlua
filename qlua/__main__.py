# qlua/__main__.py
"""
QLua CLI entry point

Allows running QLua scripts from the command line, similar to python or lua.
"""

import sys
from qlua.compiler import QLuaCompiler

def main():
    if len(sys.argv) < 2:
        print("Usage: qlua <script.qlua>")
        sys.exit(1)
    script_path = sys.argv[1]
    with open(script_path, 'r') as f:
        code = f.read()
    compiler = QLuaCompiler()
    compiler.compile(code)
    print("\n[QLua] Quantum Circuit:")
    print(compiler.get_circuit().draw())

if __name__ == "__main__":
    main()
