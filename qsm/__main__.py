# qsm/__main__.py
"""
qsm CLI entry point

Allows running qsm scripts from the command line, similar to python or lua.
"""

import sys
from qsm.compiler import qsmCompiler

def main():
    if len(sys.argv) < 2:
        print("Usage: qsm <script.qsm>")
        sys.exit(1)
    script_path = sys.argv[1]
    with open(script_path, 'r') as f:
        code = f.read()
    compiler = qsmCompiler()
    compiler.compile(code)
    print("\n[qsm] Quantum Circuit:")
    print(compiler.get_circuit().draw())
    # Only print after the circuit diagram
    compiler.execute_deferred_prints()

if __name__ == "__main__":
    main()
