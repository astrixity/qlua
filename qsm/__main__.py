# qsm/__main__.py
"""
qsm CLI entry point

Allows running qsm scripts from the command line, similar to python or lua.
"""

import sys
import argparse
from qsm.compiler import qsmCompiler

def main():
    parser = argparse.ArgumentParser(
        description='QSM Quantum Programming Language',
        prog='qsm'
    )
    parser.add_argument('script', help='QSM source file to compile and run')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output')
    
    # Parse arguments, handling both new argparse and legacy sys.argv fallback
    try:
        args = parser.parse_args()
        script_path = args.script
        debug_mode = args.debug
    except SystemExit:
        # Fallback for legacy usage without argparse
        if len(sys.argv) < 2:
            print("Usage: qsm <script.qsm> [--debug]")
            sys.exit(1)
        script_path = sys.argv[1]
        debug_mode = '--debug' in sys.argv
    
    try:
        with open(script_path, 'r') as f:
            code = f.read()
        
        # Create compiler with debug flag
        compiler = qsmCompiler(debug=debug_mode)
        compiler.compile(code)
        compiler.run_simulation()
        
    except FileNotFoundError:
        print(f"Error: File '{script_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()