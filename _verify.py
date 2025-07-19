# _verify.py
from qsm.compiler import qsmCompiler
file_path = r'D:\Quantum Programming\qsm\examples\quantum_phase_estimation.qsm'
with open(file_path, 'r') as f:
    code = f.read()
compiler = qsmCompiler()
compiler.compile(code)
compiler.execute_deferred_prints()
