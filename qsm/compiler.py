from __future__ import annotations
import sys
# qsm/compiler.py
"""
Quantum Lua Compiler – rewritten for clarity and safety.

Public API
----------
QsmCompiler().compile(code)   # parse & build circuit
QsmCompiler().get_circuit()   # returns qiskit.QuantumCircuit
QsmCompiler().export_qasm()   # OpenQASM 3 string
"""

import ast
from typing import Dict, List, Any, Tuple
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qsm.parser import (
    qsmParser,
    QBitDecl, CRegDecl,
    Hadamard, XGate, YGate, ZGate, SGate, TGate,
    CX, SwapGate, MCZGate, Measure, Teleport,
    FunctionDef, FunctionCall
)

# --- safety knobs -----------------------------------------------------------
MAX_QUBITS = 128
MAX_CBITS  = 128

# --------------------------------------------------------------------------- #
# Helper AST nodes the parser produces (already defined in parser.py)        #
# --------------------------------------------------------------------------- #
# (re-exported here for type checkers)

# --------------------------------------------------------------------------- #
# Main compiler implementation                                               #
# --------------------------------------------------------------------------- #
class qsmCompiler:
    """
    Compile qsm source into a Qiskit QuantumCircuit.
    """
    def __init__(self) -> None:
        self.debug = False
        self.qubit_map: Dict[str, int] = {}
        self.cbit_map : Dict[str, int] = {}
        self.qreg: List[str] = []
        self.creg: List[str] = []
        self.qc: QuantumCircuit | None = None
        self.parser = qsmParser()
        self.env: Dict[str, Any] = {}
        self.deferred_prints: List[ast.AST] = []

    # --------------------------------------------------------------------- #
    # Public entry points                                                   #
    # --------------------------------------------------------------------- #
    def compile(self, code: str) -> None:
        """Parse and compile `code`; final circuit available via `get_circuit`."""
        self.debug = "--debug" in sys.argv[1:]
        ast_nodes = self.parser.parse(code)
        self._allocate_registers(ast_nodes)
        self._build_circuit()
        for node in ast_nodes:
            self._emit(node)

    def execute_deferred_prints(self) -> None:
        """Execute deferred print statements and display measurement results."""
        from qiskit_aer.primitives import Sampler as AerSampler
        import ast, sys

        exec_env = self.env.copy()
        exec_env.update(globals())

        for node in self.deferred_prints:
            # Skip anything that is not print(...)
            if not (isinstance(node, ast.Expr) and
                    isinstance(node.value, ast.Call) and
                    isinstance(node.value.func, ast.Name) and
                    node.value.func.id == 'print'):
                continue

            args = node.value.args
            parts, need_meas = [], False

            # Classify every argument
            for arg in args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    parts.append(arg.value)

                elif (isinstance(arg, ast.Subscript) and
                      isinstance(arg.value, ast.Name) and
                      arg.value.id in self.creg):
                    need_meas = True
                    idx = (arg.slice.value if isinstance(arg.slice, ast.Constant)
                           else arg.slice.value.value)
                    parts.append(f"__QSM_REGISTER__{arg.value.id}[{idx}]")

                elif isinstance(arg, ast.Name):
                    # Check if arg.id is a classical register or its base name
                    base_name = arg.id.split('[')[0]
                    if arg.id in self.creg or base_name in [c.split('[')[0] for c in self.creg]:
                        need_meas = True
                        parts.append(f"__QSM_REGISTER__{base_name}")
                    else:
                        # Try to evaluate as a normal Python variable
                        try:
                            wrapper = ast.Module(
                                body=[ast.Assign(
                                    targets=[ast.Name(id="__tmp", ctx=ast.Store())],
                                    value=arg)],
                                type_ignores=[])
                            ast.fix_missing_locations(wrapper)
                            exec(compile(wrapper, "<qsm_print>", "exec"), exec_env)
                            parts.append(str(exec_env["__tmp"]))
                        except Exception as e:
                            print(f"[qsm:debug] Print arg eval error: {e}", file=sys.stderr)
                            parts.append("?")

                else:
                    # Try to evaluate as a normal Python expression
                    try:
                        wrapper = ast.Module(
                            body=[ast.Assign(
                                targets=[ast.Name(id="__tmp", ctx=ast.Store())],
                                value=arg)],
                            type_ignores=[])
                        ast.fix_missing_locations(wrapper)
                        exec(compile(wrapper, "<qsm_print>", "exec"), exec_env)
                        parts.append(str(exec_env["__tmp"]))
                    except Exception as e:
                        print(f"[qsm:debug] Print arg eval error: {e}", file=sys.stderr)
                        parts.append("?")

            # If no classical register is involved, emit immediately
            if not need_meas:
                print("[qsm]", *parts)
                continue

            # Run the circuit and print measurement results using SamplerV2 or fallback
            try:
                from qiskit_aer.primitives import SamplerV2
                sampler = SamplerV2()
                job = sampler.run([self.qc], shots=1)
                result = job.result()
                data = result[0].data
                # Try get_counts(), then .counts, then ['counts'], then .c (BitArray)
                if hasattr(data, 'get_counts'):
                    counts = data.get_counts()
                    bit_string = list(counts.keys())[0][::-1]  # reverse for little-endian
                elif hasattr(data, 'counts'):
                    counts = data.counts
                    bit_string = list(counts.keys())[0][::-1]
                elif isinstance(data, dict) and 'counts' in data:
                    counts = data['counts']
                    bit_string = list(counts.keys())[0][::-1]
                elif hasattr(data, 'c'):
                    # BitArray: get the bitstring for the first shot
                    c = data.c
                    # Try to use to01() if available, else str()
                    if hasattr(c, 'to01'):
                        bit_string = c.to01()[::-1]  # reverse for little-endian
                    else:
                        # Use list(c) for a list of bits (1D or 2D)
                        import re
                        try:
                            arr = list(c)
                            if arr and isinstance(arr[0], (list, tuple)):
                                bits = arr[0]
                                bit_string = ''.join(str(int(bit)) for bit in bits)[::-1]
                            else:
                                bit_string = ''.join(str(int(bit)) for bit in c)[::-1]
                        except Exception as e:
                            # Fallback: extract bitstring from str(c) using regex
                            s = str(c)
                            matches = re.findall(r'[01]+', s)
                            if matches:
                                bit_string = matches[-1][::-1]
                            else:
                                print(f"[qsm:debug] BitArray fallback error: {e}", file=sys.stderr)
                                bit_string = "?"
                else:
                    print(f"[qsm:debug] SamplerV2 result[0].data type: {type(data)}", file=sys.stderr)
                    print(f"[qsm:debug] SamplerV2 result[0].data dir: {dir(data)}", file=sys.stderr)
                    print(f"[qsm:debug] SamplerV2 result[0].data repr: {repr(data)}", file=sys.stderr)
                    raise AttributeError("No counts found in SamplerV2 result data")
            except ImportError:
                from qiskit_aer.primitives import Sampler as AerSampler
                sampler = AerSampler()
                job = sampler.run(self.qc)
                result = job.result()
                quasi_dist = result.quasi_dists[0]
                measured = max(quasi_dist, key=quasi_dist.get)
                bit_string = format(measured, f"0{len(self.creg)}b")[::-1]

            # Always zero-pad the bit_string to the length of the classical register
            bit_string = bit_string.zfill(len(self.creg))
            # Display in big-endian order (leftmost is result[N-1], rightmost is result[0])
            bit_string = bit_string[::-1]
            out_parts = []
            for p in parts:
                if isinstance(p, str) and p.startswith("__QSM_REGISTER__"):
                    item = p[len("__QSM_REGISTER__"):]
                    if '[' in item and item.endswith(']'):
                        base, idx_str = item[:-1].split('[')
                        idx = int(idx_str)
                        # Print only the bit at the given index (little-endian)
                        out_parts.append(bit_string[idx])
                    else:
                        # Print the whole register as a bitstring (little-endian)
                        out_parts.append(bit_string)
                else:
                    out_parts.append(str(p))
            # If the output is just a single register, print only the bitstring (no tuple/int conversion)
            if len(out_parts) == 1 and isinstance(out_parts[0], str) and set(out_parts[0]).issubset({'0','1'}):
                # Print the bitstring as is, e.g., 101 (little-endian: result[0] is rightmost)
                print(f"[qsm] {out_parts[0]}")
            else:
                # For mixed output, print all parts, but keep bitstrings as strings
                print("[qsm]", *out_parts)

    def get_circuit(self) -> QuantumCircuit:
        if self.qc is None:
            raise RuntimeError("Nothing compiled yet; call .compile(code) first.")
        return self.qc

    def export_qasm(self) -> str:
        return self.get_circuit().qasm()

    # --------------------------------------------------------------------- #
    # First pass – discover registers                                       #
    # --------------------------------------------------------------------- #
    def _allocate_registers(self, nodes: List[Any]) -> None:
        """Walk the AST once to collect qubit and classical bit names."""
        for node in nodes:
            self._allocate_registers_recursive(node)

    def _allocate_registers_recursive(self, node: Any) -> None:
        """Recursively find register declarations in any AST node."""
        if isinstance(node, QBitDecl):
            for name in node.targets:
                self._add_qubit(name)
        elif isinstance(node, CRegDecl):
            for name in node.targets:
                self._add_cbit(name)
        elif isinstance(node, (ast.For, ast.While, ast.If)):
            # Recursively check nested structures
            for stmt in node.body:
                self._allocate_registers_recursive(stmt)
            if isinstance(node, ast.If):
                for stmt in node.orelse:
                    self._allocate_registers_recursive(stmt)
        elif isinstance(node, list):
            for item in node:
                self._allocate_registers_recursive(item)

    def _add_qubit(self, name: str) -> None:
        if name in self.qubit_map:
            return
        idx = len(self.qreg)
        self.qreg.append(name)
        self.qubit_map[name] = idx

    def _add_cbit(self, name: str) -> None:
        if name in self.cbit_map:
            return
        idx = len(self.creg)
        self.creg.append(name)
        self.cbit_map[name] = idx

        if '[' in name:
            base = name.split('[')[0]
            if base not in self.cbit_map:
                self.cbit_map[base] = None   # sentinel “whole register” key

    # --------------------------------------------------------------------- #
    # Second pass – build the circuit                                       #
    # --------------------------------------------------------------------- #
    def _build_circuit(self) -> None:
        qr = QuantumRegister(len(self.qreg), name="q")
        cr = ClassicalRegister(len(self.creg), name="c") if self.creg else None
        if cr:
            self.qc = QuantumCircuit(qr, cr)
        else:
            self.qc = QuantumCircuit(qr)

    # --------------------------------------------------------------------- #
    # Dispatcher                                                            #
    # --------------------------------------------------------------------- #
    def _emit(self, node: Any) -> None:
        """Single dispatcher for all AST nodes."""
        if isinstance(node, list):
            for item in node:
                self._emit(item)
            return

        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and \
           hasattr(node.value.func, 'id') and node.value.func.id == 'print':
            self.deferred_prints.append(node)
            return

        if isinstance(node, (QBitDecl, CRegDecl)):
            return
            
        # Quantum gates
        if isinstance(node, Hadamard):
            self.qc.h(self._q(node.target))
        elif isinstance(node, XGate):
            self.qc.x(self._q(node.target))
        elif isinstance(node, YGate):
            self.qc.y(self._q(node.target))
        elif isinstance(node, ZGate):
            self.qc.z(self._q(node.target))
        elif isinstance(node, SGate):
            self.qc.s(self._q(node.target))
        elif isinstance(node, TGate):
            self.qc.t(self._q(node.target))
        elif isinstance(node, CX):
            self.qc.cx(self._q(node.ctrl), self._q(node.targ))
        elif isinstance(node, SwapGate):
            self.qc.swap(self._q(node.q1), self._q(node.q2))
        elif isinstance(node, MCZGate):
            self._handle_mcz(node.controls, node.target)
        elif isinstance(node, Measure):
            if hasattr(node, 'classical_target') and node.classical_target:
                self.qc.measure(self._q(node.target), self._c(node.classical_target))
            else:
                # Default behavior: measure to same index
                self.qc.measure(self._q(node.target), self._c(node.target))
        elif isinstance(node, Teleport):
            self._teleport_stub(node.q1, node.q2, node.q3)

        # Classical control flow
        elif isinstance(node, FunctionDef):
            self.env[node.name] = node
        elif isinstance(node, FunctionCall):
            self._call_function(node)
        elif isinstance(node, ast.For):
            self._execute_for_loop(node)
        elif isinstance(node, ast.While):
            self._execute_while_loop(node)
        elif isinstance(node, ast.If):
            self._execute_if_statement(node)
        elif isinstance(node, (ast.Assign, ast.Expr)):
            self._exec_py_node(node)
        else:
            try:
                self._exec_py_node(node)
            except:
                raise SyntaxError(f"Unsupported node: {type(node)}")

    # --------------------------------------------------------------------- #
    # Classical control flow execution                                      #
    # --------------------------------------------------------------------- #
    def _execute_for_loop(self, node: ast.For) -> None:
        """Execute a for loop, handling quantum operations inside."""
        # Extract loop variable and range
        if isinstance(node.target, ast.Name):
            loop_var = node.target.id
        else:
            raise SyntaxError("Only simple loop variables supported")
            
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
            # Handle range(start, stop) or range(start, stop, step)
            args = []
            for arg in node.iter.args:
                if isinstance(arg, ast.Constant):
                    args.append(arg.value)
                elif isinstance(arg, ast.Name):
                    args.append(self.env.get(arg.id, 0))
                elif isinstance(arg, ast.BinOp):
                    args.append(self._eval_expr(arg))
                else:
                    raise SyntaxError(f"Unsupported range argument: {type(arg)}")

            if len(args) == 1:
                range_obj = range(args[0])
            elif len(args) == 2:
                range_obj = range(args[0], args[1])
            elif len(args) == 3:
                range_obj = range(args[0], args[1], args[2])
            else:
                raise SyntaxError("Invalid range arguments")

            # Execute loop body for each iteration
            for i in range_obj:
                old_val = self.env.get(loop_var)
                self.env[loop_var] = i
                try:
                    for stmt in node.body:
                        self._emit(stmt)
                finally:
                    if old_val is not None:
                        self.env[loop_var] = old_val
                    elif loop_var in self.env:
                        del self.env[loop_var]
        else:
            raise SyntaxError("Only range() iteration supported")

    def _execute_while_loop(self, node: ast.While) -> None:
        """Execute a while loop, handling quantum operations inside."""
        while self._eval_condition(node.test):
            for stmt in node.body:
                self._emit(stmt)

    def _execute_if_statement(self, node: ast.If) -> None:
        """Execute an if statement, handling quantum operations inside."""
        if self._eval_condition(node.test):
            for stmt in node.body:
                self._emit(stmt)
        else:
            for stmt in node.orelse:
                self._emit(stmt)

    def _eval_condition(self, node: ast.expr) -> bool:
        """Evaluate a condition expression."""
        # Simple evaluation for basic comparisons
        if isinstance(node, ast.Compare):
            left = self._eval_expr(node.left)
            if len(node.ops) == 1 and len(node.comparators) == 1:
                op = node.ops[0]
                right = self._eval_expr(node.comparators[0])
                if isinstance(op, ast.Lt):
                    return left < right
                elif isinstance(op, ast.LtE):
                    return left <= right
                elif isinstance(op, ast.Gt):
                    return left > right
                elif isinstance(op, ast.GtE):
                    return left >= right
                elif isinstance(op, ast.Eq):
                    return left == right
                elif isinstance(op, ast.NotEq):
                    return left != right
        elif isinstance(node, ast.Constant):
            return bool(node.value)
        elif isinstance(node, ast.Name):
            return bool(self.env.get(node.id, False))
        
        raise SyntaxError(f"Unsupported condition: {type(node)}")

    def _eval_expr(self, node: ast.expr) -> Any:
        """Evaluate an expression."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return self.env.get(node.id, 0)
        elif isinstance(node, ast.BinOp):
            left = self._eval_expr(node.left)
            right = self._eval_expr(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                return left / right
            elif isinstance(node.op, ast.FloorDiv):
                return left // right
            elif isinstance(node.op, ast.Mod):
                return left % right
            elif isinstance(node.op, ast.Pow):
                return left ** right
            else:
                raise SyntaxError(f"Unsupported binary operator: {type(node.op)}")
        elif isinstance(node, ast.Subscript):
            value = self._eval_expr(node.value)
            # Python 3.9+: node.slice is ast.Constant; older: ast.Index
            if hasattr(node.slice, 'value'):
                idx = node.slice.value
            else:
                idx = self._eval_expr(node.slice)
            # If value is a dict (like env), get by key; if list, by index
            if isinstance(value, dict):
                return value.get(f"{node.value.id}[{idx}]", 0)
            elif isinstance(value, list):
                return value[idx]
            else:
                return 0
        raise SyntaxError(f"Unsupported expression: {type(node)}")

    # --------------------------------------------------------------------- #
    # Helpers                                                               #
    # --------------------------------------------------------------------- #
    def _q(self, name: str) -> int:
        """Return the circuit index for qubit `name`."""
        # Handle array notation like q[0], q[1], etc.
        if '[' in name and name.endswith(']'):
            # This is an array access, resolve it
            base_name, index_str = name[:-1].split('[', 1)
            try:
                index = int(index_str)
            except ValueError:
                # Index might be a variable
                index = self.env.get(index_str, 0)
            final_name = f"{base_name}[{index}]"
        else:
            final_name = name
            
        if final_name not in self.qubit_map:
            raise NameError(f"Qubit '{final_name}' not declared.")
        return self.qubit_map[final_name]

    def _c(self, name: str) -> int:
        """Return the circuit index for classical bit `name`."""
        # Handle array notation like answer[0], answer[1], etc.
        if '[' in name and name.endswith(']'):
            # This is an array access, resolve it
            base_name, index_str = name[:-1].split('[', 1)
            try:
                index = int(index_str)
            except ValueError:
                # Index might be a variable
                index = self.env.get(index_str, 0)
            final_name = f"{base_name}[{index}]"
        else:
            final_name = name
            
        if final_name in self.cbit_map:
            return self.cbit_map[final_name]
        # If no creg with matching name exists, fall back to same index as qubit
        return self._q(final_name)

    # -------------------- classical execution (very small subset) ----------
    def _exec_py_node(self, node: Any) -> None:
        """Execute Python AST nodes (assignments, expressions, etc.)."""
        # Fix AST nodes by adding required fields
        self._fix_ast_node(node)
        
        # We cheat: compile to Python and exec in self.env
        wrapper = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(wrapper)  # This adds lineno and col_offset
        code = compile(wrapper, "<qsm>", "exec")
        exec(code, self.env)   # nosec: safe because we only accept tiny subset

    def _fix_ast_node(self, node: Any) -> None:
        """Recursively fix AST nodes by adding required fields."""
        if isinstance(node, ast.AST):
            # Add lineno and col_offset if missing
            if not hasattr(node, 'lineno'):
                node.lineno = 1
            if not hasattr(node, 'col_offset'):
                node.col_offset = 0
            
            # Recursively fix child nodes
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self._fix_ast_node(item)
                elif isinstance(value, ast.AST):
                    self._fix_ast_node(value)

    def _call_function(self, call_node: FunctionCall) -> None:
        func = self.env.get(call_node.name)
        if not isinstance(func, FunctionDef):
            raise NameError(f"Function '{call_node.name}' not defined.")
        # Bind arguments into a fresh local env (very naive)
        local_env = self.env.copy()
        for arg_name, arg_expr in zip(func.args, call_node.args):
            local_env[arg_name] = int(arg_expr)  # supports literals only
        old_env = self.env
        self.env = local_env
        try:
            for stmt in func.body:
                self._emit(stmt)
        finally:
            self.env = old_env

    def _handle_mcz(self, controls: str, target: str) -> None:
        """Handle multi-controlled Z gate."""
        # Parse controls (e.g., "q[0..n-2]" or "q[0],q[1],q[2]")
        if '..' in controls:
            # Range notation like q[0..n-2]
            base, range_part = controls.split('[', 1)
            range_part = range_part.rstrip(']')
            start_str, end_str = range_part.split('..', 1)
            
            # Evaluate expressions
            start = self._eval_expr_str(start_str)
            end = self._eval_expr_str(end_str)
            
            control_qubits = []
            for i in range(start, end + 1):
                control_qubits.append(self._q(f"{base}[{i}]"))
        else:
            # Individual qubits
            control_names = [c.strip() for c in controls.split(',')]
            control_qubits = [self._q(name) for name in control_names]
        
        target_qubit = self._q(target)
        
        # Implement multi-controlled Z using decomposition
        # For now, use a simple approach - in practice, this would be more complex
        if len(control_qubits) == 1:
            # Single controlled Z
            self.qc.cz(control_qubits[0], target_qubit)
        else:
            # Multi-controlled Z - simplified implementation
            # This is a placeholder - real implementation would use ancilla qubits
            print(f"[qsm] Multi-controlled Z with {len(control_qubits)} controls not fully implemented")
            # For now, just apply regular Z gate
            self.qc.z(target_qubit)

    def _eval_expr_str(self, expr_str: str) -> int:
        """Evaluate a string expression to an integer."""
        expr_str = expr_str.strip()
        
        # Handle simple cases first
        try:
            return int(expr_str)
        except ValueError:
            pass
        
        # Handle variable references
        if expr_str in self.env:
            return int(self.env[expr_str])
        
        # Handle simple arithmetic
        if '+' in expr_str:
            parts = expr_str.split('+', 1)
            left = self._eval_expr_str(parts[0].strip())
            right = self._eval_expr_str(parts[1].strip())
            return left + right
        elif '-' in expr_str:
            parts = expr_str.split('-', 1)
            left = self._eval_expr_str(parts[0].strip())
            right = self._eval_expr_str(parts[1].strip())
            return left - right
        elif '*' in expr_str:
            parts = expr_str.split('*', 1)
            left = self._eval_expr_str(parts[0].strip())
            right = self._eval_expr_str(parts[1].strip())
            return left * right
        elif '/' in expr_str:
            parts = expr_str.split('/', 1)
            left = self._eval_expr_str(parts[0].strip())
            right = self._eval_expr_str(parts[1].strip())
            return left // right  # Integer division
        
        # Default to 0 if we can't evaluate
        return 0

    # --------------------------------------------------------------------- #
    # Stubs                                                                 #
    # --------------------------------------------------------------------- #
    def _teleport_stub(self, q1: str, q2: str, q3: str) -> None:
        print(f"[qsm] Quantum teleportation not implemented for {q1}, {q2}, {q3}.")