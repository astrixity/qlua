# qlua/compiler.py
"""
Quantum Lua Compiler

This module provides a compiler for a Lua-like quantum programming language with math and quantum support.
"""
from qiskit import QuantumCircuit
from qlua.parser import QLuaParser
import ast

class QBitDecl(ast.AST):          # qbit a, b, c
    _fields = ("targets",)

class Hadamard(ast.AST):         # hadamard a
    _fields = ("target",)

class CX(ast.AST):               # cx a, b
    _fields = ("ctrl", "targ")

class Measure(ast.AST):          # measure a
    _fields = ("target",)

class Teleport(ast.AST):         # teleport a, b, c
    _fields = ("q1", "q2", "q3")

class QLuaCompiler:
    def __init__(self):
        self.qubit_map = {}      # name -> qubit index
        self.cbit_map  = {}      # name -> classical bit index
        self.circuit   = QuantumCircuit(0, 0)
        self.parser    = QLuaParser()
        self.env       = {}

    def _get_or_add_qubit(self, name: str) -> int:
        # Qubits are registered via QBitDecl, so just return index if present
        if name not in self.qubit_map:
            raise NameError(f"Qubit '{name}' not declared. Use 'qbit {name}' first.")
        return self.qubit_map[name]

    def _get_or_add_cbit(self, name: str) -> int:
        # Same idea, but mirrored for classical bits
        if name not in self.cbit_map:
            idx = len(self.cbit_map)
            self.cbit_map[name] = idx
        return self.cbit_map[name]

    def compile(self, code: str):
        ast_nodes = self.parser.parse(code)
        for node in ast_nodes:
            self._eval_node(node)

    def _eval_node(self, node):
        # Quantum instructions
        from qlua.parser import QBitDecl, Hadamard, CX, Measure, Teleport, XGate
        if isinstance(node, QBitDecl):
            # Register qubits and classical bits
            for name in node.targets:
                if name not in self.qubit_map:
                    idx = len(self.qubit_map)
                    self.qubit_map[name] = idx
            # Re-create the circuit with new size if needed
            n = len(self.qubit_map)
            if n > self.circuit.num_qubits:
                old = self.circuit
                self.circuit = QuantumCircuit(n, n)
                self.circuit.data = old.data
        elif isinstance(node, Hadamard):
            qidx = self._get_or_add_qubit(node.target)
            self.circuit.h(qidx)
        elif isinstance(node, XGate):
            qidx = self._get_or_add_qubit(node.target)
            self.circuit.x(qidx)
        elif isinstance(node, CX):
            ctrl = self._get_or_add_qubit(node.ctrl)
            targ = self._get_or_add_qubit(node.targ)
            self.circuit.cx(ctrl, targ)
        elif isinstance(node, Measure):
            qidx = self._get_or_add_qubit(node.target)
            self.circuit.measure(qidx, qidx)
        elif isinstance(node, Teleport):
            self.quantum_teleport(node.q1, node.q2, node.q3)
        # Classical
        elif isinstance(node, ast.Expr):
            return self._eval_expr(node.value)
        elif isinstance(node, ast.Assign):
            target = node.targets[0].id
            value = self._eval_expr(node.value)
            self.env[target] = value
        elif isinstance(node, ast.If):
            test = self._eval_expr(node.test)
            if test:
                for stmt in node.body:
                    self._eval_node(stmt)
            else:
                for stmt in node.orelse:
                    self._eval_node(stmt)
        elif isinstance(node, ast.While):
            while self._eval_expr(node.test):
                for stmt in node.body:
                    self._eval_node(stmt)
        elif isinstance(node, ast.For):
            iter_obj = self._eval_expr(node.iter)
            target = node.target.id
            for val in iter_obj:
                self.env[target] = val
                for stmt in node.body:
                    self._eval_node(stmt)
        else:
            raise SyntaxError(f"Unsupported statement: {ast.dump(node)}")

    def _eval_expr(self, expr):
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name) and expr.func.id == "print":
                args = [self._eval_expr(a) for a in expr.args]
                self._emit_print(*args)
            elif isinstance(expr.func, ast.Name) and expr.func.id == "range":
                args = [self._eval_expr(a) for a in expr.args]
                return range(*args)
            else:
                raise SyntaxError(f"Unsupported function: {ast.dump(expr)}")
        elif isinstance(expr, ast.BinOp):
            left = self._eval_expr(expr.left)
            right = self._eval_expr(expr.right)
            if isinstance(expr.op, ast.Add):
                return left + right
            elif isinstance(expr.op, ast.Sub):
                return left - right
            elif isinstance(expr.op, ast.Mult):
                return left * right
            elif isinstance(expr.op, ast.Div):
                return left / right
            else:
                raise SyntaxError(f"Unsupported operator: {ast.dump(expr.op)}")
        elif isinstance(expr, ast.Compare):
            left = self._eval_expr(expr.left)
            results = []
            for op, comparator in zip(expr.ops, expr.comparators):
                right = self._eval_expr(comparator)
                if isinstance(op, ast.Eq):
                    results.append(left == right)
                elif isinstance(op, ast.NotEq):
                    results.append(left != right)
                elif isinstance(op, ast.Lt):
                    results.append(left < right)
                elif isinstance(op, ast.LtE):
                    results.append(left <= right)
                elif isinstance(op, ast.Gt):
                    results.append(left > right)
                elif isinstance(op, ast.GtE):
                    results.append(left >= right)
                else:
                    raise SyntaxError(f"Unsupported comparison operator: {ast.dump(op)}")
                left = right
            return all(results)
        elif isinstance(expr, ast.Constant):
            return expr.value
        elif isinstance(expr, ast.Name):
            if expr.id in self.env:
                return self.env[expr.id]
            elif expr.id == "range":
                return range
            else:
                raise NameError(f"Undefined variable: {expr.id}")
        else:
            raise SyntaxError(f"Unsupported expression: {ast.dump(expr)}")

    def _emit_print(self, *args):
        self.circuit.h(0)
        self.circuit.measure(0, 0)
        print("[Quantum print]", *args)

    def quantum_teleport(self, q1, q2, q3):
        # Future implementation of teleportation protocol
        print(f"[QLua] Quantum teleportation not yet implemented for {q1}, {q2}, {q3}.")

    def get_circuit(self):
        return self.circuit

    def export_qasm(self):
        """Export the current circuit to OpenQASM string."""
        return self.circuit.qasm()
