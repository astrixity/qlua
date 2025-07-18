# qlua/parser.py
"""
A parser for QLua language supporting Lua-like syntax and quantum extensions.
"""

import ast

# QLua quantum instruction AST nodes
class QBitDecl(ast.AST):         # qbit a, b, c
    _fields = ("targets",)

class Hadamard(ast.AST):         # hadamard a
    _fields = ("target",)

class CX(ast.AST):               # cx a, b
    _fields = ("ctrl", "targ")

class Measure(ast.AST):          # measure a
    _fields = ("target",)

class Teleport(ast.AST):         # teleport a, b, c
    _fields = ("q1", "q2", "q3")

class XGate(ast.AST):            # x a
    _fields = ("target",)

class QLuaParser:
    def parse(self, code: str):
        """Parse Lua-like QLua code into Python AST nodes, including quantum instructions."""
        lines = code.strip().splitlines()
        py_lines = []
        indent = 0
        block_stack = []
        ast_nodes = []
        for line in lines:
            # Remove comments (everything after --)
            code_part = line.split('--', 1)[0]
            stripped = code_part.strip()
            # Skip empty or comment-only lines
            if not stripped:
                continue
            # Quantum instructions
            if stripped.startswith('qbit '):
                # qbit a, b, c
                names = [n.strip() for n in stripped[5:].split(',')]
                ast_nodes.append(QBitDecl(targets=names))
            elif stripped.startswith('hadamard '):
                # hadamard a, b, ...
                names = [n.strip() for n in stripped[9:].split(',')]
                for name in names:
                    if name:
                        ast_nodes.append(Hadamard(target=name))
            elif stripped.startswith('x '):
                # x a, b, ...
                names = [n.strip() for n in stripped[2:].split(',')]
                for name in names:
                    if name:
                        ast_nodes.append(XGate(target=name))
            elif stripped.startswith('cx '):
                # cx a, b
                args = [n.strip() for n in stripped[3:].split(',')]
                if len(args) == 2:
                    ast_nodes.append(CX(ctrl=args[0], targ=args[1]))
            elif stripped.startswith('measure '):
                # measure a, b, ...
                names = [n.strip() for n in stripped[8:].split(',')]
                for name in names:
                    if name:
                        ast_nodes.append(Measure(target=name))
            elif stripped.startswith('teleport '):
                # teleport a, b, c
                args = [n.strip() for n in stripped[9:].split(',')]
                if len(args) == 3:
                    ast_nodes.append(Teleport(q1=args[0], q2=args[1], q3=args[2]))
            # Classical control flow
            elif stripped.startswith('if ') and stripped.endswith(' then'):
                py_lines.append('    ' * indent + 'if ' + stripped[3:-5] + ':')
                indent += 1
                block_stack.append('if')
            elif stripped == 'else':
                indent -= 1
                py_lines.append('    ' * indent + 'else:')
                indent += 1
            elif stripped.startswith('while ') and stripped.endswith(' do'):
                py_lines.append('    ' * indent + 'while ' + stripped[6:-3] + ':')
                indent += 1
                block_stack.append('while')
            elif stripped.startswith('for ') and ' = ' in stripped and stripped.endswith(' do'):
                for_head = stripped[4:-3]
                var, rng = for_head.split('=', 1)
                var = var.strip()
                rng = rng.strip()
                if ',' in rng:
                    parts = [p.strip() for p in rng.split(',')]
                    if len(parts) == 2:
                        start, stop = parts
                        py_lines.append('    ' * indent + f'for {var} in range({start}, {int(stop)+1}):')
                    elif len(parts) == 3:
                        start, stop, step = parts
                        py_lines.append('    ' * indent + f'for {var} in range({start}, {int(stop)+1}, {step}):')
                indent += 1
                block_stack.append('for')
            elif stripped == 'end':
                indent -= 1
                if block_stack:
                    block_stack.pop()
            elif stripped:
                py_lines.append('    ' * indent + stripped)
        # Parse classical code as Python AST
        if py_lines:
            py_code = '\n'.join(py_lines)
            for node in ast.parse(py_code, mode='exec').body:
                ast_nodes.append(node)
        return ast_nodes
