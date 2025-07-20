# qsm/parser.py
"""
A parser for qsm language supporting Lua-like syntax and quantum extensions.
"""

import ast

# qsm quantum instruction AST nodes
class QBitDecl(ast.AST):         # qbit a, b, c
    _fields = ("targets",)

class Hadamard(ast.AST):         # hadamard a
    _fields = ("target",)

class CX(ast.AST):               # cx a, b
    _fields = ("ctrl", "targ")

class Measure(ast.AST):          # measure a -> b
    _fields = ("target", "classical_target")

class Teleport(ast.AST):         # teleport a, b, c
    _fields = ("q1", "q2", "q3")

class XGate(ast.AST):            # x a
    _fields = ("target",)

class ZGate(ast.AST):            # z a
    _fields = ("target",)

class YGate(ast.AST):            # y a
    _fields = ("target",)

class SGate(ast.AST):            # s a
    _fields = ("target",)

class TGate(ast.AST):            # t a
    _fields = ("target",)

class SwapGate(ast.AST):         # swap a, b
    _fields = ("q1", "q2")

class MCZGate(ast.AST):          # mcz controls, target
    _fields = ("controls", "target")

class FunctionDef(ast.AST):
    _fields = ("name", "args", "body")

class FunctionCall(ast.AST):
    _fields = ("name", "args")

class CRegDecl(ast.AST):         # creg answer[4]
    _fields = ("targets",)

class QSMFor(ast.AST):           # for loop with quantum operations
    _fields = ("target", "iter", "body")

class QSMWhile(ast.AST):         # while loop with quantum operations
    _fields = ("test", "body")

class QSMIf(ast.AST):            # if statement with quantum operations
    _fields = ("test", "body", "orelse")

class qsmParser:
    def parse(self, code: str):
        """Parse Lua-like qsm code into mixed AST nodes, with multiline comment support."""
        import re
        # Remove multiline comments: --[[ ... ]]
        code = re.sub(r'--\[\[.*?\]\]--', '', code, flags=re.DOTALL)
        lines = code.strip().splitlines()
        ast_nodes = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            # Remove single-line comments (everything after --)
            code_part = line.split('--', 1)[0]
            stripped = code_part.strip()
            
            # Skip empty or comment-only lines
            if not stripped:
                i += 1
                continue
                
            # Parse the line and add to AST
            node = self._parse_line(stripped, lines, i)
            if node is not None:
                if isinstance(node, tuple):
                    # Node with new position
                    ast_nodes.append(node[0])
                    i = node[1]
                else:
                    ast_nodes.append(node)
                    i += 1
            else:
                i += 1
        return ast_nodes
    
    def _parse_line(self, stripped: str, lines: list, current_pos: int):
        """Parse a single line and return the corresponding AST node."""
        
        # Function definition
        if stripped.startswith('function '):
            return self._parse_function(stripped, lines, current_pos)
            
        # Function call
        if '(' in stripped and stripped.endswith(')') and not stripped.startswith('if ') and not stripped.startswith('print '):
            fname, argstr = stripped.split('(', 1)
            fname = fname.strip()
            argstr = argstr.rstrip(')').strip()
            args = [a.strip() for a in argstr.split(',')] if argstr else []
            return FunctionCall(name=fname, args=args)
            
        # Control flow structures
        if stripped.startswith('for ') and ' = ' in stripped and stripped.endswith(' do'):
            return self._parse_for_loop(stripped, lines, current_pos)
            
        if stripped.startswith('while ') and stripped.endswith(' do'):
            return self._parse_while_loop(stripped, lines, current_pos)
            
        if stripped.startswith('if ') and stripped.endswith(' then'):
            return self._parse_if_statement(stripped, lines, current_pos)
            
        # Quantum register declarations
        if stripped.startswith('qbit '):
            names = [n.strip() for n in stripped[5:].split(',')]
            expanded = []
            for name in names:
                if '[' in name and name.endswith(']'):
                    base, count = name[:-1].split('[')
                    count = int(count)
                    expanded.extend([f"{base}[{i}]" for i in range(count)])
                else:
                    expanded.append(name)
            return QBitDecl(targets=expanded)
            
        if stripped.startswith('creg '):
            names = [n.strip() for n in stripped[5:].split(',')]
            expanded = []
            for name in names:
                if '[' in name and name.endswith(']'):
                    base, count = name[:-1].split('[')
                    count = int(count)
                    expanded.extend([f"{base}[{i}]" for i in range(count)])
                else:
                    expanded.append(name)
            return CRegDecl(targets=expanded)
            
        # Quantum gates
        if stripped.startswith('hadamard ') or stripped.startswith('h '):
            prefix_len = 9 if stripped.startswith('hadamard ') else 2
            names = [n.strip() for n in stripped[prefix_len:].split(',')]
            # Return multiple gates for multiple targets
            gates = []
            for name in names:
                if name:
                    gates.append(Hadamard(target=name))
            return gates[0] if len(gates) == 1 else gates
            
        if stripped.startswith('x '):
            names = [n.strip() for n in stripped[2:].split(',')]
            gates = []
            for name in names:
                if name:
                    gates.append(XGate(target=name))
            return gates[0] if len(gates) == 1 else gates
            
        if stripped.startswith('z '):
            names = [n.strip() for n in stripped[2:].split(',')]
            gates = []
            for name in names:
                if name:
                    gates.append(ZGate(target=name))
            return gates[0] if len(gates) == 1 else gates
            
        if stripped.startswith('y '):
            names = [n.strip() for n in stripped[2:].split(',')]
            gates = []
            for name in names:
                if name:
                    gates.append(YGate(target=name))
            return gates[0] if len(gates) == 1 else gates
            
        if stripped.startswith('s '):
            names = [n.strip() for n in stripped[2:].split(',')]
            gates = []
            for name in names:
                if name:
                    gates.append(SGate(target=name))
            return gates[0] if len(gates) == 1 else gates
            
        if stripped.startswith('t '):
            names = [n.strip() for n in stripped[2:].split(',')]
            gates = []
            for name in names:
                if name:
                    gates.append(TGate(target=name))
            return gates[0] if len(gates) == 1 else gates
            
        if stripped.startswith('swap '):
            args = [n.strip() for n in stripped[5:].split(',')]
            if len(args) == 2:
                return SwapGate(q1=args[0], q2=args[1])
                
        if stripped.startswith('cx '):
            args = [n.strip() for n in stripped[3:].split(',')]
            if len(args) == 2:
                return CX(ctrl=args[0], targ=args[1])
                
        if stripped.startswith('measure '):
            measure_part = stripped[8:]
            if '->' in measure_part:
                qubit_part, classical_part = measure_part.split('->', 1)
                qubit_target = qubit_part.strip()
                classical_target = classical_part.strip()
                return Measure(target=qubit_target, classical_target=classical_target)
            else:
                names = [n.strip() for n in measure_part.split(',')]
                gates = []
                for name in names:
                    if name:
                        gates.append(Measure(target=name, classical_target=None))
                return gates[0] if len(gates) == 1 else gates
                
        if stripped.startswith('mcz '):
            # Multi-controlled Z gate: mcz q[0..n-2], q[n-1]
            args = [n.strip() for n in stripped[4:].split(',')]
            if len(args) == 2:
                controls = args[0]
                target = args[1]
                return MCZGate(controls=controls, target=target)
                
        if stripped.startswith('teleport '):
            args = [n.strip() for n in stripped[9:].split(',')]
            if len(args) == 3:
                return Teleport(q1=args[0], q2=args[1], q3=args[2])
                
        if stripped.startswith('print '):
            args_str = stripped[6:].strip()
            arg_strings = [a.strip() for a in args_str.split(',')]
            args = []
            for tok in arg_strings:
                # String literal with quotes?
                if (tok.startswith('"') and tok.endswith('"')) or \
                   (tok.startswith("'") and tok.endswith("'")):
                    args.append(ast.Constant(value=tok[1:-1]))
                    continue

                # Register slice, e.g. result[1] or answer[i]
                if '[' in tok and tok.endswith(']'):
                    base, idx = tok[:-1].split('[', 1)
                    # Try to parse idx as int, else as variable
                    try:
                        idx_ast = ast.Constant(value=int(idx))
                    except ValueError:
                        idx_ast = ast.Name(id=idx, ctx=ast.Load())
                    args.append(
                        ast.Subscript(
                            value=ast.Name(id=base, ctx=ast.Load()),
                            slice=idx_ast,
                            ctx=ast.Load()
                        )
                    )
                    continue

                # Bare register name
                if tok.isidentifier():
                    args.append(ast.Name(id=tok, ctx=ast.Load()))
                    continue

                # Numeric literal
                try:
                    args.append(ast.Constant(value=int(tok)))
                except ValueError:
                    args.append(ast.Constant(value=tok))

            return ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='print', ctx=ast.Load()),
                    args=args,
                    keywords=[]
                )
            )

            
        # Skip 'end' statements - they're handled by the control flow parsers
        if stripped == 'end':
            return None
            
        # Other statements - try to parse as Python
        try:
            node = ast.parse(stripped, mode='eval').body
            # If it's a bare identifier (ast.Name), skip it
            if isinstance(node, ast.Name):
                return None
            return node
        except:
            # If it fails, create a simple assignment or expression
            if '=' in stripped:
                var, val = stripped.split('=', 1)
                return ast.Assign(
                    targets=[ast.Name(id=var.strip(), ctx=ast.Store())],
                    value=ast.Constant(value=val.strip())
                )
        return None
    
    def _parse_expr(self, expr_str: str) -> ast.expr:
        """Parse an expression string into an AST node."""
        expr_str = expr_str.strip()
        
        # Try to parse as integer first
        try:
            return ast.Constant(value=int(expr_str))
        except ValueError:
            pass
        
        # Handle simple binary operations
        if '+' in expr_str:
            parts = expr_str.split('+', 1)
            left = self._parse_expr(parts[0].strip())
            right = self._parse_expr(parts[1].strip())
            return ast.BinOp(left=left, op=ast.Add(), right=right)
        elif '-' in expr_str:
            parts = expr_str.split('-', 1)
            left = self._parse_expr(parts[0].strip())
            right = self._parse_expr(parts[1].strip())
            return ast.BinOp(left=left, op=ast.Sub(), right=right)
        elif '*' in expr_str:
            parts = expr_str.split('*', 1)
            left = self._parse_expr(parts[0].strip())
            right = self._parse_expr(parts[1].strip())
            return ast.BinOp(left=left, op=ast.Mult(), right=right)
        elif '/' in expr_str:
            parts = expr_str.split('/', 1)
            left = self._parse_expr(parts[0].strip())
            right = self._parse_expr(parts[1].strip())
            return ast.BinOp(left=left, op=ast.Div(), right=right)
        
        # Handle variable names
        return ast.Name(id=expr_str, ctx=ast.Load())
    
    def _parse_function(self, stripped: str, lines: list, current_pos: int):
        """Parse a function definition."""
        header = stripped[len('function '):]
        name, argstr = header.split('(', 1)
        name = name.strip()
        argstr = argstr.rstrip(')').strip()
        args = [a.strip() for a in argstr.split(',')] if argstr else []
        
        # Parse function body until 'end'
        body_nodes = []
        i = current_pos + 1
        while i < len(lines):
            body_line = lines[i]
            if body_line.strip() == 'end':
                break
            
            code_part = body_line.split('--', 1)[0]
            stripped_body = code_part.strip()
            if stripped_body:
                node = self._parse_line(stripped_body, lines, i)
                if node is not None:
                    if isinstance(node, tuple):
                        body_nodes.append(node[0])
                        i = node[1]
                    else:
                        body_nodes.append(node)
            i += 1
            
        return (FunctionDef(name=name, args=args, body=body_nodes), i)
    
    def _parse_for_loop(self, stripped: str, lines: list, current_pos: int):
        """Parse a for loop."""
        for_head = stripped[4:-3]  # Remove 'for ' and ' do'
        var, rng = for_head.split('=', 1)
        var = var.strip()
        rng = rng.strip()
        
        # Create range AST
        if ',' in rng:
            parts = [p.strip() for p in rng.split(',')]
            if len(parts) == 2:
                start, stop = parts
                start_node = self._parse_expr(start)
                stop_node = self._parse_expr(stop)
                # Add 1 to stop for inclusive range
                stop_plus_one = ast.BinOp(
                    left=stop_node,
                    op=ast.Add(),
                    right=ast.Constant(value=1)
                )
                iter_node = ast.Call(
                    func=ast.Name(id='range', ctx=ast.Load()),
                    args=[start_node, stop_plus_one],
                    keywords=[]
                )
            elif len(parts) == 3:
                start, stop, step = parts
                start_node = self._parse_expr(start)
                stop_node = self._parse_expr(stop)
                step_node = self._parse_expr(step)
                # Add 1 to stop for inclusive range
                stop_plus_one = ast.BinOp(
                    left=stop_node,
                    op=ast.Add(),
                    right=ast.Constant(value=1)
                )
                iter_node = ast.Call(
                    func=ast.Name(id='range', ctx=ast.Load()),
                    args=[start_node, stop_plus_one, step_node],
                    keywords=[]
                )
        else:
            rng_node = self._parse_expr(rng)
            iter_node = ast.Call(
                func=ast.Name(id='range', ctx=ast.Load()),
                args=[rng_node],
                keywords=[]
            )
            
        # Parse loop body until 'end'
        body_nodes = []
        i = current_pos + 1
        while i < len(lines):
            body_line = lines[i]
            if body_line.strip() == 'end':
                break
                
            code_part = body_line.split('--', 1)[0]
            stripped_body = code_part.strip()
            if stripped_body:
                node = self._parse_line(stripped_body, lines, i)
                if node is not None:
                    if isinstance(node, tuple):
                        body_nodes.append(node[0])
                        i = node[1]
                    else:
                        body_nodes.append(node)
            i += 1
            
        # Create Python for loop AST
        for_node = ast.For(
            target=ast.Name(id=var, ctx=ast.Store()),
            iter=iter_node,
            body=body_nodes,
            orelse=[]
        )
        
        return (for_node, i)
    
    def _parse_while_loop(self, stripped: str, lines: list, current_pos: int):
        """Parse a while loop."""
        condition = stripped[6:-3]  # Remove 'while ' and ' do'
        
        # Parse condition
        test_node = ast.parse(condition, mode='eval').body
        
        # Parse loop body until 'end'
        body_nodes = []
        i = current_pos + 1
        while i < len(lines):
            body_line = lines[i]
            if body_line.strip() == 'end':
                break
                
            code_part = body_line.split('--', 1)[0]
            stripped_body = code_part.strip()
            if stripped_body:
                node = self._parse_line(stripped_body, lines, i)
                if node is not None:
                    if isinstance(node, tuple):
                        body_nodes.append(node[0])
                        i = node[1]
                    else:
                        body_nodes.append(node)
            i += 1
            
        while_node = ast.While(
            test=test_node,
            body=body_nodes,
            orelse=[]
        )
        
        return (while_node, i)
    
    def _parse_if_statement(self, stripped: str, lines: list, current_pos: int):
        """Parse an if statement."""
        condition = stripped[3:-5]  # Remove 'if ' and ' then'
        
        # Parse condition
        test_node = ast.parse(condition, mode='eval').body
        
        # Parse if body until 'else' or 'end'
        body_nodes = []
        orelse_nodes = []
        i = current_pos + 1
        in_else = False
        
        while i < len(lines):
            body_line = lines[i]
            stripped_body = body_line.strip()
            
            if stripped_body == 'end':
                break
            elif stripped_body == 'else':
                in_else = True
                i += 1
                continue
                
            code_part = body_line.split('--', 1)[0]
            stripped_body = code_part.strip()
            if stripped_body:
                node = self._parse_line(stripped_body, lines, i)
                if node is not None:
                    if isinstance(node, tuple):
                        target_list = orelse_nodes if in_else else body_nodes
                        target_list.append(node[0])
                        i = node[1]
                    else:
                        target_list = orelse_nodes if in_else else body_nodes
                        target_list.append(node)
            i += 1
            
        if_node = ast.If(
            test=test_node,
            body=body_nodes,
            orelse=orelse_nodes
        )
        
        return (if_node, i)