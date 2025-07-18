# QSM Language Reference

## Overview
QSM is a Lua-inspired quantum programming language built on top of Qiskit. It provides a simple syntax for quantum circuit construction, quantum/classical control flow, and extensibility for new quantum operations.

## Quantum Instructions
- `qbit` — Declare qubits (e.g., `qbit q[4]`)
- `creg` — Declare classical bits (e.g., `creg answer[4]`)
- `hadamard` or `h` — Hadamard gate
- `x`, `y`, `z` — Pauli gates
- `s`, `t` — S and T gates
- `cx` — CNOT (controlled-X) gate
- `swap` — SWAP gate
- `mcz` — Multi-controlled Z gate (syntax: `mcz q[0..n-2], q[n-1]`)
- `measure` — Measure qubits (e.g., `measure q[0] -> answer[0]`)
- `teleport` — Quantum teleportation (stub)

## Classical Control Flow
- `if ... then ... else ... end` — Conditional
- `while ... do ... end` — Loop
- `for i = start, stop do ... end` — Loop (inclusive range)
- `function name(args) ... end` — Function definition
- `name(args)` — Function call
- `print ...` — Output
- Assignments: `a = 5`, `b = a + 1`

## Comments
- Single-line comments start with `--`

## Example: Grover's Algorithm
```qsm
qbit q[4]
creg answer[4]

-- Create uniform superposition
for i = 0, 3 do
    hadamard q[i]
end

-- Oracle marks |1010>
function oracle()
    for i = 0, 3 do
        if i % 2 == 0 then
            z q[i]
        end
    end
end

function diffusion(n)
    for i = 0, n-1 do
        hadamard q[i]
        x q[i]
    end
    h q[n-1]
    mcz q[0..n-2], q[n-1]
    h q[n-1]
    for i = 0, n-1 do
        x q[i]
        hadamard q[i]
    end
end

diffusion(4)
oracle()
diffusion(4)

measure q[0] -> answer[0]
measure q[1] -> answer[1]
measure q[2] -> answer[2]
measure q[3] -> answer[3]
print "Grover output: ", answer
```

## Extending QSM
- Add new gates or syntax by editing `qsm/parser.py` and `qsm/compiler.py`.
- See code comments for extension points.

## See Also
- [Qiskit Documentation](https://qiskit.org/documentation/)
- Example scripts in the repository.
