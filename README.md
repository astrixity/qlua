# qsm: Quantum Lua Programming Language

This project is a Python package for developing a new quantum programming language using Qiskit.


## Features
- Lua-inspired syntax for easy quantum programming
- Modular design for extensibility
- Qiskit integration for quantum circuit and algorithm development
- Quantum instructions: `qbit`, `creg`, `hadamard`/`h`, `x`, `y`, `z`, `s`, `t`, `cx`, `swap`, `mcz`, `measure`, `teleport`
- Classical control flow: `if`, `while`, `for`, assignments, and print
- User-defined functions and function calls
- Multi-controlled Z gate (Grover's diffusion)
- Single-line comments with `--`
- See `QSM_LANGUAGE.md` for full language reference

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/astrixity/qsm.git
   cd qsm
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. (Optional) Install qsm as a command-line tool:
   ```sh
   pip install .
   ```

## Usage

Run a qsm script:
```sh
qsm <script.qsm>
```

Example:
```sh
qsm hello.qsm
```


## Example: Grover's Algorithm with Functions and Loops

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

See `QSM_LANGUAGE.md` for a full language reference and more advanced examples.

---

## Notes for Users

- qsm is designed for quantum programming with a familiar, Lua-like syntax.
- Use quantum instructions (`qbit`, `hadamard`, `x`, `z`, `y`, `s`, `t`, `cx`, `swap`, `measure`, `teleport`) to build quantum circuits.
- Classical control flow (`if`, `while`, `for`) is supported for structuring your quantum code.
- Comments start with `--`.
- Not all Lua features are supported; focus is on quantum programming.
- See the `examples/` directory (if available) for more sample scripts.
- For advanced users: you can extend qsm by editing the parser and compiler in the `qsm/` directory.

For more information, see the code examples and documentation in this repository.
