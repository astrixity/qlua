# qsm: Quantum Lua Programming Language

This project is a Python package for developing a new quantum programming language using Qiskit.

## Features
- Lua-inspired syntax for easy quantum programming
- Modular design for extensibility
- Qiskit integration for quantum circuit and algorithm development
- Supports quantum instructions:
  - `qbit` — declare qubits
  - `hadamard` — Hadamard gate (H)
  - `x` — Pauli-X (NOT) gate
  - `z` — Pauli-Z gate
  - `y` — Pauli-Y gate
  - `s` — S gate
  - `t` — T gate
  - `cx` — CNOT (controlled-X) gate
  - `swap` — SWAP gate
  - `measure` — measure qubits
  - `teleport` — quantum teleportation (stub)
- Supports basic Lua-like control flow: `if`, `while`, `for`, assignments, and print
- Single-line comments with `--`

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

## Example qsm Scripts and Output

### Bell State (Entanglement)
Suppose you have a file `main.qsm` with the following contents:

```qsm
qbit a, b
hadamard a
cx a, b
measure a, b
```

Run it with:
```sh
qsm main.qsm
```

Example output:
```
[qsm] Quantum Circuit:
     ┌───┐     ┌─┐   
q_0: ┤ H ├──■──┤M├───
     └───┘┌─┴─┐└╥┘┌─┐
q_1: ─────┤ X ├─╫─┤M├
          └───┘ ║ └╥┘
c: 2/═══════════╩══╩═
           0  1
```

### Superposition and Pauli Gates

```qsm
qbit a, b
hadamard a, b
x a
z b
measure a, b
```

### Swap Gate Example

```qsm
qbit a, b
x a
swap a, b
measure a, b
```

### Using Classical Control Flow

```qsm
qbit a
for i = 1, 2 do
    hadamard a
end
measure a
```

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
