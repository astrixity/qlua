# QLua: Quantum Lua Programming Language

This project is a Python package for developing a new quantum programming language using Qiskit.

## Features
- Modular design for extensibility
- Qiskit integration for quantum circuit and algorithm development

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/astrixity/qlua.git
   cd qlua
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. (Optional) Install QLua as a command-line tool:
   ```sh
   pip install .
   ```

## Usage

Run a QLua script:
```sh
qlua <script.qlua>
```

Example:
```sh
qlua hello.qlua
```

## Example QLua Script and Output

Suppose you have a file `main.qlua` with the following contents:

```qlua
qbit a, b
hadamard a
cx a, b
measure a, b
```

Run it with:
```sh
qlua main.qlua
```

Example output:
```
[QLua] Quantum Circuit:
     ┌───┐     ┌─┐   
q_0: ┤ H ├──■──┤M├───
     └───┘┌─┴─┐└╥┘┌─┐
q_1: ─────┤ X ├─╫─┤M├
          └───┘ ║ └╥┘
c: 2/═══════════╩══╩═
           0  1
```

---

For more information, see the code examples and documentation in this repository.
