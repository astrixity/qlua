
![QSM Logo](qsm_logo.png)

# QSM: Quantum Lua Programming Language

QSM is a **Lua-inspired** quantum programming language that transpiles to Qiskit circuits.\
Write small, readable scripts and run them locally with Aer or on IBM Quantum backends.

---

## ✨ What’s new in v0.3

- One-liner install: `pip install qsm`
- New CLI flags: `--draw`, `--shots N`, `--debug`
- Built-in pretty printer for measurement results
- Works as a **Python library** as well:

  ```python
  from qsm import QsmCompiler
  qc = QsmCompiler().compile("qbit q; hadamard q; measure q")
  ```

## ⚡ Quick-start

### Install

```bash
pip install qsm
```

### Create hello.qsm

```qsm
qbit q
creg c
hadamard q
measure q -> c
print "Measured:", c
```

### Run

```bash
qsm hello.qsm
```

You should see the ASCII circuit and a random 0 or 1.

## 📦 Installation Matrix

| Method | When to use | Command |
| --- | --- | --- |
| PyPI wheel | Just want to run scripts | `pip install qsm` |
| Source | Hacking on compiler/parser | `git clone … && pip install -e .` |
| Dev extras | Need Aer & docs | `pip install -e '.[dev]'` |

## Requirements

- Python 3.9+
- Qiskit ≥ 0.45
- qiskit-aer (auto-installed via pip)

## 🧪 Mini Language Tour

| Quantum | Classical |
| --- | --- |
| `qbit q[8]` | `for i = 0, 7 do … end` |
| `h q[0]` | `if x == 3 then … end` |
| `cx q[0], q[1]` | `print "result:", answer` |
| `measure q[0] -> answer[0]` | `function foo(n) … end` |
| `mcz q[0..2], q[3]` | `while k < n do … end` |

Full grammar and semantics → QSM_LANGUAGE.md.

## 🎯 Grover Example (from repo)

```qsm
qbit q[4]
creg answer[4]

-- uniform superposition
for i = 0, 3 do h q[i] end

-- oracle marks |1010>
function oracle()
    z q[1]; z q[3]   -- 1010 has bits 1 & 3 set to 1
end

-- diffusion operator
function diffusion(n)
    for i = 0, n-1 do h q[i]; x q[i] end
    h q[n-1]; mcz q[0..n-2], q[n-1]; h q[n-1]
    for i = 0, n-1 do x q[i]; h q[i] end
end

oracle(); diffusion(4); oracle(); diffusion(4)

measure q -> answer
print "Grover result:", answer
```

## 🛠️ CLI Reference

```
usage: qsm [-h] [--draw {text,mpl,latex}] [--shots N] [--debug] file

positional arguments:
  file                  QSM source file

optional arguments:
  -h, --help            show this help message and exit
  --draw {text,mpl,latex}
                        Render circuit diagram (default: text)
  --shots N             Number of simulation shots (default: 1)
  --debug               Emit verbose compiler logs
```

## 🐞 Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: qiskit_aer` | `pip install qiskit-aer` |
| `AttributeError: mcz (old Qiskit)` | `Upgrade: pip install -U qiskit` |

## 🧩 Embedding QSM in Python

```python
from qsm import QsmCompiler

src = '''
qbit q[2]
cx q[0], q[1]
measure q -> c
'''
compiler = QsmCompiler(debug=True)
qc = compiler.compile(src)
counts = compiler.run_simulation()  # returns dict
```

## Contributing

- Fork & clone
- `pip install -e '.[dev]'`
- `pytest` (unit tests)
- PRs welcome! Please run `black qsm/` and `isort qsm/`