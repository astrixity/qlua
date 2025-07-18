from setuptools import setup, find_packages

setup(
    name="qsm",
    version="0.1.0",
    description="qsm: Quantum Lua-like Language using Qiskit",
    author="Astrixity",
    packages=find_packages(),
    install_requires=["qiskit"],
    entry_points={
        "console_scripts": [
            "qsm = qsm.__main__:main"
        ]
    },
    python_requires=">=3.8",
)
