from setuptools import setup, find_packages

setup(
    name="qlua",
    version="0.1.0",
    description="QLua: Quantum Lua-like Language using Qiskit",
    author="Astrixity",
    packages=find_packages(),
    install_requires=["qiskit"],
    entry_points={
        "console_scripts": [
            "qlua = qlua.__main__:main"
        ]
    },
    python_requires=">=3.8",
)
