#!/usr/bin/env python3
import importlib
import sys
from packaging.version import parse

# Lista de paquetes a verificar: (nombre legible, nombre de import, versión mínima)
packages = [
    ("numpy", "numpy", "1.21.0"),
    ("scipy", "scipy", "1.7.0"),
    ("pandas", "pandas", "1.3.0"),
    ("matplotlib", "matplotlib", "3.5.0"),
    ("seaborn", "seaborn", "0.11.0"),
    ("qiskit", "qiskit", "0.45.0"),
    ("qiskit_aer", "qiskit_aer", "0.12.0"),
    ("qiskit_ibm_runtime", "qiskit_ibm_runtime", "0.12.0"),
    ("qiskit_machine_learning", "qiskit_machine_learning", "0.6.0"),
    ("scikit_learn", "sklearn", "1.0.0"),
    ("networkx", "networkx", "2.6.0"),
    ("pillow", "PIL", "9.0.0"),
    ("plotly", "plotly", "5.0.0"),
    ("bokeh", "bokeh", "2.4.0"),
    ("altair", "altair", "4.2.0"),
    ("tqdm", "tqdm", "4.62.0"),
    ("pyyaml", "yaml", "6.0"),
    ("rich", "rich", "13.0.0"),
    ("numba", "numba", "0.56.0"),
    ("dask", "dask", "2021.0.0"),
    ("h5py", "h5py", "3.0.0"),
    ("sympy", "sympy", "1.9.0"),
    ("statsmodels", "statsmodels", "0.13.0"),
]

def check_package(name, import_name, min_version):
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", None)
        module_path = getattr(module, "__file__", "ruta no disponible")

        # Qiskit submodules no siempre tienen __version__, obtener desde qiskit.__qiskit_version__
        if name.startswith("qiskit") and version is None:
            try:
                import qiskit
                qv = getattr(qiskit, "__qiskit_version__", {})
                version = qv.get(name.replace("qiskit_", ""), "0.0.0")
            except Exception:
                version = "0.0.0"

        # Rich no siempre tiene __version__
        if name == "rich" and version is None:
            print(f"⚠ {name:<25} instalado (ruta: {module_path}) pero no reporta versión")
            return True

        if version is None:
            print(f"✗ {name:<25} no reporta versión (ruta: {module_path})")
            return False

        if parse(version) >= parse(min_version):
            print(f"✓ {name:<25} {version:<10} (mínimo {min_version}) ruta: {module_path}")
            return True
        else:
            print(f"✗ {name:<25} {version:<10} (mínimo {min_version}) ruta: {module_path}")
            return False

    except ImportError:
        print(f"✗ {name:<25} no instalado")
        return False

def main():
    print("\nVerificando entorno de simulación cuántica en Debian 12\n")
    print(f"Python usado: {sys.executable}\n")
    all_ok = True
    for name, import_name, min_version in packages:
        if not check_package(name, import_name, min_version):
            all_ok = False

    print("\n" + ("✔ Todos los paquetes necesarios están instalados y actualizados."
                 if all_ok else "⚠ Algunos paquetes faltan o tienen versión antigua."))

if __name__ == "__main__":
    main()
