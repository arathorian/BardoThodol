import os
import ast

def find_imports_in_file(filepath):
    """
    Devuelve lista de todos los módulos importados en un archivo Python.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def check_imports(base_dir="."):
    """
    Revisa todos los imports en archivos .py y detecta si apuntan a rutas inexistentes.
    """
    print("🔍 Revisando imports en el proyecto...")
    python_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".py") and "venv" not in root and "__pycache__" not in root:
                python_files.append(os.path.join(root, f))

    # Rutas disponibles (simples, sin .py)
    available_modules = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".py"):
                mod_path = os.path.relpath(os.path.join(root, f), base_dir)
                mod_name = mod_path.replace("/", ".").replace("\\", ".").replace(".py", "")
                available_modules.append(mod_name)

    broken_imports = {}
    for pyfile in python_files:
        imports = find_imports_in_file(pyfile)
        for imp in imports:
            # Chequeo simple: si no existe en los módulos locales y no parece ser lib estándar
            if not any(imp == m or imp.startswith(m + ".") for m in available_modules):
                if imp not in ["os", "sys", "math", "time", "re", "json", "unittest", 
                               "numpy", "scipy", "matplotlib", "seaborn", "qiskit"]:
                    broken_imports.setdefault(pyfile, []).append(imp)

    if broken_imports:
        print("⚠️ Se encontraron imports posiblemente rotos:")
        for file, imports in broken_imports.items():
            print(f"  📄 {file}: {', '.join(imports)}")
    else:
        print("✅ Todos los imports parecen correctos.")


if __name__ == "__main__":
    check_imports(".")
