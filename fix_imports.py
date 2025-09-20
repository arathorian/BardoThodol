import os
import re

# Reglas de reemplazo conocidas (del refactor anterior)
IMPORT_FIXES = {
    r"\bfrom\s+models\.": "from core.",
    r"\bimport\s+models\.": "import core.",

    r"\bfrom\s+utils\.states\b": "from core.states",
    r"\bimport\s+utils\.states\b": "import core.states",

    r"\bfrom\s+utils\.": "from utils.",
    r"\bimport\s+utils\.": "import utils.",
}

def fix_imports_in_file(filepath):
    """
    Corrige imports en un archivo Python según reglas conocidas.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = content
    changes = []
    for pattern, replacement in IMPORT_FIXES.items():
        new_content, n = re.subn(pattern, replacement, new_content)
        if n > 0:
            changes.append((pattern, replacement, n))

    if changes:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"✅ Arreglados imports en {filepath}")
        for pat, rep, count in changes:
            print(f"   • {count} reemplazo(s): {pat} → {rep}")


def process_project(base_dir="."):
    """
    Recorre todos los .py del proyecto y aplica correcciones de imports.
    """
    print("🔧 Corrigiendo imports en el proyecto...")

    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".py") and "venv" not in root and "__pycache__" not in root:
                filepath = os.path.join(root, f)
                fix_imports_in_file(filepath)

    print("✨ Proceso completado. Revisa y ejecuta tests.")


if __name__ == "__main__":
    process_project(".")
