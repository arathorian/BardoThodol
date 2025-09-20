import os
import shutil
import re

# Carpetas a crear
TARGET_DIRS = ["core", "utils", "tests", "docs"]

# Reglas de movimiento (src -> dst)
MOVE_RULES = {
    "models/bardo_states.py": "core/states.py",
    "models/post_mortem_states.py": "core/states.py",
    "utils/states.py": "core/states.py",
    "quantum_circuit.py": "core/circuit_factory.py",
    "simulation.py": "core/circuit_factory.py",
    "bardo_simulation.py": "core/circuit_factory.py",
    "visualizations.py": "utils/visualizations.py",
}

# Archivos obsoletos
DELETE_FILES = ["legacy_bardo_sim.py", "test_old.py"]

# Función para crear carpetas
def ensure_dirs():
    for d in TARGET_DIRS:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"📂 Creada carpeta: {d}")

# Función para mover archivos
def move_files():
    for src, dst in MOVE_RULES.items():
        if os.path.exists(src):
            dst_dir = os.path.dirname(dst)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            with open(src, "r", encoding="utf-8") as f:
                content = f.read()
            # Actualizar imports
            content = re.sub(r"from\s+models\.", "from core.", content)
            content = re.sub(r"from\s+utils\.", "from utils.", content)
            # Guardar en nuevo destino
            with open(dst, "a", encoding="utf-8") as f:  # "a" para consolidar
                f.write("\n\n# ---- Módulo fusionado ----\n\n")
                f.write(content)
            os.remove(src)
            print(f"✅ Movido {src} → {dst}")

# Función para borrar archivos obsoletos
def delete_obsolete():
    for f in DELETE_FILES:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑️ Eliminado archivo obsoleto: {f}")

# Crear .gitignore básico
def create_gitignore():
    if not os.path.exists(".gitignore"):
        with open(".gitignore", "w", encoding="utf-8") as f:
            f.write(
                """# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/

# LaTeX
*.aux
*.log
*.toc
*.out

# Env
.env
ibm_config.json
"""
            )
        print("📄 Creado .gitignore básico")

# Crear requirements.txt si no existe
def create_requirements():
    if not os.path.exists("requirements.txt"):
        with open("requirements.txt", "w", encoding="utf-8") as f:
            f.write(
                """qiskit==0.43.0
numpy==1.24.0
matplotlib==3.7.0
scipy==1.11.3
seaborn==0.12.2
"""
            )
        print("📄 Creado requirements.txt con versiones fijas")

# Crear placeholder de tests
def create_tests():
    test_file = os.path.join("tests", "test_placeholder.py")
    if not os.path.exists(test_file):
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(
                """import unittest

class TestPlaceholder(unittest.TestCase):
    def test_dummy(self):
        self.assertTrue(True)
"""
            )
        print("🧪 Creado tests/test_placeholder.py")

# MAIN
if __name__ == "__main__":
    print("🔧 Iniciando refactorización del proyecto BardoThodol...")
    ensure_dirs()
    move_files()
    delete_obsolete()
    create_gitignore()
    create_requirements()
    create_tests()
    print("✨ Refactorización completada. Revisa imports y ejecuta tests.")
