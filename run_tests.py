import subprocess
import sys
import shutil

def ensure_pytest():
    """Verifica si pytest está instalado, si no lo instala."""
    if shutil.which("pytest") is None:
        print("⚠️ pytest no encontrado. Instalando...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])

def run_pytest():
    """Ejecuta pytest y muestra resultados claros (compatible con Python 3.6+)."""
    print("🧪 Ejecutando tests con pytest...\n")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--maxfail=5", "--disable-warnings", "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        print(result.stdout)

        if result.returncode == 0:
            print("✅ Todos los tests pasaron exitosamente.")
        else:
            print("⚠️ Algunos tests fallaron. Revisa el log arriba.")
            print(result.stderr)

    except Exception as e:
        print(f"❌ Error al ejecutar pytest: {e}")

if __name__ == "__main__":
    ensure_pytest()
    run_pytest()
