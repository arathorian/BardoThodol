#!/usr/bin/env python3
import os
import sys

def setup_project():
    """Configura la estructura del proyecto."""
    print("🔧 Configurando proyecto BardoThodol...")

    # Crear directorios necesarios
    directories = ['src', 'src/core', 'src/visualization', 'src/data', 'tests']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directorio creado: {directory}")

    # Crear archivos __init__.py
    init_files = ['src/__init__.py', 'src/core/__init__.py',
                  'src/visualization/__init__.py', 'src/data/__init__.py',
                  'tests/__init__.py']

    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('# Package initialization\n')
        print(f"📄 Archivo creado: {init_file}")

    print("✅ Estructura del proyecto configurada correctamente")

if __name__ == "__main__":
    setup_project()
