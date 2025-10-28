#!/bin/bash
# scripts/setup_environment.sh - Configuración compatible con Debian

set -e  # Exit on error

echo "🔧 Configurando entorno para Bardo Quantum Model (Debian/Ubuntu)"

# Detectar sistema operativo
OS_TYPE=$(uname -s)

if [ "$OS_TYPE" = "Linux" ]; then
    echo "✅ Sistema Linux detectado"

    # Instalar dependencias del sistema para Debian/Ubuntu
    if command -v apt-get &> /dev/null; then
        echo "📦 Instalando dependencias del sistema (Debian/Ubuntu)"
        sudo apt-get update
        sudo apt-get install -y python3-pip python3-venv build-essential
    elif command -v yum &> /dev/null; then
        echo "📦 Instalando dependencias del sistema (RHEL/CentOS)"
        sudo yum install -y python3-pip python3-venv gcc
    fi
fi

# Crear entorno virtual
echo "🐍 Creando entorno virtual Python"
python3 -m venv bardo_env
source bardo_env/bin/activate

# Instalar dependencias Python
echo "📚 Instalando dependencias Python"
pip install --upgrade pip
pip install -r requirements.txt

# Verificar instalación
echo "✅ Verificando instalación"
python -c "import qiskit, numpy, matplotlib; print('✅ Todas las dependencias instaladas correctamente')"

echo "🎉 Configuración completada. Active el entorno con: source bardo_env/bin/activate"
