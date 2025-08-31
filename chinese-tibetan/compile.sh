#!/bin/bash
echo "Compilando documento chino-tibetano..."
echo ""

# Verificar que xelatex está disponible
if ! command -v xelatex &> /dev/null; then
    echo "❌ XeLaTeX no encontrado. Instale TeX Live o MiKTeX."
    exit 1
fi

# Primera compilación
xelatex main.tex
if [ $? -ne 0 ]; then
    echo "❌ Error en primera compilación"
    exit 1
fi

# Compilación de bibliografía
biber main
if [ $? -ne 0 ]; then
    echo "❌ Error en compilación de bibliografía"
    exit 1
fi

# Compilaciones adicionales
xelatex main.tex
xelatex main.tex

echo "✅ Compilación completada: main.pdf"
