#!/bin/bash
echo "Compilando documento español-inglés..."
echo ""

# Primera compilación
pdflatex main.tex
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
pdflatex main.tex
pdflatex main.tex

echo "✅ Compilación completada: main.pdf"
