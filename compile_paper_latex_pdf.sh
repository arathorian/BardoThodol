#!/bin/bash
# compile_paper_latex_pdf.sh - VERSIÓN CORREGIDA

echo "=== Compilación BardoThodol Quantum Framework ==="

# Verificar y crear directorios necesarios
mkdir -p graphics/generated data scripts

# Verificar si el script Python existe
if [ ! -f "scripts/generate_basic_plots.py" ]; then
    echo "ERROR: Script generate_basic_plots.py no encontrado"
    echo "Por favor, crea el script en scripts/generate_basic_plots.py"
    exit 1
fi

# Generar gráficos primero (si no existen)
if [ ! -f "graphics/generated/qutrit_space.png" ]; then
    echo "Generando gráficos automáticamente..."
    python3 scripts/generate_basic_plots.py
else
    echo "✓ Gráficos ya existen, omitiendo generación"
fi

# Compilar LaTeX
echo "Compilando documento LaTeX..."
pdflatex -interaction=nonstopmode main.tex

# Compilar bibliografía si existe
if [ -f "main.aux" ]; then
    pdflatex -interaction=nonstopmode main.tex
fi

echo "=== Compilación completada ==="
echo "Documento generado: main.pdf"

# Verificar si el PDF se creó
if [ -f "main.pdf" ]; then
    echo "✅ ¡PDF generado exitosamente!"
else
    echo "❌ Error: No se pudo generar el PDF"
fi