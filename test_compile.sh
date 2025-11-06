#!/bin/bash
echo "=== Compilación de prueba ==="

# Compilación mínima para detectar errores
pdflatex -interaction=nonstopmode main.tex

if [ $? -eq 0 ]; then
    echo "✓ Compilación exitosa"
else
    echo "✗ Error en compilación - revisar main.log"
fi