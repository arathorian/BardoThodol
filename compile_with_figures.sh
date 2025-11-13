#!/bin/bash
echo "=== Compilación Completa Bardo Thodol ==="

# 1. Generar figuras
echo "1. Generando figuras..."
python generate_minimal_figures.py

# 2. Compilar LaTeX
echo "2. Compilando documento LaTeX..."
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# 3. Limpiar
# echo "3. Limpiando archivos temporales..."
# rm -f *.aux *.log *.toc *.out *.lof *.lot *.bbl *.blg *.synctex.gz

echo "=== COMPILACIÓN EXITOSA ==="
echo "Documento: main.pdf"
echo "Figuras: figures/"