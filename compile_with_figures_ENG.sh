#!/bin/bash
echo "=== Compilación Completa Bardo Thodol en Ingles ==="

# 1. Generar figuras
echo "1. Generando figuras..."
python generate_minimal_figures.py

# 2. Compilar LaTeX
echo "2. Compilando documento LaTeX..."
pdflatex -interaction=nonstopmode main_eng.tex
bibtex main_eng
pdflatex -interaction=nonstopmode main_eng.tex
pdflatex -interaction=nonstopmode main_eng.tex

# 3. Limpiar
# echo "3. Limpiando archivos temporales..."
# rm -f *.aux *.log *.toc *.out *.lof *.lot *.bbl *.blg *.synctex.gz

echo "=== COMPILACIÓN EXITOSA ==="
echo "Documento: main_eng.pdf"
echo "Figuras: figures/"