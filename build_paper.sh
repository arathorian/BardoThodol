#!/bin/bash
# Script de construcción completa del artículo
# Genera figuras y compila el documento LaTeX

echo "=== Sistema de Construcción Bardo Thodol ==="
echo "Repositorio: https://github.com/arathorian/BardoThodol"

# Crear directorio de figuras
mkdir -p figures

# Generar todas las figuras
echo "1. Generando figuras científicas..."
python generate_figures.py

# Compilar documento LaTeX
echo "2. Compilando documento LaTeX..."
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# Limpiar archivos auxiliares
echo "3. Limpiando archivos temporales..."
rm -f *.aux *.log *.toc *.out *.lof *.lot *.bbl *.blg *.synctex.gz

echo "=== Construcción completada ==="
echo "Documento principal: main.pdf"
echo "Figuras generadas: figures/"