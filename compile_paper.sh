#!/bin/bash
# compile_paper.sh

echo "Compilando documento LaTeX..."

# Compilar PDF principal
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Verificar que el PDF se generó correctamente
if [ -f "main.pdf" ]; then
    echo "✅ PDF generado exitosamente: main.pdf"
    
    # Crear copia con nombre descriptivo
    cp main.pdf "Bardo_Quantum_Model_Paper_$(date +%Y%m%d).pdf"
    echo "✅ Copia creada: Bardo_Quantum_Model_Paper_$(date +%Y%m%d).pdf"
else
    echo "❌ Error en la generación del PDF"
    exit 1
fi

# Limpiar archivos auxiliares
rm -f main.aux main.log main.out main.bbl main.blg main.toc
echo "✅ Archivos auxiliares limpiados"