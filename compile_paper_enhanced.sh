#!/bin/bash
# compile_paper_enhanced.sh

echo "Compilando documento LaTeX con verificaciones..."

# Verificar que todos los archivos necesarios existen
required_files=("main.tex" "references.bib")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Archivo $file no encontrado"
        exit 1
    fi
done

echo "✅ Todos los archivos necesarios presentes"

# Compilación paso a paso con verificación de errores
echo "Paso 1: Compilación inicial de LaTeX"
if ! pdflatex -interaction=nonstopmode main.tex; then
    echo "❌ Error en primera compilación LaTeX"
    exit 1
fi

echo "Paso 2: Compilación de bibliografía"
if ! bibtex main; then
    echo "❌ Error en compilación BibTeX"
    exit 1
fi

echo "Paso 3: Segunda compilación LaTeX"
if ! pdflatex -interaction=nonstopmode main.tex; then
    echo "❌ Error en segunda compilación LaTeX"
    exit 1
fi

echo "Paso 4: Compilación final"
if ! pdflatex -interaction=nonstopmode main.tex; then
    echo "❌ Error en compilación final"
    exit 1
fi

# Verificar que el PDF se generó correctamente
if [ -f "main.pdf" ]; then
    echo "✅ PDF generado exitosamente: main.pdf"
    
    # Crear copia con nombre descriptivo
    final_name="Bardo_Quantum_Model_Paper_$(date +%Y%m%d).pdf"
    cp main.pdf "$final_name"
    echo "✅ Copia creada: $final_name"
    
    # Verificar tamaño del PDF
    file_size=$(stat -f%z "main.pdf" 2>/dev/null || stat -c%s "main.pdf")
    echo "📊 Tamaño del PDF: $((file_size / 1024)) KB"
else
    echo "❌ Error: PDF no generado"
    exit 1
fi

# Limpiar archivos auxiliares (opcional)
echo "🧹 Limpiando archivos auxiliares..."
rm -f main.aux main.log main.out main.bbl main.blg main.toc main.lof main.lot

echo "🎉 Proceso de compilación completado exitosamente"