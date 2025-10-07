#!/bin/bash
# compile_paper_corrected.sh

echo "Compilando documento LaTeX con Biber..."

# Verificar que todos los archivos necesarios existen
required_files=("main.tex" "references.bib")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Archivo $file no encontrado"
        exit 1
    fi
done

echo "✅ Todos los archivos necesarios presentes"

# Compilación paso a paso con Biber
echo "Paso 1: Compilación inicial de LaTeX"
if ! pdflatex -interaction=nonstopmode main.tex; then
    echo "❌ Error en primera compilación LaTeX"
    exit 1
fi

echo "Paso 2: Procesamiento de bibliografía con Biber"
if ! biber main; then
    echo "❌ Error en Biber"
    echo "Instalando Biber si es necesario..."
    sudo apt install -y biber
    if ! biber main; then
        echo "❌ Error persistente en Biber"
        exit 1
    fi
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
    
    # Verificar que la bibliografía se incluyó
    if grep -q "Hameroff" main.pdf; then
        echo "✅ Bibliografía incluida correctamente"
    else
        echo "⚠️  Advertencia: La bibliografía podría no estar incluida"
    fi
else
    echo "❌ Error: PDF no generado"
    exit 1
fi

# Limpiar archivos auxiliares
echo "🧹 Limpiando archivos auxiliares..."
rm -f main.aux main.log main.out main.bbl main.blg main.toc main.lof main.lot main.run.xml main.bcf

echo "🎉 Proceso de compilación completado exitosamente"