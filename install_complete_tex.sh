#!/bin/bash
# install_complete_tex.sh

echo "Instalando sistema LaTeX completo con Biber..."

# Actualizar sistema
sudo apt update

# Instalar paquetes LaTeX esenciales
sudo apt install -y \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-science \
    texlive-publishers \
    texlive-lang-spanish \
    texlive-lang-english \
    texlive-bibtex-extra \
    biber \
    latexmk

# Paquetes específicos requeridos
sudo apt install -y \
    texlive-fonts-extra \
    lmodern \
    cm-super

# Verificar instalación
echo "Verificando instalación..."
if command -v biber &> /dev/null; then
    echo "✅ Biber instalado correctamente"
else
    echo "❌ Biber no está instalado"
    exit 1
fi

if command -v pdflatex &> /dev/null; then
    echo "✅ pdflatex instalado correctamente"
else
    echo "❌ pdflatex no está instalado"
    exit 1
fi

# Verificar paquetes LaTeX
echo "Verificando paquetes LaTeX críticos..."
for package in braket.sty biblatex.sty babel.sty; do
    if kpsewhich "$package" > /dev/null; then
        echo "✅ $package"
    else
        echo "❌ $package - FALTANTE"
    fi
done

echo "🎉 Instalación completada. Ejecuta: ./compile_paper_corrected.sh"