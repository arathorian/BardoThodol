#!/bin/bash
# install_tex_dependencies.sh

echo "Instalando paquetes LaTeX necesarios..."

# Instalar paquetes esenciales
sudo apt update
sudo apt install -y texlive-latex-base texlive-latex-extra
sudo apt install -y texlive-science texlive-publishers
sudo apt install -y texlive-lang-spanish texlive-lang-english
sudo apt install -y texlive-bibtex-extra biblatex
sudo apt install -y latexmk

# Paquetes específicos requeridos
sudo apt install -y texlive-fonts-extra
sudo apt install -y lmodern cm-super

echo "✅ Paquetes LaTeX instalados correctamente"

# Verificar instalación
echo "Verificando paquetes críticos..."
kpsewhich braket.sty
kpsewhich biblatex.sty
kpsewhich babel.sty

echo "Instalación completada"