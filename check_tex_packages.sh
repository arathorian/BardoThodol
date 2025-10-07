#!/bin/bash
# check_tex_packages.sh

echo "Verificando paquetes LaTeX instalados..."

# Paquetes críticos a verificar
critical_packages=(
    "braket.sty"
    "babel.sty" 
    "biblatex.sty"
    "amsmath.sty"
    "amssymb.sty"
    "graphicx.sty"
    "hyperref.sty"
)

echo "Paquetes críticos requeridos:"
for package in "${critical_packages[@]}"; do
    if kpsewhich "$package" > /dev/null; then
        echo "✅ $package"
    else
        echo "❌ $package - NO ENCONTRADO"
    fi
done

# Verificar versión de texlive
echo -e "\nInformación del sistema LaTeX:"
pdflatex --version | head -n1
latex --version | head -n1
bibtex --version | head -n1

echo -e "\nVerificación completada"