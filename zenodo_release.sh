#!/bin/bash
# create_zenodo_release.sh

# Configuración
VERSION="1.0.0"
RELEASE_DIR="BardoThodol_Zenodo_v${VERSION}"
ZIP_FILE="${RELEASE_DIR}.zip"

echo "Creando release para Zenodo versión ${VERSION}..."

# Crear directorio de release
mkdir -p "${RELEASE_DIR}/paper"
mkdir -p "${RELEASE_DIR}/code/src/core"
mkdir -p "${RELEASE_DIR}/code/src/simulations"
mkdir -p "${RELEASE_DIR}/code/src/utils"
mkdir -p "${RELEASE_DIR}/scripts"

# Copiar archivos del paper
cp "Bardo_Quantum_Model_Paper_20251007.pdf" "${RELEASE_DIR}/paper/"
cp "main.tex" "${RELEASE_DIR}/paper/"
cp "references.bib" "${RELEASE_DIR}/paper/"
cp "compile_paper_corrected.sh" "${RELEASE_DIR}/paper/"

# Copiar código fuente (asumiendo que están en estos directorios relativos)
cp -r src/* "${RELEASE_DIR}/code/src/"
cp requirements.txt "${RELEASE_DIR}/code/"
cp setup.py "${RELEASE_DIR}/code/"

# Copiar scripts
cp install_complete_tex.sh "${RELEASE_DIR}/scripts/"
cp compile_paper_corrected.sh "${RELEASE_DIR}/scripts/"

# Copiar metadatos y README
cp zenodo_metadata.json "${RELEASE_DIR}/"
cp README_ZENODO.md "${RELEASE_DIR}/"

# Crear archivo ZIP
zip -r "${ZIP_FILE}" "${RELEASE_DIR}"

# Limpiar directorio temporal
rm -rf "${RELEASE_DIR}"

echo "✅ Release creado: ${ZIP_FILE}"