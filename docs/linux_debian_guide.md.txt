
## 🐧 Guía Linux Debian - Entorno Óptimo

### docs/linux_debian_guide.md

```markdown
# Guía de Implementación en Linux Debian

## Ventajas de Debian para Investigación Científica

**Estabilidad**: Release cycles predecibles para investigación a largo plazo
**Repositorios científicos**: Paquetes optimizados para computación numérica
**Transparencia**: Software libre permite verificación completa

## Instalación y Configuración

### 1. Dependencias del Sistema

```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias base
sudo apt install python3.11 python3.11-venv python3.11-dev
sudo apt install build-essential cmake git
sudo apt install libopenblas-dev liblapack-dev libatlas-base-dev

# Instalar herramientas científicas
sudo apt install octave r-base r-cran-ggplot2
sudo apt install texlive-science texlive-publishers texlive-latex-extra