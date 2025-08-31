# BardoThodol - Simulación Cuántica Multilingüe

Este proyecto contiene la investigación "Simulación Cuántica de Estados Post-Mortem del Bardo Thödol" en cuatro idiomas: español, inglés, chino tradicional y tibetano.

## Estructura del Proyecto

BardoThodol/
├── spanish-english/          # Versión español-inglés
│   ├── main.tex             # Documento principal LaTeX
│   ├── references.bib       # Base de datos bibliográfica
│   ├── compile.bat          # Script de compilación (Windows)
│   ├── compile.sh           # Script de compilación (Linux/macOS)
│   └── figures/             # Figuras y diagramas
├── chinese-tibetan/         # Versión chino-tibetano
│   ├── main.tex             # Documento principal LaTeX
│   ├── references.bib       # Base de datos bibliográfica
│   ├── compile.bat          # Script de compilación (Windows)
│   ├── compile.sh           # Script de compilación (Linux/macOS)
│   └── figures/             # Figuras y diagramas
├── resources/               # Recursos adicionales
│   ├── translation_guide.md # Guía de traducción
│   └── font_installation.md # Instrucciones de fuentes
├── README.md               # Este archivo
└── LICENSE                 # Licencia MIT

## Requisitos de Compilación

### Versión Español-Inglés
- TeX Live o MiKTeX (Windows)
- Paquetes: biblatex-nature, quantikz

### Versión Chino-Tibetano
- XeLaTeX (parte de TeX Live/MiKTeX)
- Fuentes: Noto Serif CJK TC, Noto Serif Tibetan

## Instrucciones de Compilación

### Windows (Git Bash)
# Versión español-inglés
cd spanish-english
./compile.sh

# Versión chino-tibetano  
cd ../chinese-tibetan
./compile.sh

### Windows (CMD)
:: Versión español-inglés
cd spanish-english
compile.bat

:: Versión chino-tibetano
cd ..\chinese-tibetan
compile.bat

## Autor
- GitHub: arathorian
- Email: horaciohamann@gmail.com

## Agradecimientos
- Qiskit e IBM Quantum por acceso a simuladores cuánticos
- DeepSeek AI (https://deepseek.com) por asistencia computacional
- Traductores que hicieron posible las versiones multilingües

## Licencia
MIT License - Ver LICENSE para detalles.
