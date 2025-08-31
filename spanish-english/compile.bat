@echo off
echo Compilando documento español-inglés...
echo.

REM Primera compilación
pdflatex main.tex
if errorlevel 1 (
    echo ❌ Error en primera compilación
    exit /b 1
)

REM Compilación de bibliografía
biber main
if errorlevel 1 (
    echo ❌ Error en compilación de bibliografía
    exit /b 1
)

REM Compilaciones adicionales
pdflatex main.tex
pdflatex main.tex

echo ✅ Compilación completada: main.pdf
