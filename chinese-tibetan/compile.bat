@echo off
echo Compilando documento chino-tibetano...
echo.

REM Verificar que xelatex está disponible
where xelatex >nul 2>&1
if errorlevel 1 (
    echo ❌ XeLaTeX no encontrado. Instale TeX Live o MiKTeX.
    exit /b 1
)

REM Primera compilación
xelatex main.tex
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
xelatex main.tex
xelatex main.tex

echo ✅ Compilación completada: main.pdf
