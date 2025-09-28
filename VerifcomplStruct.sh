# Verificar qué archivos tenemos actualmente
find . -name "*.py" -type f | grep -v __pycache__ | sort
