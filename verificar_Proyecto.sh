# 1. Ejecutar verificación
python verificar_proyecto.py

# 2. Si hay problemas, crearlos automáticamente
python -c "
import os
os.makedirs('src/core', exist_ok=True)
os.makedirs('src/visualization', exist_ok=True)
open('src/__init__.py', 'a').close()
open('src/core/__init__.py', 'a').close()
open('src/visualization/__init__.py', 'a').close()
print('Estructura básica creada')
"

# 3. Ejecutar main.py
python main.py
