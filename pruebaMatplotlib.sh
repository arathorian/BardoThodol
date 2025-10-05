# Probar la importación de matplotlib
python -c "import matplotlib.pyplot as plt; print('✅ matplotlib.pyplot funciona correctamente')"

# Probar todas las importaciones del proyecto
python -c "
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import qiskit
from qiskit_aer import Aer
print('✅ Todas las dependencias funcionan')
"
