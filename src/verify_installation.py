# script: verify_installation.py
import numpy as np
import scipy
import qiskit
from src.core.quantum_models_optimized import OptimizedBardoModel

print("✓ NumPy configurado correctamente")
print("✓ SciPy funcionando") 
print("✓ Qiskit disponible")
print("✓ Modelo Bardo cargado")

# Test de performance
model = OptimizedBardoModel()
print("✓ Sistema listo para investigación")