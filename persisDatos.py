# Guardar resultados en JSON
import json
with open('resultados_simulacion.json', 'w') as f:
    json.dump(results, f, indent=2)
