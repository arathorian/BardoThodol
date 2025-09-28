"""
Módulo para análisis avanzado de los resultados del Bardo Thödol.
"""
import json
import numpy as np
from typing import Dict, List
from pathlib import Path

class BardoAnalyzer:
    """Analizador de resultados de simulaciones del Bardo."""

    def __init__(self, resultados_file: str = "resultados_simulacion.json"):
        self.resultados_file = Path(resultados_file)
        self.datos = self.cargar_datos()

    def cargar_datos(self) -> Dict:
        """Carga los datos desde el archivo JSON."""
        if not self.resultados_file.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.resultados_file}")

        with open(self.resultados_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def calcular_estadisticas(self) -> Dict:
        """Calcula estadísticas básicas de la simulación."""
        resultados = self.datos['resultados_cuanticos']
        total = sum(resultados.values())

        return {
            'total_ejecuciones': total,
            'numero_estados': len(resultados),
            'estado_mas_probable': max(resultados.items(), key=lambda x: x[1]),
            'estado_menos_probable': min(resultados.items(), key=lambda x: x[1]),
            'entropia_shannon': self.calcular_entropia(resultados),
            'uniformidad': self.calcular_uniformidad(resultados)
        }

    def calcular_entropia(self, resultados: Dict[str, int]) -> float:
        """Calcula la entropía de Shannon de la distribución."""
        total = sum(resultados.values())
        probabilidades = [count / total for count in resultados.values()]
        return -sum(p * np.log2(p) for p in probabilidades if p > 0)

    def calcular_uniformidad(self, resultados: Dict[str, int]) -> float:
        """Calcula qué tan uniforme es la distribución (0-1)."""
        total = sum(resultados.values())
        probabilidades = [count / total for count in resultados.values()]
        max_entropia = np.log2(len(probabilidades))
        entropia_real = self.calcular_entropia(resultados)
        return entropia_real / max_entropia if max_entropia > 0 else 0

    def generar_reporte(self) -> str:
        """Genera un reporte textual de los resultados."""
        stats = self.calcular_estadisticas()

        reporte = f"""
📊 REPORTE DE SIMULACIÓN - BARDO THÖDOL
{'=' * 50}

📈 ESTADÍSTICAS GENERALES:
• Total de ejecuciones: {stats['total_ejecuciones']:,}
• Estados cuánticos únicos: {stats['numero_estados']}
• Entropía de Shannon: {stats['entropia_shannon']:.3f} bits
• Grado de uniformidad: {stats['uniformidad']:.1%}

🎯 DISTRIBUCIÓN:
• Estado más probable: {stats['estado_mas_probable'][0]}
  ({stats['estado_mas_probable'][1]} ocurrencias, {stats['estado_mas_probable'][1]/stats['total_ejecuciones']:.1%})
• Estado menos probable: {stats['estado_menos_probable'][0]}
  ({stats['estado_menos_probable'][1]} ocurrencias, {stats['estado_menos_probable'][1]/stats['total_ejecuciones']:.1%})

🌌 ESTADOS DEL BARDO:
"""
        for i, estado in enumerate(self.datos['estados_bardo']):
            prob = self.datos['probabilidades'][i]
            reporte += f"• {estado}: {prob:.1%}\n"

        return reporte

    def guardar_reporte(self, filename: str = "reporte_analisis.txt"):
        """Guarda el reporte en un archivo de texto."""
        reporte = self.generar_reporte()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(reporte)
        print(f"📄 Reporte guardado en: {filename}")

# Función de conveniencia
def analizar_simulacion(archivo: str = "resultados_simulacion.json"):
    """Función rápida para analizar una simulación."""
    analyzer = BardoAnalyzer(archivo)
    print(analyzer.generar_reporte())
    return analyzer
