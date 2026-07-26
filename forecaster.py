"""
forecaster.py
=============
Pronóstico de demanda semanal basado en las features del cluster.

NO usa datos históricos ni índice estacional.
La media semanal (feature 'media') ya resume el comportamiento histórico
de demanda. El clustering calibra α para ajustar el nivel de servicio.

Ecuación (del reporte técnico):
    Ec. (2)  V̂ = α_cluster + media_cluster

donde:
    media_cluster — feature 'media' calculada sobre el historial y
                    disponible en la tabla de features
    α_cluster     — piso de seguridad asignado por cluster
"""


class Forecaster:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def pronostico(self, media_cluster: float, alpha: float = 1.0) -> float:
        """
        Ec. (2): V̂ = α_cluster + media_cluster

        Parámetros
        ----------
        media_cluster : feature 'media' del par (Loc, SKU)
        alpha         : piso de seguridad del cluster (α_cluster)

        Retorna
        -------
        float : pronóstico de demanda semanal
        """
        return float(alpha) + max(0.0, float(media_cluster))
