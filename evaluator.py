"""
evaluator.py
============
Simula utilidad base vs propuesta usando la feature 'media' como
proxy de demanda esperada (ya que no hay historial transaccional).

Esquema base    → inventario actual I (hoja Inventario)
Esquema propuesto → INV_IDEAL calculado por el modelo

Ecuaciones:
    Ec. (8)  U_vendidas = min(INV, demanda_esperada)
    Ec. (9)  U_merma    = (INV − U_vendidas) × 1[TV ≤ 14]
    Ec. (10) UTILIDAD   = U_vendidas × (P − C) − U_merma × C
"""

import pandas as pd
from typing import List, Tuple


def _utilidad(inv: float, demanda: float, precio: float,
              costo: float, tv: int) -> float:
    vendido = min(inv, demanda)
    merma   = (inv - vendido) if tv <= 14 else 0.0
    return vendido * (precio - costo) - merma * costo


class Evaluator:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def simulate(
        self,
        inv_df       : pd.DataFrame,
        cat_df       : pd.DataFrame,
        features_df  : pd.DataFrame,
        resultado_df : pd.DataFrame,
        fechas_target: List[Tuple],
    ) -> dict:
        """
        Usa 'media' de la tabla de features como demanda esperada semanal.
        """
        cat = cat_df.set_index("Sku")[["Precio", "Costo", "TiempoVida"]]
        inv_base = inv_df.set_index(["Loc", "Sku"])["Inventario"]
        fecha_cols = [c for c, _ in fechas_target]
        n_semanas  = len(fecha_cols)

        # Demanda esperada = feature 'media'
        demanda_idx = features_df.set_index(["Loc", "Sku"])["media"]

        res = resultado_df.merge(
            cat_df[["Sku", "Precio", "Costo", "TiempoVida"]],
            on="Sku", how="left"
        )

        util_base = util_prop = 0.0
        n_rows = 0

        for _, row in res.iterrows():
            loc = int(row["Loc"])
            sku = int(row["Sku"])
            p   = float(row.get("Precio",     30))
            c   = float(row.get("Costo",      18))
            tv  = int(row.get("TiempoVida",   14))
            inv_b = float(inv_base.get((loc, sku), 0))
            dem   = float(demanda_idx.get((loc, sku), inv_b))

            for col_name in fecha_cols:
                inv_p = float(row.get(col_name, inv_b))
                util_base += _utilidad(inv_b, dem, p, c, tv)
                util_prop += _utilidad(inv_p, dem, p, c, tv)
                n_rows += 1

        delta      = util_prop - util_base
        mejora_pct = (delta / abs(util_base) * 100) if util_base != 0 else 0.0

        stats = {
            "util_base" : util_base,
            "util_prop" : util_prop,
            "delta"     : delta,
            "mejora_pct": mejora_pct,
            "n_eval"    : n_rows,
        }
        if self.verbose:
            print(f"      Pares × semanas  : {n_rows:,}")
            print(f"      Utilidad Base    : ${util_base:>14,.2f}")
            print(f"      Utilidad Propuesta: ${util_prop:>13,.2f}")
            print(f"      Mejora           : ${delta:>14,.2f}  ({mejora_pct:+.1f}%)")
        return stats
