"""
reabasto.py
===========
Motor principal de re-abasto.

Ecuaciones:
    Ec. (2)  V̂          = α_k + media
    Ec. (3)  PEDIDO      = max(0, S_k − I)
    Ec. (4)  PEDIDO_RED  = ⌈PEDIDO / TS⌉ × TS
    Ec. (5)  INV_IDEAL   = I + PEDIDO_RED
    Ec. (9)  Remanente_s = max(0, INV_IDEAL_{s-1} − V̂)
    Ec. (10) I_s = 0 si TV ≤ 14  |  Remanente si TV > 14
"""

import math
import pandas as pd
import numpy as np
from typing import Tuple, List

from src.forecaster   import Forecaster
from src.data_loader  import _any_to_timestamp   # reutilizamos el mismo parser


def _parse_fechas(res_df: pd.DataFrame) -> List[Tuple]:
    """
    Detecta columnas de fecha en res_df.
    Funciona con Timestamp, string dd/mm/yyyy y número serial Excel.
    Ignora columnas Loc/Sku y NaN.
    """
    SKIP = {"loc", "sku", "location", "tienda", "store",
            "articulo", "producto", "item", "codigo"}

    pairs = []
    for col in res_df.columns:
        # Ignorar NaN
        if col is None:
            continue
        if isinstance(col, float) and np.isnan(col):
            continue

        # Ignorar identificadores
        if isinstance(col, str):
            if col.lower().strip().replace(" ", "") in SKIP:
                continue
            if col.startswith("Unnamed"):
                continue

        ts = _any_to_timestamp(col)
        if ts is not None:
            pairs.append((col, ts))

    pairs.sort(key=lambda x: x[1])

    if not pairs:
        raise ValueError(
            "No se encontraron columnas de fecha en la hoja Resultados.\n"
            f"Columnas encontradas: {list(res_df.columns)}\n"
            "Asegúrate de que las fechas estén en la fila 1 (encabezado).\n"
            "Tip: el data_loader ahora lee sin header=0, "
            "las fechas deben estar en la primera fila del Excel."
        )

    return pairs


def _nivel_objetivo(vhat, p90, factor_S, usar_p90):
    if usar_p90:
        return max(float(p90), vhat)
    return vhat * float(factor_S)


def _pedido_redondeado(pedido, tam_surtido):
    if pedido <= 0:
        return 0
    ts = max(int(tam_surtido), 1)
    return int(math.ceil(pedido / ts) * ts)


class ReabastoEngine:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run(self, inv_df, cat_df, res_df, features_df,
            forecaster: Forecaster):

        fechas_target = _parse_fechas(res_df)

        if self.verbose:
            print(f"      Fechas detectadas ({len(fechas_target)}):")
            for col, ts in fechas_target:
                print(f"        {ts.strftime('%d/%m/%Y')}")

        # Merge inventario × catálogo × features
        pares = inv_df.merge(cat_df, on="Sku", how="left")
        feat_cols = ["Loc", "Sku", "media", "p90",
                     "alpha_cluster", "factor_S", "usar_p90", "cluster"]
        feat_cols = [c for c in feat_cols if c in features_df.columns]
        pares = pares.merge(features_df[feat_cols], on=["Loc", "Sku"], how="left")

        pares["media"]         = pares["media"].fillna(0.0)
        pares["p90"]           = pares["p90"].fillna(1.0)
        pares["alpha_cluster"] = pares["alpha_cluster"].fillna(1.0)
        pares["factor_S"]      = pares["factor_S"].fillna(1.0)
        pares["usar_p90"]      = pares["usar_p90"].fillna(False)
        pares["cluster"]       = pares["cluster"].fillna("A")
        pares["TiempoVida"]    = pares["TiempoVida"].fillna(14).astype(int)
        pares["TamSurtido"]    = pares["TamSurtido"].fillna(1).astype(int)

        total      = len(pares)
        resultados = []

        for i, row in pares.iterrows():
            loc      = int(row["Loc"])
            sku      = int(row["Sku"])
            tv       = int(row["TiempoVida"])
            ts_      = int(row["TamSurtido"])
            media    = float(row["media"])
            alpha    = float(row["alpha_cluster"])
            factor_s = float(row["factor_S"])
            usar_p90_= bool(row["usar_p90"])
            p90      = float(row["p90"])

            vhat = forecaster.pronostico(media_cluster=media, alpha=alpha)
            S    = _nivel_objetivo(vhat, p90, factor_s, usar_p90_)

            I_actual = float(row["Inventario"])
            fila = {"Loc": loc, "Sku": sku}

            for col_original, _ in fechas_target:
                pedido     = max(0.0, S - I_actual)
                pedido_red = _pedido_redondeado(pedido, ts_)
                inv_ideal  = I_actual + pedido_red
                fila[col_original] = int(inv_ideal)

                remanente = max(0.0, inv_ideal - vhat)
                I_actual  = 0.0 if tv <= 14 else remanente

            resultados.append(fila)

            if self.verbose and (i + 1) % 500 == 0:
                print(f"      Procesados: {i+1:,}/{total:,}", end="\r")

        if self.verbose:
            print(f"      Procesados: {total:,}/{total:,}  ✓")

        return pd.DataFrame(resultados), fechas_target
