"""
data_loader.py
==============
Dos métodos de carga:
    load_excel(path)         → Inventario, CatSku, Resultados
    load_features_csv(path)  → CSV de features con separador | (pipe)
"""

import pandas as pd
import numpy as np
import io

_COL_MAP = {
    "loc":     ["loc", "location", "tienda", "store", "sucursal"],
    "sku":     ["sku", "articulo", "producto", "item", "codigo"],
    "inv":     ["inventario", "inventory", "stock", "inv", "existencia"],
    "precio":  ["precio", "price", "pvp", "p"],
    "costo":   ["costo", "cost", "c", "costo_unitario"],
    "vida":    ["tiempovida", "tiempo_vida", "vidautil", "vida_util",
                "vida", "shelf_life", "shelflife", "life"],
    "surtido": ["tamañosurtido", "tamanosurtido", "tamaño_surtido",
                "tamsurtido", "tam_surtido", "pack_size", "packsize",
                "surtido", "caja"],
}

_FEAT_NUM_COLS = [
    "media", "std", "cv", "p90", "intermitencia", "acf1",
    "ratio_demanda_vida", "ratio_surtido", "margen_relativo", "costo_unitario",
]


def _find_col(df: pd.DataFrame, key: str) -> str:
    cols_lower = {c.lower().strip().replace(" ", ""): c for c in df.columns}
    for cand in _COL_MAP[key]:
        if cand in cols_lower:
            return cols_lower[cand]
    raise KeyError(
        f"No se encontró columna para '{key}'. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def _any_to_timestamp(val) -> pd.Timestamp:
    """Convierte cualquier tipo a Timestamp. Retorna None si no es fecha."""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val
    if hasattr(val, 'year'):
        return pd.Timestamp(val)
    if isinstance(val, str):
        val = val.strip()
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y",
                    "%d-%m-%Y", "%Y/%m/%d"):
            try:
                return pd.Timestamp(pd.to_datetime(val, format=fmt))
            except Exception:
                pass
        try:
            return pd.Timestamp(pd.to_datetime(val, dayfirst=True))
        except Exception:
            pass
    if isinstance(val, (int, float)):
        try:
            return pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(val))
        except Exception:
            pass
    return None


def _detect_separator(header_line: str) -> tuple:
    """
    Detecta el separador de columnas contando ocurrencias en el encabezado.
    El encabezado no tiene decimales, así que el separador más frecuente
    es el correcto.

    Retorna (sep, decimal).
    """
    # Prioridad: | y tab primero (no aparecen en números)
    candidates = ['|', '\t', ';', ',']
    counts = {s: header_line.count(s) for s in candidates}
    sep = max(counts, key=counts.get)
    # Si sep es ';' los decimales probablemente son ','
    decimal = ',' if sep == ';' else '.'
    return sep, decimal


def _read_csv_robust(path: str) -> pd.DataFrame:
    """
    Lee el CSV detectando automáticamente el separador desde el encabezado.
    Soporta: | (pipe), ; (punto y coma), , (coma), \t (tab).
    """
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
        raw = f.read()

    lines = [l for l in raw.splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"El CSV está vacío: {path}")

    # Detectar separador desde el encabezado (sin decimales)
    sep, decimal = _detect_separator(lines[0])

    # Intentar con el separador detectado
    try:
        df = pd.read_csv(io.StringIO(raw), sep=sep, decimal=decimal)
        if len(df.columns) > 3:
            return df
    except Exception:
        pass

    # Fallback: probar todas las combinaciones en orden
    for s, d in [('|', '.'), (';', ','), (';', '.'), (',', '.'), ('\t', '.')]:
        try:
            df = pd.read_csv(io.StringIO(raw), sep=s, decimal=d)
            if len(df.columns) > 3:
                return df
        except Exception:
            pass

    raise ValueError(
        f"No se pudo leer el CSV con ningún separador conocido.\n"
        f"Encabezado: {lines[0][:200]}"
    )


class DataLoader:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def load_excel(self, path: str):
        inv_df = self._load_inventario(path)
        cat_df = self._load_catsku(path)
        res_df = self._load_resultados(path)
        return inv_df, cat_df, res_df

    def _load_inventario(self, path: str) -> pd.DataFrame:
        df = pd.read_excel(path, sheet_name="Inventario")
        df.columns = df.columns.str.strip()
        df = df[[_find_col(df, "loc"),
                 _find_col(df, "sku"),
                 _find_col(df, "inv")]].copy()
        df.columns = ["Loc", "Sku", "Inventario"]
        df["Loc"]        = df["Loc"].astype(int)
        df["Sku"]        = df["Sku"].astype(int)
        df["Inventario"] = pd.to_numeric(df["Inventario"], errors="coerce").fillna(0)
        if self.verbose:
            print(f"      Inventario   : {len(df):,} filas")
        return df

    def _load_catsku(self, path: str) -> pd.DataFrame:
        df = pd.read_excel(path, sheet_name="CatSku")
        df.columns = df.columns.str.strip()
        df = df[[_find_col(df, "sku"),
                 _find_col(df, "precio"),
                 _find_col(df, "costo"),
                 _find_col(df, "vida"),
                 _find_col(df, "surtido")]].copy()
        df.columns = ["Sku", "Precio", "Costo", "TiempoVida", "TamSurtido"]
        df["Sku"]        = df["Sku"].astype(int)
        df["Precio"]     = pd.to_numeric(df["Precio"],     errors="coerce").fillna(30)
        df["Costo"]      = pd.to_numeric(df["Costo"],      errors="coerce").fillna(18)
        df["TiempoVida"] = pd.to_numeric(df["TiempoVida"], errors="coerce").fillna(14).astype(int)
        df["TamSurtido"] = pd.to_numeric(df["TamSurtido"], errors="coerce").fillna(1).astype(int)
        df["TamSurtido"] = df["TamSurtido"].clip(lower=1)
        if self.verbose:
            print(f"      CatSku       : {len(df):,} SKUs")
        return df

    def _load_resultados(self, path: str) -> pd.DataFrame:
        """Lee sin header para capturar fechas datetime correctamente."""
        raw = pd.read_excel(path, sheet_name="Resultados", header=None)
        headers = raw.iloc[0].tolist()
        df = raw.iloc[1:].copy()
        df.columns = headers
        df = df.reset_index(drop=True)
        for col in df.columns:
            if _any_to_timestamp(col) is None:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if self.verbose:
            n_fechas = sum(1 for c in df.columns if _any_to_timestamp(c) is not None)
            print(f"      Resultados   : {len(df):,} filas | "
                  f"{n_fechas} columnas de fecha detectadas")
        return df

    def load_features_csv(self, path: str) -> pd.DataFrame:
        """
        Lee el CSV de features.
        Detecta automáticamente el separador (|, ;, ,, tab).
        """
        df = _read_csv_robust(path)
        df.columns = df.columns.str.strip()

        # Normalizar Loc y Sku
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ("loc", "location", "tienda", "store"):
                col_map[c] = "Loc"
            elif cl in ("sku", "articulo", "item", "codigo", "producto"):
                col_map[c] = "Sku"
        df = df.rename(columns=col_map)

        if "Loc" not in df.columns or "Sku" not in df.columns:
            raise ValueError(
                f"El CSV debe tener columnas Loc y Sku.\n"
                f"Columnas encontradas: {list(df.columns)}"
            )

        df["Loc"] = pd.to_numeric(df["Loc"], errors="coerce").astype(int)
        df["Sku"] = pd.to_numeric(df["Sku"], errors="coerce").astype(int)

        for c in _FEAT_NUM_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # Ignorar cluster si viene (se recalcula con K-Means)
        if "cluster" in df.columns:
            df = df.drop(columns=["cluster"])
            if self.verbose:
                print("      (columna 'cluster' ignorada — se recalculará)")

        if self.verbose:
            presentes = [c for c in _FEAT_NUM_COLS if c in df.columns]
            faltantes  = [c for c in _FEAT_NUM_COLS if c not in df.columns]
            print(f"      Features CSV : {len(df):,} pares Loc-SKU")
            print(f"        Presentes  : {presentes}")
            if faltantes:
                print(f"        ⚠ Faltantes : {faltantes}")
            # Muestra para verificar que se leyó bien
            if "media" in df.columns:
                print(f"        media[0:3]        : {df['media'].head(3).tolist()}")
            if "intermitencia" in df.columns:
                print(f"        intermitencia[0:3]: {df['intermitencia'].head(3).tolist()}")

        return df
