"""
clustering.py
=============
Clustering K-Means sobre las features del CSV.

Pipeline:
    1. Seleccionar features relevantes para la metodología
    2. Imputar NaN con mediana
    3. StandardScaler
    4. K-Means con K óptimo por índice de silueta (K ∈ 2..6)
       o K fijo si se especifica
    5. Etiquetar clusters como A/B/C/D según el perfil del centroide
    6. Añadir parámetros de metodología: alpha_cluster, factor_S, usar_p90

Features usadas para el clustering (subconjunto de tu CSV):
    media, cv, p90, intermitencia, acf1,
    ratio_demanda_vida, ratio_surtido, margen_relativo

Parámetros de metodología por cluster:
    A — Alta demanda estable   : alpha=0.5, S=V̂×1.05
    B — Alta demanda volátil   : alpha=2.0, S=V̂×1.20
    C — Demanda intermitente   : alpha=1.5, S=p90 (usar_p90=True)
    D — Tendencia / cíclica    : alpha=1.0, S=V̂×1.10
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


# ── Features que entran al K-Means ───────────────────────────────────────
# Solo las que tienen sentido para diferenciar perfiles de demanda
CLUSTER_FEATURES = [
    "media",            # nivel de demanda
    "cv",               # variabilidad relativa
    "p90",              # cola derecha de la distribución
    "intermitencia",    # frecuencia de semanas en cero
    "acf1",             # estructura temporal
    "ratio_demanda_vida",  # riesgo de merma
    "ratio_surtido",       # exceso por redondeo
    "margen_relativo",     # critical ratio Newsvendor
]

# ── Parámetros de metodología por perfil ─────────────────────────────────
CLUSTER_PROFILES = {
    "A": {
        "nombre"        : "Alta demanda estable",
        "alpha_cluster" : 0.5,
        "factor_S"      : 1.05,
        "usar_p90"      : False,
    },
    "B": {
        "nombre"        : "Alta demanda volátil",
        "alpha_cluster" : 2.0,
        "factor_S"      : 1.20,
        "usar_p90"      : False,
    },
    "C": {
        "nombre"        : "Demanda intermitente",
        "alpha_cluster" : 1.5,
        "factor_S"      : 1.0,   # no se usa (usar_p90=True)
        "usar_p90"      : True,
    },
    "D": {
        "nombre"        : "Tendencia / cíclica",
        "alpha_cluster" : 1.0,
        "factor_S"      : 1.10,
        "usar_p90"      : False,
    },
}

_DEFAULT_PARAMS = {"alpha_cluster": 1.0, "factor_S": 1.0, "usar_p90": False}


class ClusterEngine:
    def __init__(self, n_clusters: int = 0, verbose: bool = True):
        """
        n_clusters = 0  → K óptimo automático (silueta, K ∈ 2..6)
        n_clusters > 0  → K fijo
        """
        self.n_clusters = n_clusters
        self.verbose    = verbose
        self.k_opt      = None
        self.scaler     = None
        self.km         = None
        self.label_map  = {}    # cluster_id numérico → letra A/B/C/D

    # ── API principal ─────────────────────────────────────────────────────
    def fit_predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Corre el clustering sobre features_df y devuelve el mismo
        DataFrame con tres columnas nuevas:
            cluster        — letra A/B/C/D
            alpha_cluster  — parámetro α para Ec. (2)
            factor_S       — multiplicador de S para Ec. (5)
            usar_p90       — bandera para Cluster C
        """
        df = features_df.copy()

        # 1. Seleccionar features disponibles
        feat_cols = [c for c in CLUSTER_FEATURES if c in df.columns]
        if not feat_cols:
            raise ValueError(
                "El CSV no contiene ninguna de las features esperadas: "
                f"{CLUSTER_FEATURES}"
            )
        if self.verbose:
            falt = [c for c in CLUSTER_FEATURES if c not in df.columns]
            if falt:
                print(f"      ⚠ Features faltantes (no usadas): {falt}")
            print(f"      Features para clustering: {feat_cols}")

        X = df[feat_cols].copy()

        # 2. Imputar con mediana (robusto a outliers)
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], 0.0)

        # 3. Escalar
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 4. K óptimo
        if self.n_clusters > 0:
            self.k_opt = self.n_clusters
            if self.verbose:
                print(f"      K fijo: {self.k_opt}")
        else:
            self.k_opt = self._optimo_k(X_scaled)
            if self.verbose:
                print(f"      K óptimo (silueta): {self.k_opt}")

        # 5. K-Means
        self.km = KMeans(
            n_clusters=self.k_opt, n_init=20,
            max_iter=300, random_state=42
        )
        df["_cid"] = self.km.fit_predict(X_scaled)

        # 6. Calcular centroides en espacio original para etiquetar
        centroids_orig = (
            df.groupby("_cid")[feat_cols].mean()
        )

        # 7. Asignar letras A/B/C/D
        self.label_map = self._assign_labels(centroids_orig, feat_cols)
        df["cluster"]  = df["_cid"].map(self.label_map)

        # 8. Añadir parámetros de metodología
        df = self._add_params(df)
        df = df.drop(columns=["_cid"])

        return df

    def print_summary(self, features_df: pd.DataFrame):
        total = len(features_df)
        print()
        print("      " + "─" * 66)
        print(f"      {'Cluster':<10} {'Perfil':<26} "
              f"{'N pares':>9} {'%':>6} {'α':>5} {'factor S':>9}")
        print("      " + "─" * 66)
        for letra in sorted(features_df["cluster"].unique()):
            sub = features_df[features_df["cluster"] == letra]
            n   = len(sub)
            nom = CLUSTER_PROFILES.get(letra, {}).get("nombre", "Otro")
            alp = CLUSTER_PROFILES.get(letra, {}).get("alpha_cluster", "—")
            fs  = CLUSTER_PROFILES.get(letra, {}).get("factor_S", "—")
            fs_str = "p90" if CLUSTER_PROFILES.get(letra,{}).get("usar_p90") else str(fs)
            print(f"      {letra:<10} {nom:<26} "
                  f"{n:>9,} {100*n/total:>5.1f}% {alp:>5} {fs_str:>9}")
        print("      " + "─" * 66)

    # ── Internos ──────────────────────────────────────────────────────────
    def _optimo_k(self, X_scaled: np.ndarray) -> int:
        """Elige K ∈ {2..6} con mayor silueta promedio."""
        n   = min(len(X_scaled), 10_000)
        idx = np.random.RandomState(42).choice(len(X_scaled), n, replace=False)
        Xs  = X_scaled[idx]

        best_k, best_sil = 4, -1.0
        for k in range(2, 7):
            km  = KMeans(n_clusters=k, n_init=10, max_iter=200, random_state=42)
            lbl = km.fit_predict(Xs)
            try:
                sil = silhouette_score(Xs, lbl, sample_size=min(3_000, n))
            except Exception:
                sil = -1.0
            if self.verbose:
                print(f"        K={k}  silueta={sil:.4f}")
            if sil > best_sil:
                best_sil, best_k = sil, k
        return best_k

    @staticmethod
    def _assign_labels(centroids: pd.DataFrame,
                       feat_cols: list) -> dict:
        """
        Asigna letras A/B/C/D a los IDs numéricos usando las features
        de los centroides:

            C → mayor (intermitencia + cv − media)     demanda esporádica
            D → mayor acf1  (entre los que no son C)   tendencia / ciclos
            A → menor cv    (entre los restantes)       estable
            B → mayor cv    (entre los restantes)       volátil
        """
        ids = list(centroids.index)

        # ── C: intermitente ───────────────────────────────────────────────
        if "intermitencia" in feat_cols and "cv" in feat_cols and "media" in feat_cols:
            c_score = centroids["intermitencia"] + centroids["cv"] - centroids["media"]
        elif "intermitencia" in feat_cols:
            c_score = centroids["intermitencia"]
        else:
            c_score = pd.Series(0, index=ids)
        cid_C = c_score.idxmax()

        rest = [i for i in ids if i != cid_C]
        if not rest:
            return {cid_C: "C"}

        # ── D: tendencia/cíclica ──────────────────────────────────────────
        if "acf1" in feat_cols:
            cid_D = centroids.loc[rest, "acf1"].idxmax()
        else:
            cid_D = rest[0]

        rest2 = [i for i in rest if i != cid_D]
        if not rest2:
            return {cid_C: "C", cid_D: "D"}
        if len(rest2) == 1:
            return {rest2[0]: "A", cid_C: "C", cid_D: "D"}

        # ── A (estable, bajo cv) y B (volátil, alto cv) ───────────────────
        if "cv" in feat_cols:
            cid_A = centroids.loc[rest2, "cv"].idxmin()
        else:
            cid_A = rest2[0]
        cid_B = [i for i in rest2 if i != cid_A][0]

        label_map = {cid_A: "A", cid_B: "B", cid_C: "C", cid_D: "D"}

        # Clusters extra (si K > 4) → asignar al perfil más cercano
        for i in ids:
            if i not in label_map:
                dists = {
                    lbl: float(np.linalg.norm(
                        centroids.loc[i].values - centroids.loc[cid].values
                    ))
                    for cid, lbl in label_map.items()
                }
                label_map[i] = min(dists, key=dists.get)

        return label_map

    @staticmethod
    def _add_params(df: pd.DataFrame) -> pd.DataFrame:
        """Añade alpha_cluster, factor_S, usar_p90 según el cluster."""
        for col in ["alpha_cluster", "factor_S", "usar_p90"]:
            df[col] = df["cluster"].map(
                lambda c, col=col:
                CLUSTER_PROFILES.get(str(c).strip(), _DEFAULT_PARAMS)[col]
            )
        return df
