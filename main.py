"""
╔══════════════════════════════════════════════════════════════════════════╗
║          PROYECTO ABASTO — Solver con Clustering                        ║
║          NEXO MX · CIMAT · Febrero 2026                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Salidas:                                                                ║
║    resultados_abasto.csv   — INV_IDEAL para 1M pares × 12 semanas       ║
║    resumen_abasto.xlsx     — clusters, features y simulación             ║
╚══════════════════════════════════════════════════════════════════════════╝

Uso:
    python main.py --excel datos.xlsx --features features.csv
    python main.py --excel datos.xlsx --features features.csv --n-clusters 4
"""

import argparse
import time
from src.data_loader import DataLoader
from src.clustering  import ClusterEngine
from src.forecaster  import Forecaster
from src.reabasto    import ReabastoEngine
from src.evaluator   import Evaluator
from src.writer      import ResultWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel",      required=True)
    parser.add_argument("--features",   required=True)
    parser.add_argument("--out-csv",    default="resultados_abasto.csv")
    parser.add_argument("--out-xlsx",   default="resumen_abasto.xlsx")
    parser.add_argument("--n-clusters", type=int, default=0)
    parser.add_argument("--verbose",    action="store_true", default=True)
    args = parser.parse_args()

    SEP = "═" * 68
    t0  = time.time()
    print(f"\n{SEP}")
    print("  PROYECTO ABASTO — Pipeline")
    print(f"{SEP}\n")

    # 1. Lectura
    print("[1/5] Leyendo datos...")
    t1 = time.time()
    loader = DataLoader(verbose=args.verbose)
    inv_df, cat_df, res_df = loader.load_excel(args.excel)
    features_df            = loader.load_features_csv(args.features)
    print(f"      ({time.time()-t1:.1f}s)")

    # 2. Clustering
    print("\n[2/5] Ejecutando clustering...")
    t1 = time.time()
    ce = ClusterEngine(n_clusters=args.n_clusters, verbose=args.verbose)
    features_df = ce.fit_predict(features_df)
    ce.print_summary(features_df)
    print(f"      ({time.time()-t1:.1f}s)")

    # 3. INV_IDEAL
    print("\n[3/5] Calculando INV_IDEAL (12 semanas)...")
    t1 = time.time()
    forecaster   = Forecaster(verbose=args.verbose)
    re           = ReabastoEngine(verbose=args.verbose)
    resultado_df, fechas_target = re.run(
        inv_df      = inv_df,
        cat_df      = cat_df,
        res_df      = res_df,
        features_df = features_df,
        forecaster  = forecaster,
    )
    print(f"      ({time.time()-t1:.1f}s)")

    # 4. Simulación
    print("\n[4/5] Simulando utilidad...")
    t1 = time.time()
    ev        = Evaluator(verbose=args.verbose)
    sim_stats = ev.simulate(
        inv_df        = inv_df,
        cat_df        = cat_df,
        features_df   = features_df,
        resultado_df  = resultado_df,
        fechas_target = fechas_target,
    )
    print(f"      ({time.time()-t1:.1f}s)")

    # 5. Escritura
    print(f"\n[5/5] Escribiendo resultados...")
    t1 = time.time()
    writer = ResultWriter(verbose=args.verbose)
    writer.write(
        resultado_df  = resultado_df,
        features_df   = features_df,
        fechas_target = fechas_target,
        sim_stats     = sim_stats,
        path_csv      = args.out_csv,
        path_xlsx     = args.out_xlsx,
    )
    print(f"      ({time.time()-t1:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n{SEP}")
    print("  RESUMEN FINAL")
    print(f"{SEP}")
    print(f"  Pares Loc-SKU      : {len(resultado_df):,}")
    print(f"  Semanas            : {len(fechas_target)}")
    print(f"  Total celdas CSV   : {len(resultado_df)*len(fechas_target):,}")
    print(f"  Clusters           : {features_df['cluster'].nunique()}")
    for k, g in features_df.groupby("cluster"):
        print(f"    {k} — {len(g):,} pares")
    print(f"  Utilidad Base      : ${sim_stats['util_base']:>14,.2f}")
    print(f"  Utilidad Propuesta : ${sim_stats['util_prop']:>14,.2f}")
    print(f"  Mejora estimada    : {sim_stats['mejora_pct']:+.1f}%")
    print(f"  CSV generado       : {args.out_csv}")
    print(f"  Excel resumen      : {args.out_xlsx}")
    print(f"  Tiempo total       : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
