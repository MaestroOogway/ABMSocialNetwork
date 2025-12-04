# main.py
import random, os, csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from loadegonetwork import (
    load_ego_graph,
    build_model_from_graph,
    add_random_bots,
    add_random_newsreels,
)
from simulation import simulate_multi_news


def pedir_porcentajes():
    print("\n=== CONFIGURACIÓN DE PORCENTAJES ===")
    while True:
        try:
            frac_s = float(input("Ingrese porcentaje de agentes SUSCEPTIBLES (0-100): ").strip()) / 100
            frac_k = float(input("Ingrese porcentaje de agentes ESCEPTICOS (0-100): ").strip()) / 100

            if frac_s < 0 or frac_k < 0 or frac_s + frac_k != 1:
                print("❌ Error: los porcentajes deben sumar exactamente 100%. Intente nuevamente.\n")
                continue
            return frac_s, frac_k
        except:
            print("❌ Entrada inválida. Intente nuevamente.\n")


def determinar_caso(frac_s, frac_k):
    if frac_s == frac_k:
        return "caso1"
    elif frac_s > frac_k:
        return "caso2"
    else:
        return "caso3"


def run_single_experiment(G, run_id, frac_s, frac_k):
    """Ejecuta un experimento completo y devuelve las series agregadas."""
    model = build_model_from_graph(G, frac_susceptible=frac_s, frac_skeptic=frac_k)

    bot_ids = add_random_bots(G, model, n_bots=10, edges_per_bot=3, bot_prefix=f"bot{run_id}_")
    reel_ids = add_random_newsreels(G, model, n_reels=10, edges_per_reel=3, reel_prefix=f"reel{run_id}_")

    # crear news
    news_list = []
    initial_seed_map = {}

    for bid in bot_ids:
        n = model.get_agent(bid).create_news()
        news_list.append(n)
        initial_seed_map[n.id] = [bid]

    for rid in reel_ids:
        n = model.get_agent(rid).create_news()
        news_list.append(n)
        initial_seed_map[n.id] = [rid]

    metrics, aggregated, actual_iters, detailed_logs = simulate_multi_news(
        model, news_list, initial_seed_map, max_iters=30
    )

    return aggregated, actual_iters


if __name__ == "__main__":
    # =============================================================
    #   1) PEDIR PORCENTAJES AL USUARIO
    # =============================================================
    frac_s, frac_k = pedir_porcentajes()
    caso = determinar_caso(frac_s, frac_k)

    # =============================================================
    #   2) CREAR DIRECTORIOS AUTOMÁTICOS DEPENDIENDO DEL CASO
    # =============================================================
    log_dir = f"logs/{caso}"
    fig_dir = f"figures/{caso}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\n➡ Configuración detectada: {frac_s*100:.0f}% susceptibles, {frac_k*100:.0f}% escépticos")
    print(f"➡ Guardando resultados en: logs/{caso}/ y figures/{caso}/\n")

    # =============================================================
    #   3) CARGAR O GENERAR RED
    # =============================================================
    try:
        G = load_ego_graph("facebook", 0)
    except:
        G = nx.erdos_renyi_graph(300, 0.02)
        G = nx.relabel_nodes(G, lambda x: str(x))

    N_RUNS = 100

    # almacenar series de cada corrida
    runs_true_exposed = []
    runs_false_exposed = []
    runs_true_shared = []
    runs_false_shared = []

    # =============================================================
    #   4) EJECUTAR LOS EXPERIMENTOS
    # =============================================================
    for run in range(N_RUNS):
        print(f"Running experiment {run+1}/{N_RUNS}")
        aggregated, iters = run_single_experiment(G, run, frac_s, frac_k)

        runs_true_exposed.append(aggregated[True]["exposed"])
        runs_false_exposed.append(aggregated[False]["exposed"])
        runs_true_shared.append(aggregated[True]["shared"])
        runs_false_shared.append(aggregated[False]["shared"])

    # igualar tamaños
    max_len = max(len(s) for s in runs_true_exposed)

    def pad(series_list):
        return [s + [0] * (max_len - len(s)) for s in series_list]

    runs_true_exposed = pad(runs_true_exposed)
    runs_false_exposed = pad(runs_false_exposed)
    runs_true_shared = pad(runs_true_shared)
    runs_false_shared = pad(runs_false_shared)

    # convertir a array
    A_true_exp = np.array(runs_true_exposed)
    A_false_exp = np.array(runs_false_exposed)
    A_true_sh = np.array(runs_true_shared)
    A_false_sh = np.array(runs_false_shared)

    # estadísticas
    mean_true_exp = A_true_exp.mean(axis=0)
    mean_false_exp = A_false_exp.mean(axis=0)
    std_true_exp = A_true_exp.std(axis=0)
    std_false_exp = A_false_exp.std(axis=0)

    mean_true_sh = A_true_sh.mean(axis=0)
    mean_false_sh = A_false_sh.mean(axis=0)
    std_true_sh = A_true_sh.std(axis=0)
    std_false_sh = A_false_sh.std(axis=0)

    iters_range = np.arange(max_len)

    # =============================================================
    #   5) GUARDAR GRÁFICOS EN figure/casoX/
    # =============================================================

    # --- GRAFICO 1: SHARED ---
    plt.figure(figsize=(10, 5))
    plt.plot(iters_range, mean_true_sh, marker="o", label="True Shared")
    plt.fill_between(iters_range, mean_true_sh - std_true_sh, mean_true_sh + std_true_sh, alpha=0.2)
    plt.plot(iters_range, mean_false_sh, marker="o", label="False Shared", color="red")
    plt.fill_between(iters_range, mean_false_sh - std_false_sh, mean_false_sh + std_false_sh, alpha=0.2, color="red")
    plt.title("Cantidad de noticias compartidas (media ± std) - 100 corridas")
    plt.xlabel("Iteración"); plt.ylabel("Agentes que compartieron")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{fig_dir}/line_shared.png")
    plt.close()

    # --- GRAFICO 2: EXPOSED ---
    plt.figure(figsize=(10, 5))
    plt.plot(iters_range, mean_true_exp, marker="o", label="True Exposed")
    plt.fill_between(iters_range, mean_true_exp - std_true_exp, mean_true_exp + std_true_exp, alpha=0.2)
    plt.plot(iters_range, mean_false_exp, marker="o", label="False Exposed", color="red")
    plt.fill_between(iters_range, mean_false_exp - std_false_exp, mean_false_exp + std_false_exp, alpha=0.2, color="red")
    plt.title("Cantidad de noticias expuestas (media ± std) - 100 corridas")
    plt.xlabel("Iteración"); plt.ylabel("Agentes expuestos")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{fig_dir}/line_exposed.png")
    plt.close()

    # --- GRAFICO 3: BOX SHARED ---
    plt.figure(figsize=(10, 5))
    plt.boxplot([A_true_sh.flatten(), A_false_sh.flatten()],
                labels=["True Shared", "False Shared"])
    plt.title("Distribución de noticias compartidas (100 corridas)")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/box_shared.png")
    plt.close()

    # --- GRAFICO 4: BOX EXPOSED ---
    plt.figure(figsize=(10, 5))
    plt.boxplot([A_true_exp.flatten(), A_false_exp.flatten()],
                labels=["True Exposed", "False Exposed"])
    plt.title("Distribución de noticias expuestas (100 corridas)")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/box_exposed.png")
    plt.close()

    print(f"\nExperimentos completados. Gráficos guardados en {fig_dir}/\n")
