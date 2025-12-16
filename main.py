# main.py
import random, os, csv
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from loadegonetwork import (load_ego_graph, build_model_from_graph, add_random_bots, add_random_newsreels,)
from simulation import simulate_multi_news
from visualizer import SimulationVisualizer

def pedir_porcentajes():
    """Ask the user for the percentage of susceptible and skeptic agents."""
    print("\n=== CONFIGURACIÓN DE PORCENTAJES ===")
    while True:
        try:
            frac_s = float(input("Ingrese porcentaje de agentes SUSCEPTIBLES (0-100): ").strip()) / 100
            frac_k = float(input("Ingrese porcentaje de agentes ESCEPTICOS (0-100): ").strip()) / 100
            if frac_s < 0 or frac_k < 0 or abs((frac_s + frac_k) - 1.0) > 1e-9:
                print(" Error: los porcentajes deben sumar exactamente 100%. Intente nuevamente.\n")
                continue
            return frac_s, frac_k
        except:
            print(" Entrada inválida. Intente nuevamente.\n")


def pedir_visualizacion():
    """Ask whether real-time visualization should be activated."""
    print("\n=== CONFIGURACIÓN DE VISUALIZACIÓN ===")
    while True:
        respuesta = input("¿Desea visualización en tiempo real? (s/n): ").strip().lower()
        if respuesta in ["s", "si", "sí", "yes", "y"]:
            return True
        if respuesta in ["n", "no"]:
            return False
        print(" Por favor ingrese 's' o 'n'")


def pedir_numero_experimentos():
    """Ask for the number of experiments to run."""
    print("\n=== CONFIGURACIÓN DE EXPERIMENTOS ===")
    print("Opciones recomendadas:")
    print("  1 = Solo visualización (rápido, ~1 minuto)")
    print("  10 = Estadísticas básicas (~5-10 minutos)")
    print("  100 = Estadísticas completas (~30-60 minutos)")
    while True:
        try:
            n = int(input("\n¿Cuántos experimentos desea ejecutar? (1-100): ").strip())
            if 1 <= n <= 100:
                return n
            print(" Por favor ingrese un número entre 1 y 100")
        except:
            print(" Entrada inválida. Ingrese un número.")


def determinar_caso(frac_s, frac_k):
    """Determine scenario category based on input percentages."""
    if abs(frac_s - frac_k) < 1e-9:
        return "caso1"
    if frac_s > frac_k:
        return "caso2"
    return "caso3"


def run_single_experiment(G, run_id, frac_s, frac_k, log_dir, enable_viz=False):
    """Run a single simulation experiment."""
    model = build_model_from_graph(G, frac_susceptible=frac_s, frac_skeptic=frac_k)

    bot_ids = add_random_bots(G, model, n_bots=10, edges_per_bot=3, bot_prefix=f"bot{run_id}_")
    reel_ids = add_random_newsreels(G, model, n_reels=10, edges_per_reel=3, reel_prefix=f"reel{run_id}_")

    news_list, initial_seed_map = [], {}

    for bid in bot_ids:
        n = model.get_agent(bid).create_news()
        news_list.append(n)
        initial_seed_map[n.id] = [bid]

    for rid in reel_ids:
        n = model.get_agent(rid).create_news()
        news_list.append(n)
        initial_seed_map[n.id] = [rid]

    visualizer = SimulationVisualizer(G, model, news_list, max_iters=30) if enable_viz else None

    save_agg = f"{log_dir}/run_{run_id}_aggregated.csv"
    save_log = f"{log_dir}/run_{run_id}_detailed.csv"

    # <-- capture perception (5th returned value)
    metrics, aggregated, iters, detailed_logs, perception = simulate_multi_news(
        model,
        news_list,
        initial_seed_map,
        max_iters=30,
        save_aggregated_csv=save_agg,
        save_detailed_log=save_log,
        visualizer=visualizer
    )

    if visualizer:
        input("\n⏸️ Press Enter to continue...")
        visualizer.close()

    return aggregated, iters, perception


def plot_perception_stats(perception_stats, fig_dir, prefix="perception"):
    # Series
    sus_mean = np.array(perception_stats["susceptible"]["mean"], dtype=float)
    sus_std  = np.array(perception_stats["susceptible"]["std"], dtype=float)
    sk_mean  = np.array(perception_stats["skeptic"]["mean"], dtype=float)
    sk_std   = np.array(perception_stats["skeptic"]["std"], dtype=float)

    actual_iters = len(sus_mean)  # asumiendo mismas longitudes
    iters_range = np.arange(actual_iters)

    plt.figure(figsize=(10,5))
    plt.plot(iters_range, sus_mean, marker="o", label="Susceptible (media)")
    plt.fill_between(iters_range,
                     sus_mean - sus_std,
                     sus_mean + sus_std,
                     alpha=0.2)

    plt.plot(iters_range, sk_mean, marker="o", label="Skeptic (media)", color="tab:orange")
    plt.fill_between(iters_range,
                     sk_mean - sk_std,
                     sk_mean + sk_std,
                     alpha=0.2,
                     color="tab:orange")

    plt.title("Evolución de la percepción (media ± std)")
    plt.xlabel("Iteración")
    plt.ylabel("Percepción (−1 … 1)")
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlim(0, max(0, actual_iters-1))
    plt.legend()
    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}/{prefix}_sus_vs_skeptic.png", dpi=200)
    plt.close()


if __name__ == "__main__":

    SEED = 42

    # Input parameters
    frac_s, frac_k = pedir_porcentajes()
    caso = determinar_caso(frac_s, frac_k)
    enable_visualization = pedir_visualizacion()
    N_RUNS = pedir_numero_experimentos()

    # Output directories
    log_dir = f"logs/{caso}"
    fig_dir = f"figures/{caso}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\n➡ Configuración detectada: {frac_s*100:.0f}% susceptibles, {frac_k*100:.0f}% escépticos")
    print(f"➡ Número de experimentos: {N_RUNS}")
    print(f"➡ Visualización en tiempo real: {'ACTIVADA ✅' if enable_visualization else 'DESACTIVADA'}")
    print(f"➡ Guardando resultados en: logs/{caso}/ y figures/{caso}/")

    if enable_visualization and N_RUNS > 1:
        print("\n💡 Only the first experiment will display visualization.\n")
    elif enable_visualization and N_RUNS == 1:
        print("\n✨ Visualization enabled for a single experiment.\n")

    # Load or generate graph
    try:
        G = load_ego_graph("facebook", 0)
    except:
        G = nx.erdos_renyi_graph(300, 0.02)
        G = nx.relabel_nodes(G, lambda x: str(x))

    runs_true_exposed, runs_false_exposed = [], []
    runs_true_shared, runs_false_shared = [], []
    runs_perception = []  # <-- nuevo: almacenar perception por run

    # Run experiments
    for run in range(N_RUNS):
        run_seed = SEED + run
        random.seed(run_seed)
        np.random.seed(run_seed)
        print(f"Running experiment {run+1}/{N_RUNS}")
        show_viz = enable_visualization and (run == 0)
        aggregated, iters, perception = run_single_experiment(G, run, frac_s, frac_k, log_dir, enable_viz=show_viz)

        runs_true_exposed.append(aggregated[True]["exposed"])
        runs_false_exposed.append(aggregated[False]["exposed"])
        runs_true_shared.append(aggregated[True]["shared"])
        runs_false_shared.append(aggregated[False]["shared"])
        runs_perception.append(perception)  # <-- guardar perception

    # Normalize lengths for shared/exposed
    max_len = max(len(s) for s in runs_true_exposed)
    def pad(series_list): return [s + [0] * (max_len - len(s)) for s in series_list]

    runs_true_exposed = pad(runs_true_exposed)
    runs_false_exposed = pad(runs_false_exposed)
    runs_true_shared = pad(runs_true_shared)
    runs_false_shared = pad(runs_false_shared)

    A_true_exp = np.array(runs_true_exposed)
    A_false_exp = np.array(runs_false_exposed)
    A_true_sh = np.array(runs_true_shared)
    A_false_sh = np.array(runs_false_shared)

    mean_true_exp, mean_false_exp = A_true_exp.mean(axis=0), A_false_exp.mean(axis=0)
    std_true_exp, std_false_exp = A_true_exp.std(axis=0), A_false_exp.std(axis=0)
    mean_true_sh, mean_false_sh = A_true_sh.mean(axis=0), A_false_sh.mean(axis=0)
    std_true_sh, std_false_sh = A_true_sh.std(axis=0), A_false_sh.std(axis=0)

    iters_range = np.arange(max_len)

    plt.figure(figsize=(10, 5))
    plt.plot(iters_range, mean_true_sh, marker="o", label="True Shared")
    plt.fill_between(iters_range, mean_true_sh - std_true_sh, mean_true_sh + std_true_sh, alpha=0.2)

    plt.plot(iters_range, mean_false_sh, marker="o", label="False Shared")
    plt.fill_between(iters_range, mean_false_sh - std_false_sh, mean_false_sh + std_false_sh, alpha=0.2)

    plt.title("Cantidad de noticias compartidas (media ± std)")
    plt.xlabel("Iteración")
    plt.ylabel("Agentes que compartieron")

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlim(min(iters_range), max(iters_range))

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/line_shared.png")
    plt.close()


    plt.figure(figsize=(10, 5))
    plt.plot(iters_range, mean_true_exp, marker="o", label="True Exposed")
    plt.fill_between(iters_range, mean_true_exp - std_true_exp, mean_true_exp + std_true_exp, alpha=0.2)

    plt.plot(iters_range, mean_false_exp, marker="o", label="False Exposed")
    plt.fill_between(iters_range, mean_false_exp - std_false_exp, mean_false_exp + std_false_exp, alpha=0.2)

    plt.title("Cantidad de noticias expuestas (media ± std)")
    plt.xlabel("Iteración")
    plt.ylabel("Agentes expuestos")

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlim(min(iters_range), max(iters_range))

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/line_exposed.png")
    plt.close()


    # Plot: Box Shared
    plt.figure(figsize=(10, 5))
    plt.boxplot([A_true_sh.flatten(), A_false_sh.flatten()], labels=["True Shared", "False Shared"])
    plt.title("Distribución de noticias compartidas")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/box_shared.png")
    plt.close()

    # Plot: Box Exposed
    plt.figure(figsize=(10, 5))
    plt.boxplot([A_true_exp.flatten(), A_false_exp.flatten()], labels=["True Exposed", "False Exposed"])
    plt.title("Distribución de noticias expuestas")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/box_exposed.png")
    plt.close()

    # --- NEW: aggregate perception across runs ---
    if len(runs_perception) > 0:
        # determinar longitud máxima (iteraciones)
        max_p_len = max(len(p["susceptible"]["mean"]) for p in runs_perception)

        # construir matrices padded con np.nan
        sus_means = []
        sk_means = []
        for p in runs_perception:
            sus = p["susceptible"]["mean"]
            sk = p["skeptic"]["mean"]
            sus_padded = sus + [np.nan] * (max_p_len - len(sus))
            sk_padded = sk + [np.nan] * (max_p_len - len(sk))
            sus_means.append(sus_padded)
            sk_means.append(sk_padded)

        M_sus = np.array(sus_means, dtype=float)  # shape (n_runs, max_p_len)
        M_sk  = np.array(sk_means, dtype=float)

        sus_mean_across_runs = np.nanmean(M_sus, axis=0)
        sus_std_across_runs  = np.nanstd(M_sus, axis=0)
        sk_mean_across_runs  = np.nanmean(M_sk, axis=0)
        sk_std_across_runs   = np.nanstd(M_sk, axis=0)

        perception_stats = {
            "susceptible": {"mean": sus_mean_across_runs.tolist(), "std": sus_std_across_runs.tolist()},
            "skeptic": {"mean": sk_mean_across_runs.tolist(), "std": sk_std_across_runs.tolist()}
        }

        # Graficar percepción
        plot_perception_stats(perception_stats, fig_dir)
    else:
        print("No hay datos de percepción para graficar.")

    print(f"\nExperimentos completados. Gráficos guardados en {fig_dir}/\n")
