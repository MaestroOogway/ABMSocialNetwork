# main.py
import random
import networkx as nx
import matplotlib.pyplot as plt
import os
import csv
from loadegonetwork import (
    load_ego_graph,
    build_model_from_graph,
    add_random_bots,
    add_random_newsreels,
)
from simulation import simulate_multi_news
from loadegonetwork import draw_agent_network  # updated version that does not show plots when saving

if __name__ == "__main__":
    # prepare folders
    os.makedirs("logs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # load or generate graph
    try:
        path = "facebook"
        ego_id = 0
        G = load_ego_graph(path, ego_id)
        print(f"Graph loaded from '{path}' with ego {ego_id}")
    except Exception:
        print("Edges file not found. Generating random graph.")
        G = nx.erdos_renyi_graph(300, 0.02)
        G = nx.relabel_nodes(G, lambda x: str(x))

    model = build_model_from_graph(G, frac_susceptible=0.6, frac_skeptic=0.4)

    # add bots and newsreels
    bot_ids = add_random_bots(G, model, n_bots=10, edges_per_bot=3, bot_prefix="bot")
    reel_ids = add_random_newsreels(G, model, n_reels=10, edges_per_reel=3, reel_prefix="newsreel")
    print("Added bots:", bot_ids)
    print("Added newsreels:", reel_ids)

    # create news per generator
    news_list = []
    initial_seed_map = {}
    for bid in bot_ids:
        bot = model.get_agent(bid)
        n = bot.create_news()
        news_list.append(n)
        initial_seed_map[n.id] = [bid]
    for rid in reel_ids:
        reel = model.get_agent(rid)
        n = reel.create_news()
        news_list.append(n)
        initial_seed_map[n.id] = [rid]

    print("Created news items:", [(n.id, n.veracity, n.polarity, n.credibility) for n in news_list])

    # run simulation and save aggregated CSV + detailed propagation log
    aggregated_csv_path = os.path.join("logs", "aggregated_results.csv")
    detailed_log_path = os.path.join("logs", "detailed_propagation_log.csv")

    metrics, aggregated, actual_iters, detailed_logs = simulate_multi_news(
        model,
        news_list,
        initial_seed_map,
        max_iters=30,
        save_aggregated_csv=aggregated_csv_path,
        save_detailed_log=detailed_log_path
    )

    # extract time series
    it = list(range(actual_iters))
    true_shared = aggregated[True]['shared']
    false_shared = aggregated[False]['shared']
    true_exposed = aggregated[True]['exposed']
    false_exposed = aggregated[False]['exposed']

    # PLOT 1: shared per iteration (true vs false)
    plt.figure()
    plt.plot(it, true_shared, label="true_shared")
    plt.plot(it, false_shared, label="false_shared")
    plt.xlabel("iteration")
    plt.ylabel("shared count")
    plt.title("Shared per iteration: True vs False")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "shared_true_vs_false.png"))
    plt.close()  # do not show

    # PLOT 2: propagation velocity (exposed per iteration) true vs false
    plt.figure()
    plt.plot(it, true_exposed, label="true_exposed")
    plt.plot(it, false_exposed, label="false_exposed")
    plt.xlabel("iteration")
    plt.ylabel("exposed count")
    plt.title("Propagation velocity (exposed) True vs False")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "velocity_true_vs_false.png"))
    plt.close()  # do not show

    # save per-news metrics optionally (one CSV per news)
    for nid, m in metrics.items():
        fname = os.path.join("logs", f"news_{nid}_metrics.csv")
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "new_exposed", "new_shared"])
            for i in range(len(m['new_exposed'])):
                writer.writerow([i, m['new_exposed'][i], m['new_shared'][i]])

    # draw network to figures (no interactive show)
    draw_agent_network(G, model, save_path=os.path.join("figures", "network_visualization.png"))

    print("Saved aggregated CSV in logs/, detailed logs in logs/, and figures in figures/")
