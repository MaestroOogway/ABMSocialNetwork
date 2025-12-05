# graph_loader.py
import networkx as nx
import random
from agents import Model, Susceptible, Skeptic, BOT, NewsReel
import matplotlib.pyplot as plt

def load_ego_graph(path, ego_id):
    # Load ego graph from SNAP format and reconnect ego node
    edges_file = f"{path}/{ego_id}.edges"
    G = nx.read_edgelist(edges_file)
    ego_str = str(ego_id)
    G.add_node(ego_str)
    for node in list(G.nodes()):
        if node != ego_str:
            G.add_edge(ego_str, node)
    return G

def build_model_from_graph(G, frac_susceptible=0.5, frac_skeptic=0.5, seed_types=None):
    # Create a model using graph nodes and assign human agent types
    model = Model(G)
    for n in G.nodes():
        node_id = str(n)
        if seed_types and node_id in seed_types:
            agent_type = seed_types[node_id]
        else:
            r = random.random()
            agent_type = Susceptible if r < frac_susceptible else Skeptic
        agent_type(model, node_id)
    return model

def add_random_bots(G, model, n_bots=10, edges_per_bot=3, bot_prefix="bot"):
    # Add bot nodes connected to random human nodes
    existing_nodes = [
        n for n in G.nodes() 
        if not str(n).startswith(bot_prefix) and not str(n).startswith("newsreel")
    ]
    if len(existing_nodes) == 0:
        raise ValueError("Graph has no valid nodes to attach bots to.")

    bot_ids = []
    for i in range(n_bots):
        bot_id = f"{bot_prefix}_{i}"
        G.add_node(bot_id)
        k = min(edges_per_bot, len(existing_nodes))
        neighbors = random.sample(existing_nodes, k)
        for nb in neighbors:
            G.add_edge(bot_id, str(nb))
        bot = BOT(model, id=bot_id)
        model.agents[bot_id] = bot
        bot_ids.append(bot_id)

    return bot_ids

def add_random_newsreels(G, model, n_reels=10, edges_per_reel=3, reel_prefix="newsreel"):
    # Add newsreel nodes connected to random human nodes
    existing_nodes = [
        n for n in G.nodes() 
        if not str(n).startswith("bot") and not str(n).startswith(reel_prefix)
    ]
    if len(existing_nodes) == 0:
        raise ValueError("Graph has no valid nodes to attach newsreels to.")

    reel_ids = []
    for i in range(n_reels):
        reel_id = f"{reel_prefix}_{i}"
        G.add_node(reel_id)
        k = min(edges_per_reel, len(existing_nodes))
        neighbors = random.sample(existing_nodes, k)
        for nb in neighbors:
            G.add_edge(reel_id, str(nb))
        reel = NewsReel(model, id=reel_id)
        model.agents[reel_id] = reel
        reel_ids.append(reel_id)

    return reel_ids

def draw_agent_network(G, model, save_path=None, show=False):
    # Classify nodes by agent type
    bot_nodes = []
    reel_nodes = []
    susceptible_nodes = []
    skeptic_nodes = []

    for node in G.nodes():
        try:
            agent = model.get_agent(str(node))
        except Exception:
            agent = None

        if agent is None:
            continue

        cls_name = agent.__class__.__name__
        if cls_name == "BOT":
            bot_nodes.append(node)
        elif cls_name == "NewsReel":
            reel_nodes.append(node)
        elif cls_name == "Susceptible":
            susceptible_nodes.append(node)
        elif cls_name == "Skeptic":
            skeptic_nodes.append(node)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.15)
    nx.draw_networkx_nodes(G, pos, nodelist=susceptible_nodes, node_color="skyblue", node_size=80)
    nx.draw_networkx_nodes(G, pos, nodelist=skeptic_nodes, node_color="orange", node_size=80)
    nx.draw_networkx_nodes(G, pos, nodelist=bot_nodes, node_color="red", node_size=150)
    nx.draw_networkx_nodes(G, pos, nodelist=reel_nodes, node_color="green", node_size=150)

    plt.legend(scatterpoints=1)
    plt.title("Agent Network Visualization")
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Graph saved as {save_path}")
    else:
        if show:
            plt.show()
        else:
            plt.close()

def remove_small_components(G, min_size=10):
    # Remove connected components smaller than min_size
    components = list(nx.connected_components(G))
    big_components = [c for c in components if len(c) >= min_size]
    nodes_to_keep = set().union(*big_components)
    G_clean = G.subgraph(nodes_to_keep).copy()

    print(f"Original nodes: {len(G.nodes())}")
    print(f"Nodes after cleaning: {len(G_clean.nodes())}")
    print(f"Removed {len(G.nodes()) - len(G_clean.nodes())} nodes from small components")

    return G_clean

# Example usage
G = load_ego_graph("facebook", 0)
G.remove_node("0")
G = remove_small_components(G, min_size=10)
