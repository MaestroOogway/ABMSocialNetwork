# visualizer.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Patch
from agents import Susceptible, Skeptic, BOT, NewsReel


class SimulationVisualizer:
    """Visualizador en tiempo real para la simulación ABM de noticias."""

    def __init__(self, G, model, news_list, max_iters=30):
        """
        Inicializa el visualizador.

        Args:
            G: Grafo de NetworkX
            model: Instancia del modelo
            news_list: Lista de noticias en la simulación
            max_iters: Número máximo de iteraciones
        """
        self.G = G
        self.model = model
        self.news_list = news_list
        self.max_iters = max_iters

        # Datos para las series temporales
        self.iterations = []
        self.true_shared_history = []
        self.false_shared_history = []
        self.true_exposed_history = []
        self.false_exposed_history = []
        self.susceptible_count_history = []
        self.skeptic_count_history = []

        # Configurar la figura con subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Subplot 1: Red (ocupa 2 filas)
        self.ax_network = self.fig.add_subplot(self.gs[:2, 0])

        # Subplot 2: Métricas de compartición
        self.ax_shared = self.fig.add_subplot(self.gs[0, 1])

        # Subplot 3: Métricas de exposición
        self.ax_exposed = self.fig.add_subplot(self.gs[1, 1])

        # Subplot 4: Distribución de agentes
        self.ax_agents = self.fig.add_subplot(self.gs[2, :])

        # Calcular layout de la red (solo una vez para mantener posiciones)
        self.pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        # Activar modo interactivo
        plt.ion()
        plt.show()

    def update(self, iteration, frontiers, aggregated, news_by_id):
        """
        Actualiza todos los gráficos con los datos de la iteración actual.

        Args:
            iteration: Número de iteración actual
            frontiers: Diccionario {news_id: set(agent_ids)} de agentes activos
            aggregated: Diccionario con métricas agregadas por veracidad
            news_by_id: Diccionario {news_id: News}
        """
        # Actualizar series temporales (con acceso seguro a los índices)
        self.iterations.append(iteration)

        # Obtener valores de manera segura
        true_shared = aggregated[True]["shared"][iteration] if iteration < len(aggregated[True]["shared"]) else 0
        false_shared = aggregated[False]["shared"][iteration] if iteration < len(aggregated[False]["shared"]) else 0
        true_exposed = aggregated[True]["exposed"][iteration] if iteration < len(aggregated[True]["exposed"]) else 0
        false_exposed = aggregated[False]["exposed"][iteration] if iteration < len(aggregated[False]["exposed"]) else 0

        self.true_shared_history.append(true_shared)
        self.false_shared_history.append(false_shared)
        self.true_exposed_history.append(true_exposed)
        self.false_exposed_history.append(false_exposed)

        # Contar tipos de agentes
        susceptible_count = sum(1 for a in self.model.agents.values() if isinstance(a, Susceptible) and not isinstance(a, BOT))
        skeptic_count = sum(1 for a in self.model.agents.values() if isinstance(a, Skeptic))
        self.susceptible_count_history.append(susceptible_count)
        self.skeptic_count_history.append(skeptic_count)

        # Actualizar cada subplot
        self._update_network(frontiers, news_by_id)
        self._update_shared_metrics()
        self._update_exposed_metrics()
        self._update_agent_distribution()

        # Título general con información de actividad
        active_count = sum(len(sharers) for sharers in frontiers.values())
        title = f"Simulación ABM - Iteración {iteration}/{self.max_iters}"
        if active_count == 0:
            title += " ⏸️ (Sin actividad - Propagación completada)"
        else:
            title += f" ✅ ({active_count} agentes compartiendo)"
        self.fig.suptitle(title, fontsize=16, fontweight="bold")

        # Refrescar display
        plt.pause(0.1)

    def _update_network(self, frontiers, news_by_id):
        """Actualiza la visualización de la red."""
        self.ax_network.clear()

        # Determinar el color de cada nodo según su estado
        node_colors = []
        node_sizes = []

        # Obtener todos los agentes que están compartiendo activamente
        active_sharers = set()
        for news_id, sharers in frontiers.items():
            active_sharers.update(sharers)

        for node in self.G.nodes():
            node_str = str(node)
            agent = self.model.get_agent(node_str)

            # Color base según tipo de agente
            if isinstance(agent, BOT):
                color = "#FF0000"  # Rojo para bots
                size = 300
            elif isinstance(agent, NewsReel):
                color = "#00FF00"  # Verde para news reels
                size = 300
            elif isinstance(agent, Skeptic):
                color = "#FFA500"  # Naranja para escépticos
                size = 100
            elif isinstance(agent, Susceptible):
                color = "#1E90FF"  # Azul para susceptibles
                size = 100
            else:
                color = "#808080"  # Gris por defecto
                size = 100

            # Si el agente está compartiendo activamente, hacerlo más grande y brillante
            if node_str in active_sharers:
                size *= 2
                # Agregar brillo (más saturación)
                if color == "#1E90FF":
                    color = "#00BFFF"  # Azul más brillante
                elif color == "#FFA500":
                    color = "#FF8C00"  # Naranja más brillante

            node_colors.append(color)
            node_sizes.append(size)

        # Dibujar la red
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors, node_size=node_sizes, alpha=0.7, ax=self.ax_network)

        nx.draw_networkx_edges(self.G, self.pos, alpha=0.2, width=0.5, ax=self.ax_network)

        # Leyenda
        legend_elements = [
            Patch(facecolor="#FF0000", label="Bot (noticias falsas)"),
            Patch(facecolor="#00FF00", label="NewsReel (noticias verdaderas)"),
            Patch(facecolor="#1E90FF", label="Susceptible"),
            Patch(facecolor="#FFA500", label="Escéptico"),
            Patch(facecolor="#00BFFF", edgecolor="black", label="Compartiendo activamente"),
        ]
        self.ax_network.legend(handles=legend_elements, loc="upper left", fontsize=8)

        self.ax_network.set_title("Red de Agentes", fontsize=12, fontweight="bold")
        self.ax_network.axis("off")

    def _update_shared_metrics(self):
        """Actualiza el gráfico de métricas de compartición."""
        self.ax_shared.clear()

        if len(self.iterations) > 0:
            self.ax_shared.plot(self.iterations, self.true_shared_history, "g-o", label="Noticias verdaderas", linewidth=2, markersize=4)
            self.ax_shared.plot(self.iterations, self.false_shared_history, "r-o", label="Noticias falsas", linewidth=2, markersize=4)

            self.ax_shared.set_xlabel("Iteración", fontsize=10)
            self.ax_shared.set_ylabel("Agentes que compartieron", fontsize=10)
            self.ax_shared.set_title("Compartición de Noticias", fontsize=12, fontweight="bold")
            self.ax_shared.legend(loc="best", fontsize=9)
            self.ax_shared.grid(True, alpha=0.3)

    def _update_exposed_metrics(self):
        """Actualiza el gráfico de métricas de exposición."""
        self.ax_exposed.clear()

        if len(self.iterations) > 0:
            self.ax_exposed.plot(self.iterations, self.true_exposed_history, "g-s", label="Noticias verdaderas", linewidth=2, markersize=4)
            self.ax_exposed.plot(self.iterations, self.false_exposed_history, "r-s", label="Noticias falsas", linewidth=2, markersize=4)

            self.ax_exposed.set_xlabel("Iteración", fontsize=10)
            self.ax_exposed.set_ylabel("Agentes expuestos", fontsize=10)
            self.ax_exposed.set_title("Exposición a Noticias", fontsize=12, fontweight="bold")
            self.ax_exposed.legend(loc="best", fontsize=9)
            self.ax_exposed.grid(True, alpha=0.3)

    def _update_agent_distribution(self):
        """Actualiza el gráfico de distribución de agentes."""
        self.ax_agents.clear()

        if len(self.iterations) > 0:
            width = 0.35
            x = np.arange(len(self.iterations))

            self.ax_agents.bar(x - width / 2, self.susceptible_count_history, width, label="Susceptibles", color="#1E90FF", alpha=0.8)
            self.ax_agents.bar(x + width / 2, self.skeptic_count_history, width, label="Escépticos", color="#FFA500", alpha=0.8)

            self.ax_agents.set_xlabel("Iteración", fontsize=10)
            self.ax_agents.set_ylabel("Cantidad de agentes", fontsize=10)
            self.ax_agents.set_title("Evolución de Tipos de Agentes", fontsize=12, fontweight="bold")
            self.ax_agents.legend(loc="best", fontsize=9)
            self.ax_agents.grid(True, alpha=0.3, axis="y")

    def close(self):
        """Cierra la visualización y desactiva el modo interactivo."""
        plt.ioff()
        plt.close(self.fig)

    def save_final(self, filepath):
        """Guarda la visualización final en un archivo."""
        self.fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Visualización guardada en: {filepath}")
