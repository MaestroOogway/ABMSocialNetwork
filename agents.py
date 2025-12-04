import random
import math
from typing import List, Optional, Iterable

PARTY = ["N"]
POLARITYNEWS = [-1, 1]
VERACITYNEWS = [False, True]
PHI = 0.6
ALPHA = 0.1

THRESHOLD_TO_SKEPTIC = -0.3
THRESHOLD_TO_SUSCEPTIBLE = 0.3

DEFAULT_wP = 0.45  # peso sobre el interés del agente
DEFAULT_wF = 0.30  # peso sobre la credibilidad de la noticia
DEFAULT_wC = 0.25  # peso sobre la credibilidad del agente

def clamp(x, lo, hi): return max(lo, min(hi, x))
def roundto(x, ndigits=3): return round(x, ndigits)

class News:
    count = 0

    def __init__(self, id=None,
                 party=None,
                 polarity=None,
                 veracity=None,
                 credibility=None,
                 source: Optional[str] = None,
                 topics: Optional[List[str]] = None,
                 salience: Optional[float] = None,
                 relevance: Optional[float] = None):

        if id is None:
            id = News.count
        self.id = id
        News.count += 1

        self.party = party if party is not None else random.choice(PARTY)
        self.polarity = polarity if polarity is not None else random.choice(POLARITYNEWS)
        self.veracity = veracity if veracity is not None else random.choice(VERACITYNEWS)

        if credibility is None:
            self.credibility = random.uniform(0.7, 0.9) if self.veracity else random.uniform(0.1, 0.35)
        else:
            self.credibility = credibility

        self.source = source if source is not None else "generic"
        self.topics = topics if topics is not None else []
        # salience en [0,1]
        self.salience = salience if salience is not None else random.random()
        self.relevance = relevance

    def __repr__(self):
        return (f"News(id={self.id}, party={self.party}, pol={self.polarity}, "
                f"veracity={self.veracity}, cred={self.credibility:.2f}, "
                f"src={self.source}, sal={self.salience:.2f})")


# MODEL (cola, registro de agentes, noticias)
class Model:
    def __init__(self, G):
        self.G = G
        self.agents = {}
        self.news_map = {}
        self.exposure_queue = {}  # news_id -> set(dest_ids)
        self.news_propagation = []
        self.converted_agents = []
        self.conversions_to_skeptic = 0
        self.conversions_to_susceptible = 0

    def register_agent(self, agent):
        self.agents[agent.id] = agent
        agent.model = self

    def get_agent(self, node_id):
        return self.agents[node_id]

    def register_news(self, news: News):
        if news.id not in self.news_map:
            self.news_map[news.id] = news

    def get_news(self, news_id):
        return self.news_map[news_id]

    def queue_exposure(self, news_or_id, dest_id_or_iterable):
        if hasattr(news_or_id, "id"):
            nid = news_or_id.id
            self.register_news(news_or_id)
        else:
            nid = news_or_id

        if isinstance(dest_id_or_iterable, (list, set, tuple)):
            dests = dest_id_or_iterable
        else:
            dests = [dest_id_or_iterable]

        self.exposure_queue.setdefault(nid, set())
        for d in dests:
            self.exposure_queue[nid].add(d)

    def process_queued_exposures(self):
        queued = self.exposure_queue
        self.exposure_queue = {}

        new_exposed = 0
        for nid, dests in queued.items():
            if nid not in self.news_map:
                continue
            news = self.get_news(nid)
            for dest in list(dests):
                try:
                    agent = self.get_agent(dest)
                except KeyError:
                    continue

                first_time = agent.on_exposure(news)
                if first_time:
                    new_exposed += 1
        return new_exposed

    def run_propagation(self, max_rounds=100, stop_if_no_new=True):
        round_no = 0
        history = []
        while round_no < max_rounds:
            round_no += 1
            new = self.process_queued_exposures()
            history.append(new)
            self.news_propagation.append({"round": round_no, "new_exposed": new})
            if stop_if_no_new and new == 0:
                break
        return {"rounds": round_no, "new_exposed_history": history, "total_new": sum(history)}


# USER (clase base)
class User:
    def __init__(self, model: Model, id,
                 party: Optional[str] = None,
                 credibility: Optional[float] = None,
                 perception: Optional[float] = None,
                 interests: Optional[List[str]] = None,
                 trust: Optional[dict] = None):

        self.model = model
        self.id = id

        self.party = party if party is not None else random.choice(PARTY)
        self.credibility = credibility if credibility is not None else 0.5
        self.perception = perception if perception is not None else 0.0
        self.interests = interests if interests is not None else []
        self.trust = trust if trust is not None else {}

        self.newsShared = []
        self.newsReceived = []
        self.newsReceivedIds = set()
        self.newsSharedIds = set()
        self.newsExposureCount = {}
        self.newsSharedCount = {}  # cuenta de reshares por noticia por agente

        # configurable por agente: cuántas veces máximo puede compartir la misma noticia
        self.max_reshares = 3
        # cuanto decae la probabilidad por cada reshare ya hecho (0.0..1.0), cerca de 1 = poca decaída
        self.reshare_decay = 0.91

        if self.model is not None:
            self.model.register_agent(self)

    # INTERÉS Y PROBABILIDAD
    def compute_interest(self, news: News) -> float:
        """Devuelve el interés I_{i,j} en [0,1]."""
        if news.topics and self.interests:
            if any(t in self.interests for t in news.topics):
                return 0.85
        return 0.2 + random.random() * 0.7  # en [0.2,0.9]

    def computeShareProbability(self, news: News,
                                w_P=DEFAULT_wP,
                                w_f=DEFAULT_wF,
                                w_c=DEFAULT_wC,
                                exposure_count=1):
        # componentes
        I_ij = self.compute_interest(news)
        f_k = news.credibility
        c_i = self.credibility
        # bonus por repetición
        bonus = 0.08 * (exposure_count / (exposure_count + 2.0))
        # combinación lineal
        P_base = w_P * I_ij + w_f * f_k + w_c * c_i + bonus

        return clamp(P_base, 0.0, 1.0)

    # EXPOSICIÓN
    def on_exposure(self, news: News, sender: Optional[str] = None):
        first_time = news.id not in self.newsReceivedIds

        if first_time:
            self.newsReceived.append(news)
            self.newsReceivedIds.add(news.id)
            self.newsExposureCount[news.id] = 1
            if self.model is not None:
                self.model.register_news(news)
        else:
            self.newsExposureCount[news.id] += 1

        exposure_count = self.newsExposureCount[news.id]

        # actualización de percepción SIEMPRE
        self.updatePerception(news, exposure_count)

        # posible conversión
        new_type = self.checkConversion()
        if new_type is not None:
            self.convertTo(new_type)

        # decidir compartir (aplicamos decay por reshares previos)
        try:
            pc = self.computeShareProbability(news, exposure_count=exposure_count)
        except TypeError:
            pc = self.computeShareProbability(news)

        prev_shares = self.newsSharedCount.get(news.id, 0)
        # aplicar decay por cada reshare anterior
        effective_pc = pc * (self.reshare_decay ** prev_shares)

        if random.random() < effective_pc:
            # solo permitir hasta max_reshares
            if prev_shares < self.max_reshares:
                self.newsSharedCount[news.id] = prev_shares + 1
                if news.id not in self.newsSharedIds:
                    self.newsSharedIds.add(news.id)
                    self.newsShared.append(news)
                # encolar a vecinos
                for neighbor in self.model.G.neighbors(self.id):
                    self.model.queue_exposure(news, neighbor)

        return first_time

    # PERCEPCIÓN
    def updatePerception(self, news: News, exposure_count: int = 1):
        """Actualiza percepción SIN usar el interés I_{i,j} (según tu petición)."""
        c_i = self.credibility
        repeat_factor = 1 - (0.6 ** exposure_count)
        delta = ALPHA * news.polarity * c_i * repeat_factor
        self.perception = roundto(clamp(self.perception + delta, -1.0, 1.0))

    # CONVERSIÓN
    def checkConversion(self):
        """
        Evitamos conversiones para BOT y NewsReel 
        """
        try:
            if isinstance(self, (BOT, NewsReel)):
                return None
        except NameError:
            # En caso improbable de llamada antes de la definición de clases,
            # no bloqueamos la conversión aquí (esto no ocurre en ejecución normal).
            pass

        value = self.perception

        if isinstance(self, Susceptible):
            if value <= THRESHOLD_TO_SKEPTIC:
                return Skeptic

        elif isinstance(self, Skeptic):
            if value >= THRESHOLD_TO_SUSCEPTIBLE:
                return Susceptible

        return None

    def convertTo(self, new_type):

        try:
            if isinstance(self, (BOT, NewsReel)):
                # no convertir estos tipos; registrar opcionalmente un aviso
                print(f">> SKIP CONVERSION: {self.__class__.__name__} {self.id} no puede convertirse a {new_type.__name__}")
                return
        except NameError:
            # si las clases no existen aún (caso improbable), seguir normalmente
            pass

        old_type_name = self.__class__.__name__
        new_type_name = new_type.__name__

        if new_type == Skeptic:
            self.credibility = clamp(self.credibility * 0.5, 0.1, 0.3)
            self.model.conversions_to_skeptic += 1

        elif new_type == Susceptible:
            self.credibility = clamp(self.credibility * 2, 0.6, 0.9)
            self.model.conversions_to_susceptible += 1

        self.__class__ = new_type
        self.model.converted_agents.append({
            "id": self.id,
            "old_type": old_type_name,
            "new_type": new_type_name,
            "party": self.party,
            "perception": self.perception,
            "new_credibility": self.credibility
        })

        print(f">> CONVERSION: {old_type_name} {self.id} -> {new_type_name} (cred={self.credibility:.3f})")

    # método abstracto
    def shareDecision(self, news: News, exposure_count: int = 1) -> bool:
        raise NotImplementedError


# SUSCEPTIBLE (más propensos a re-compartir)
class Susceptible(User):
    def __init__(self, model: Model, id: str, credibility: Optional[float] = None, **kwargs):
        super().__init__(model, id, credibility=credibility, **kwargs)
        if credibility is None:
            self.credibility = random.uniform(0.6, 0.9)
        self.max_reshares = 10
        self.reshare_decay = 0.91

    def shareDecision(self, news: News, exposure_count: int = 1) -> bool:
        w_P = 0.5
        w_f = 0.3
        w_c = 0.2
        pc = self.computeShareProbability(
            news, w_P=w_P, w_f=w_f, w_c=w_c, exposure_count=exposure_count
        )
        return random.random() < pc


# SKEPTIC (menos re-share, pero permiten compartir noticias verdaderas)
class Skeptic(User):
    def __init__(self, model: Model, id: str, credibility: Optional[float] = None, **kwargs):
        super().__init__(model, id, credibility=credibility, **kwargs)
        if credibility is None:
            self.credibility = random.uniform(0.1, 0.3)
        self.max_reshares = 5
        self.reshare_decay = 0.89

    def shareDecision(self, news: News, exposure_count: int = 1) -> bool:
        # ahora permiten compartir noticias falsas si tienen credibilidad relativamente alta
        if not news.veracity and news.credibility < 0.4:
            return False
        w_P = 0.35
        w_f = 0.5
        w_c = 0.15
        pc = self.computeShareProbability(
            news, w_P=w_P, w_f=w_f, w_c=w_c, exposure_count=exposure_count
        )
        return random.random() < pc

    def updatePerception(self, news: News, exposure_count: int = 1):
        # los escépticos no actualizan percepción ante noticias falsas
        if not news.veracity:
            return
        c_i = self.credibility
        repeat_factor = 1 - (0.6 ** exposure_count)
        delta = ALPHA * news.polarity * c_i * repeat_factor
        self.perception = roundto(clamp(self.perception + delta, -1.0, 1.0))


# BOT (comparten mucho, actúan como seeds recurrentes)
class BOT(User):
    def __init__(self, model: Model, id: str,
                 initialnews: Optional[List[News]] = None, **kwargs):
        super().__init__(model, id, **kwargs)
        self.initialnews = initialnews if initialnews is not None else []
        self.max_reshares = 9999
        self.reshare_decay = 0.99

    def create_news(self, veracity=False, topics: Optional[List[str]] = None, credibility: Optional[float] = None):
        news = News(veracity=veracity, topics=topics, credibility=credibility)
        self.initialnews.append(news)
        if self.model is not None:
            self.model.register_news(news)
        return news

    def shareDecision(self, news: News, exposure_count: int = 1) -> bool:
        return True


# NEWS REEL
class NewsReel(User):
    def __init__(self, model: Model, id: str,
                 initialnews: Optional[List[News]] = None, **kwargs):
        super().__init__(model, id, **kwargs)
        self.initialnews = initialnews if initialnews is not None else []

    def create_news(self, veracity=True, topics: Optional[List[str]] = None, credibility: Optional[float] = None):
        news = News(veracity=veracity, topics=topics, credibility=credibility)
        self.initialnews.append(news)
        if self.model is not None:
            self.model.register_news(news)
        return news

    def shareDecision(self, news: News, exposure_count: int = 1) -> bool:
        return False
