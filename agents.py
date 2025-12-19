import random
import math
from typing import List, Optional

PARTY = ["N"]
POLARITYNEWS = [-1, 1]
VERACITYNEWS = [False, True]
PHI = 0.6
ALPHA = 0.1


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def roundto(x, ndigits=3):
    return round(x, ndigits)


class News:
    count = 0

    def __init__(self, id=None, party=None, polarity=None, veracity=None, credibility=None, source: Optional[str] = None, topics: Optional[List[str]] = None, salience: Optional[float] = None, relevance: Optional[float] = None):

        if id is None:
            id = News.count
        self.id = id
        News.count += 1

        self.party = party if party is not None else random.choice(PARTY)
        self.polarity = polarity if polarity is not None else random.choice(POLARITYNEWS)
        self.veracity = veracity if veracity is not None else random.choice(VERACITYNEWS)

        if credibility is None:
            self.credibility = random.uniform(0.70, 0.99) if self.veracity else random.uniform(0.01, 0.35)
        else:
            self.credibility = credibility

        self.source = source if source is not None else "generic"
        self.topics = topics if topics is not None else []
        self.salience = salience if salience is not None else random.random()
        self.relevance = relevance

    def __repr__(self):
        return f"News(id={self.id}, party={self.party}, pol={self.polarity}, " f"veracity={self.veracity}, cred={self.credibility:.2f}, " f"src={self.source}, sal={self.salience:.2f})"


class Model:
    def __init__(self, G):
        self.G = G
        self.agents = {}
        self.news_map = {}
        self.exposure_queue = {}
        self.news_propagation = []

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

        dests = dest_id_or_iterable if isinstance(dest_id_or_iterable, (list, set, tuple)) else [dest_id_or_iterable]

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


class User:
    def __init__(self, model: Model, id, party: Optional[str] = None, credibility: Optional[float] = None, perception: Optional[float] = None, interests: Optional[List[str]] = None, trust: Optional[dict] = None):

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
        self.newsSharedCount = {}

        self.max_reshares = 3
        self.reshare_decay = 0.91

        if self.model is not None:
            self.model.register_agent(self)

    def compute_interest(self, news: News) -> float:
        """Returns interest score in [0,1]."""
        if news.topics and self.interests:
            if any(t in self.interests for t in news.topics):
                return 0.85
        return 0.2 + random.random() * 0.7

    def computeShareProbability(self, news: News, w_P: float, w_f: float, w_c: float, exposure_count=1):
        I_ij = self.compute_interest(news)
        f_k = news.credibility
        c_i = self.credibility
        bonus = 0.08 * (exposure_count / (exposure_count + 2.0))
        P_base = w_P * I_ij + w_f * f_k + w_c * c_i + bonus
        return clamp(P_base, 0.0, 1.0)

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
        self.updatePerception(news, exposure_count)

        pc = 1.0 if self.shareDecision(news, exposure_count) else 0.0
        prev_shares = self.newsSharedCount.get(news.id, 0)
        effective_pc = pc * (self.reshare_decay**prev_shares)

        if random.random() < effective_pc and prev_shares < self.max_reshares:
            self.newsSharedCount[news.id] = prev_shares + 1
            if news.id not in self.newsSharedIds:
                self.newsSharedIds.add(news.id)
                self.newsShared.append(news)
            for neighbor in self.model.G.neighbors(self.id):
                self.model.queue_exposure(news, neighbor)

        return first_time

    def updatePerception(self, news: News, exposure_count: int = 1):
        """Updates perception using exposure-based reinforcement."""
        c_i = self.credibility
        repeat_factor = 1 - (0.6**exposure_count)
        delta = ALPHA * news.polarity * c_i * repeat_factor
        self.perception = roundto(clamp(self.perception + delta, -1.0, 1.0))

    def shareDecision(self, news: News, exposure_count: int = 1) -> bool:
        raise NotImplementedError


class Susceptible(User):
    def __init__(self, model: Model, id: str, credibility: Optional[float] = None, **kwargs):
        super().__init__(model, id, credibility=credibility, **kwargs)
        if credibility is None:
            self.credibility = random.uniform(0.6, 0.9)
        self.max_reshares = 10
        self.reshare_decay = 0.91

    def shareDecision(self, news: News, exposure_count: int = 1) -> bool:
        w_P = 0.4
        w_f = 0.1
        w_c = 0.6
        pc = self.computeShareProbability(news, w_P=w_P, w_f=w_f, w_c=w_c, exposure_count=exposure_count)
        return random.random() < pc


class Skeptic(User):
    def __init__(self, model: Model, id: str, credibility: Optional[float] = None, **kwargs):
        super().__init__(model, id, credibility=credibility, **kwargs)
        if credibility is None:
            self.credibility = random.uniform(0.1, 0.3)
        self.max_reshares = 5
        self.reshare_decay = 0.89

    def shareDecision(self, news: News, exposure_count: int = 1) -> bool:
        w_P = 0.20
        w_f = 0.60
        w_c = 0.10
        pc = self.computeShareProbability(news, w_P=w_P, w_f=w_f, w_c=w_c, exposure_count=exposure_count)
        return random.random() < pc


class BOT(User):
    def __init__(self, model: Model, id: str, initialnews: Optional[List[News]] = None, **kwargs):
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


class NewsReel(User):
    def __init__(self, model: Model, id: str, initialnews: Optional[List[News]] = None, **kwargs):
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
