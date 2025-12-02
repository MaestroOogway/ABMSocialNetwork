import random
from typing import List, Optional


PARTY = ["A", "B"]
POLARITYNEWS = [-1, 1]
VERACITYNEWS = [False, True]
PHI = 0.6
ALPHA = 0.1

THRESHOLD_TO_SKEPTIC = -0.3
THRESHOLD_TO_SUSCEPTIBLE = 0.3


def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def roundto(x, ndigits=3):
    return round(x, ndigits)


class News:
    count = 0

    def __init__(self, id=None, party=None, polarity=None, veracity=None, credibility=None):
        if id is None:
            id = News.count
        self.id = id
        News.count += 1

        self.party = party if party is not None else random.choice(PARTY)
        self.polarity = polarity if polarity is not None else random.choice(POLARITYNEWS)
        self.veracity = veracity if veracity is not None else random.choice(VERACITYNEWS)

        if credibility is None:
            self.credibility = random.uniform(0.7, 0.9) if self.veracity else random.uniform(0.1, 0.3)
        else:
            self.credibility = credibility

    def __repr__(self):
        return f"News(id={self.id}, party={self.party}, pol={self.polarity}, veracity={self.veracity}, cred={self.credibility:.2f})"


class Model:
    def __init__(self, G):
        self.G = G
        self.agents = {}
        self.news_propagation = []
        self.converted_agents = []
        self.conversions_to_skeptic = 0
        self.conversions_to_susceptible = 0

    def register_agent(self, agent):
        self.agents[agent.id] = agent
        agent.model = self

    def get_agent(self, node_id):
        return self.agents[node_id]


class User:
    def __init__(self, model: Model, id: str, party: Optional[str] = None,
                 credibility: Optional[float] = None, perception: Optional[dict] = None):
        self.model = model
        self.id = id
        self.party = party if party is not None else random.choice(PARTY)

        self.credibility = credibility
        self.perception = perception if perception is not None else {"A": 0.0, "B": 0.0}

        self.newsShared = []
        self.newsReceived = []
        self.newsReceivedIds = set()
        self.newsSharedIds = set()
        self.newsExposureCount = {}

        if self.model is not None:
            self.model.register_agent(self)

    def computeShareProbability(self, news: News, w_m, w_f, w_c):
        p_i = 1 if self.party == "A" else -1
        p_j = 1 if news.party == "A" else -1
        polarity = news.polarity

        m_ij = (1 + polarity * p_j * p_i) / 2.0
        f_k = news.credibility
        c_i = self.credibility

        P_C = clamp(w_m * m_ij + w_f * f_k + w_c * c_i, 0, 1)
        return P_C

    def on_exposure(self, news: News, sender: Optional[str] = None):
        if news.id in self.newsReceivedIds:
            self.newsExposureCount[news.id] = self.newsExposureCount.get(news.id, 1) + 1
            return False

        self.newsReceived.append(news)
        self.newsReceivedIds.add(news.id)
        self.newsExposureCount[news.id] = 1

        self.updatePerception(news)

        new_type = self.checkConversion()
        if new_type is not None:
            self.convertTo(new_type)

        share = False
        try:
            share = self.shareDecision(news)
        except NotImplementedError:
            share = False

        if share:
            self.newsShared.append(news)
            self.newsSharedIds.add(news.id)

        return True

    def updatePerception(self, news: News):
        x_j = 1 if news.party != self.party else 0
        delta = x_j * ALPHA * news.polarity * (self.credibility if self.credibility is not None else 0.5)
        old = self.perception[news.party]
        self.perception[news.party] = roundto(clamp(old + delta, -1.0, 1.0))

    def checkConversion(self):
        other = "B" if self.party == "A" else "A"
        value = self.perception[other]

        if isinstance(self, Susceptible):
            if value <= THRESHOLD_TO_SKEPTIC:
                return Skeptic

        elif isinstance(self, Skeptic):
            if value >= THRESHOLD_TO_SUSCEPTIBLE:
                return Susceptible

        return None

    def convertTo(self, new_type):
        old_type_name = self.__class__.__name__
        new_type_name = new_type.__name__

        if new_type == Skeptic:
            self.credibility = clamp((self.credibility if self.credibility is not None else 0.5) * 0.5, 0.1, 0.3)
            self.model.conversions_to_skeptic += 1

        elif new_type == Susceptible:
            self.credibility = clamp((self.credibility if self.credibility is not None else 0.5) * 2.0, 0.6, 0.9)
            self.model.conversions_to_susceptible += 1

        self.__class__ = new_type

        self.model.converted_agents.append({
            "id": self.id,
            "old_type": old_type_name,
            "new_type": new_type_name,
            "party": self.party,
            "perception": dict(self.perception),
            "new_credibility": self.credibility
        })

        print(f">> CONVERSION: {old_type_name} {self.id} -> {new_type_name} (cred={self.credibility:.3f})")

    def shareDecision(self, news: News) -> bool:
        raise NotImplementedError


class Susceptible(User):
    def __init__(self, model: Model, id: str, credibility: Optional[float] = None, **kwargs):
        super().__init__(model, id, credibility=credibility, **kwargs)
        if self.credibility is None:
            self.credibility = random.uniform(0.6, 0.9)

    def shareDecision(self, news: News) -> bool:
        w1 = 0.1
        w2 = 0.3
        w3 = 0.6
        pc = self.computeShareProbability(news, w1, w2, w3)
        return random.random() < pc

    def updatePerception(self, news: News):
        x_j = 1 if news.party != self.party else 0
        delta = x_j * ALPHA * news.polarity * self.credibility
        old = self.perception[news.party]
        self.perception[news.party] = roundto(clamp(old + delta, -1.0, 1.0))


class Skeptic(User):
    def __init__(self, model: Model, id: str, credibility: Optional[float] = None, **kwargs):
        super().__init__(model, id, credibility=credibility, **kwargs)
        if self.credibility is None:
            self.credibility = random.uniform(0.1, 0.3)

    def shareDecision(self, news: News) -> bool:
        w1 = 0.3
        w2 = 0.6
        w3 = 0.1
        if news.veracity is False:
            return False
        pc = self.computeShareProbability(news, w1, w2, w3)
        return random.random() < pc

    def updatePerception(self, news: News):
        if not news.veracity:
            return
        x_j = 1 if news.party != self.party else 0
        delta = x_j * ALPHA * news.polarity * self.credibility
        old = self.perception[news.party]
        self.perception[news.party] = roundto(clamp(old + delta, -1.0, 1.0))


class BOT(User):
    def __init__(self, model: Model, id: str, initialnews: Optional[List[News]] = None, **kwargs):
        super().__init__(model, id, **kwargs)
        self.initialnews = initialnews if initialnews is not None else []

    def create_news(self):
        news = News(veracity=False)
        self.initialnews.append(news)
        return news


class NewsReel(User):
    def __init__(self, model: Model, id: str, initialnews: Optional[List[News]] = None, **kwargs):
        super().__init__(model, id, **kwargs)
        self.initialnews = initialnews if initialnews is not None else []

    def create_news(self):
        news = News(veracity=True)
        self.initialnews.append(news)
        return news
