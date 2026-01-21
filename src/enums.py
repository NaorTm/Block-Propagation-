from enum import Enum


class Protocol(Enum):
    NAIVE = "naive"
    TWO_PHASE = "two-phase"
    PUSH = "push"
    PULL = "pull"
    PUSH_PULL = "push-pull"
    BITCOIN_COMPACT = "bitcoin-compact"


class Topology(Enum):
    RANDOM_REGULAR = "random-regular"
    SCALE_FREE = "scale-free"
    SMALL_WORLD = "small-world"
    STAR = "star"
    LINE = "line"


class LatencyDist(Enum):
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"


class BandwidthDist(Enum):
    FIXED = "fixed"
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"
