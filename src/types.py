from typing import Dict, List, Set, Tuple, TypeAlias

NodeId: TypeAlias = int
EdgeKey: TypeAlias = Tuple[NodeId, NodeId]
Adjacency: TypeAlias = List[Set[NodeId]]
LatencyMap: TypeAlias = Dict[EdgeKey, float]
BandwidthMap: TypeAlias = Dict[EdgeKey, float]
