import networkx as nx
import random
import matplotlib.pyplot as plt
from math import comb


def generate_random_graph(n, target_density, seed: int | None = None):
    if seed is not None:
        random.seed(seed)

    max_edges = comb(n, 2)
    min_edges = n - 1  # 최소 n-1개는 있어야 모든 노드가 연결된 상태가 됨

    target_edges = int(target_density * max_edges)
    target_edges = max(target_edges, min_edges)

    # 초기화
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = set()

    # 각 노드가 적어도 하나의 간선을 갖도록 보장 (랜덤하게 연결)
    nodes = list(range(n))
    random.shuffle(nodes)
    for i in range(n - 1):
        u, v = nodes[i], nodes[i + 1]
        edges.add((min(u, v), max(u, v)))  # 중복 방지

    # 마지막 노드를 다른 아무 노드에 연결 (만약 n == 2면 이미 연결됨)
    if n > 2:
        u, v = nodes[-1], random.choice(nodes[:-1])
        edges.add((min(u, v), max(u, v)))

    # 추가 간선 랜덤 생성 (중복/자기자신 제외)
    while len(edges) < target_edges:
        u, v = random.sample(range(n), 2)
        edge = (min(u, v), max(u, v))
        if edge not in edges:
            edges.add(edge)

    # 그래프에 간선 추가
    G.add_edges_from(edges)

    return G


if __name__ == "__main__":
    # 파라미터 설정
    n = 4  # 노드 수
    density = random.random()  # 밀도 (0~1)

    G = generate_random_graph(n, density)

    # 시각화
    nx.draw(G, with_labels=True)
    plt.title(f"Random Graph (n={n}, density={density:.2f})")
    plt.show()

    # 유효성 확인
    print("모든 노드 degree ≥ 1:", all(dict(G.degree()).values()))  # type: ignore
    print("자기 루프 있음?", nx.number_of_selfloops(G) > 0)
    print("총 간선 수:", G.number_of_edges())
