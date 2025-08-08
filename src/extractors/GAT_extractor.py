import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATConv


class GATConvFeatureExtractor(BaseFeaturesExtractor):
    """
    GAT 기반 특징 추출기
    CandidateWrapper의 observation space를 받아서 candidate edges에 대한 점수를 반환
    """

    def __init__(
        self, observation_space: gym.spaces.Dict, features_dim: int = 16
    ) -> None:
        super().__init__(observation_space, features_dim)

        # Observation space 정보 추출
        nodes_shape = observation_space["nodes"].shape
        candidates_shape = observation_space["candidate_edges"].shape

        if nodes_shape is None or candidates_shape is None:
            raise ValueError("Invalid observation space shapes")

        self.node_features_dim = nodes_shape[1]  # (node_num, 2) -> 2
        self.max_candidates = candidates_shape[0]  # candidate_num

        # GAT 레이어들
        self.gat1 = GATConv(
            in_channels=self.node_features_dim,
            out_channels=32,
            heads=4,
            dropout=0.1,
            concat=True,
        )
        self.gat2 = GATConv(
            in_channels=32 * 4,  # 4 heads * 32 channels
            out_channels=16,
            heads=1,
            dropout=0.1,
            concat=False,
        )

        # Edge scoring network
        # 두 노드의 특징을 연결해서 점수를 매기는 네트워크
        self.edge_scorer = torch.nn.Sequential(
            torch.nn.Linear(16 * 2, 32),  # 두 노드 특징 연결
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),  # 최종 점수
            torch.nn.Sigmoid(),  # 0-1 사이 점수
        )

        # 최종 출력 레이어 (candidate scores를 features_dim으로 변환)
        self.output_layer = torch.nn.Linear(self.max_candidates, features_dim)

    def forward(self, observations) -> torch.Tensor:
        """
        최적화된 forward 함수
        """
        batch_size = observations["nodes"].shape[0]
        device = observations["nodes"].device

        # 모든 배치를 한 번에 처리하기 위한 벡터화된 접근

        # 1. 모든 노드를 한 번에 준비
        all_nodes = observations["nodes"].float()  # (batch_size, node_num, 2)
        batch_node_num = all_nodes.shape[
            1
        ]  # 모든 그래프가 같은 node_num을 가진다고 가정

        # 2. 엣지 처리를 벡터화
        all_edge_links = observations[
            "edge_links"
        ].long()  # (batch_size, max_edge_num, 2)
        all_edges_num = observations["edges_num"].long().squeeze()  # (batch_size,)

        # 3. 배치별 offset 계산
        node_offsets = torch.arange(batch_size, device=device) * batch_node_num

        # 4. 모든 노드를 평면화
        all_nodes_flat = all_nodes.view(-1, 2)  # (total_nodes, 2)

        # 5. 엣지 인덱스를 배치별로 오프셋 적용하여 수집
        edge_indices_list = []
        for i in range(batch_size):
            edges_num = int(all_edges_num[i].item())
            if edges_num > 0:
                valid_edges = all_edge_links[i][:edges_num]
                valid_mask = (valid_edges >= 0).all(dim=1)
                if valid_mask.any():
                    valid_edge_indices = valid_edges[valid_mask] + node_offsets[i]
                    edge_indices_list.append(valid_edge_indices.t())

        if edge_indices_list:
            all_edge_indices = torch.cat(edge_indices_list, dim=1)
        else:
            all_edge_indices = torch.empty((2, 0), dtype=torch.long, device=device)

        # 6. GAT 통과 (전체 배치를 한 번에)
        x = self.gat1(all_nodes_flat, all_edge_indices)
        x = torch.relu(x)
        x = self.gat2(x, all_edge_indices)
        x = torch.relu(x)  # (total_nodes, 16)

        # 7. 노드 특징을 배치별로 재구성
        x_batched = x.view(batch_size, batch_node_num, -1)  # (batch_size, node_num, 16)

        # 8. Candidate edge scoring을 벡터화
        all_candidate_edges = observations[
            "candidate_edges"
        ].long()  # (batch_size, candidate_num, 2)
        all_candidate_scores = torch.zeros(
            batch_size, self.max_candidates, device=device
        )

        # 배치별로 candidate edges 처리 (여전히 루프가 필요하지만 내부 연산은 벡터화)
        for i in range(batch_size):
            candidate_edges = all_candidate_edges[i]  # (candidate_num, 2)
            graph_features = x_batched[i]  # (node_num, 16)

            # 유효한 candidate edges 마스크
            valid_mask = (candidate_edges >= 0).all(dim=1) & (
                candidate_edges < batch_node_num
            ).all(dim=1)

            if valid_mask.any():
                valid_candidates = candidate_edges[valid_mask]  # (valid_num, 2)

                # 벡터화된 특징 추출
                node1_features = graph_features[
                    valid_candidates[:, 0]
                ]  # (valid_num, 16)
                node2_features = graph_features[
                    valid_candidates[:, 1]
                ]  # (valid_num, 16)
                edge_features = torch.cat(
                    [node1_features, node2_features], dim=1
                )  # (valid_num, 32)

                # 한 번에 점수 계산
                scores = self.edge_scorer(edge_features).squeeze(-1)  # (valid_num,)

                # 점수 할당
                valid_indices = torch.where(valid_mask)[0]
                all_candidate_scores[i, valid_indices] = scores

        # 9. 최종 출력
        output_features = self.output_layer(all_candidate_scores)
        return output_features
