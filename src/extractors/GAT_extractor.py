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

        # GAT 레이어들 - 더 효율적인 구조
        self.gat1 = GATConv(
            in_channels=self.node_features_dim,
            out_channels=16,  # 채널 수 감소
            heads=2,  # 헤드 수 감소
            dropout=0.0,  # 드롭아웃 제거 (추론 시 불필요)
            concat=True,
        )
        self.gat2 = GATConv(
            in_channels=16 * 2,  # 2 heads * 16 channels
            out_channels=16,
            heads=1,
            dropout=0.0,
            concat=False,
        )

        # Edge scoring network - 더 간단한 구조
        self.edge_scorer = torch.nn.Sequential(
            torch.nn.Linear(16 * 2, 16),  # 층 수 감소
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),  # 최종 점수
            torch.nn.Sigmoid(),
        )

        # 최종 출력 레이어 (candidate scores를 features_dim으로 변환)
        self.output_layer = torch.nn.Linear(self.max_candidates, features_dim)

    def forward(self, observations) -> torch.Tensor:
        """
        최고 성능 최적화 버전 - CPU 사용량 최소화
        """
        batch_size = observations["nodes"].shape[0]
        device = observations["nodes"].device

        # 1. 모든 텐서를 한 번에 처리
        all_nodes = observations["nodes"].float()  # (batch_size, node_num, 2)
        batch_node_num = all_nodes.shape[1]
        all_nodes_flat = all_nodes.reshape(-1, 2)  # (total_nodes, 2) - view -> reshape

        # 2. 엣지 처리 완전 벡터화
        all_edge_links = observations[
            "edge_links"
        ].long()  # (batch_size, max_edge_num, 2)
        all_edges_num = observations["edges_num"].long().squeeze()  # (batch_size,)

        # 3. 노드 오프셋을 GPU에서 한 번에 계산
        node_offsets = (
            torch.arange(batch_size, device=device).unsqueeze(1) * batch_node_num
        )

        # 4. 마스크 기반 엣지 처리 (루프 최소화)
        max_edges = all_edge_links.shape[1]
        edge_mask = torch.arange(max_edges, device=device).unsqueeze(
            0
        ) < all_edges_num.unsqueeze(1)
        valid_edge_mask = (all_edge_links >= 0).all(dim=2) & edge_mask

        # 5. 유효한 엣지들을 한 번에 수집
        flat_edge_links = all_edge_links.reshape(-1, 2)  # view -> reshape
        flat_valid_mask = valid_edge_mask.reshape(-1)  # view -> reshape
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(
            max_edges
        )

        if flat_valid_mask.any():
            valid_edges = flat_edge_links[flat_valid_mask]
            valid_batch_indices = batch_indices[flat_valid_mask]
            edge_offsets = node_offsets.squeeze(1)[valid_batch_indices]
            all_edge_indices = (valid_edges + edge_offsets.unsqueeze(1)).t()
        else:
            all_edge_indices = torch.empty((2, 0), dtype=torch.long, device=device)

        # 6. GAT 통과 (단일 호출)
        x = self.gat1(all_nodes_flat, all_edge_indices)
        x = torch.relu(x)
        x = self.gat2(x, all_edge_indices)
        x = torch.relu(x)

        # 7. 배치별 재구성
        x_batched = x.reshape(batch_size, batch_node_num, -1)  # view -> reshape

        # 8. Candidate scoring 완전 벡터화
        all_candidate_edges = observations["candidate_edges"].long()

        # 모든 배치의 candidate edges를 한 번에 처리
        batch_idx = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(-1, self.max_candidates)
        )
        candidate_mask = (all_candidate_edges >= 0).all(dim=2) & (
            all_candidate_edges < batch_node_num
        ).all(dim=2)

        # 유효한 candidate들의 특징을 한 번에 추출
        flat_candidates = all_candidate_edges.reshape(-1, 2)  # view -> reshape
        flat_candidate_mask = candidate_mask.reshape(-1)  # view -> reshape
        flat_batch_idx = batch_idx.reshape(-1)  # view -> reshape

        all_candidate_scores = torch.zeros(
            batch_size, self.max_candidates, device=device
        )

        if flat_candidate_mask.any():
            valid_candidates = flat_candidates[flat_candidate_mask]
            valid_batch_indices = flat_batch_idx[flat_candidate_mask]

            # 배치별 노드 특징 추출 (고급 인덱싱)
            node1_features = x_batched[valid_batch_indices, valid_candidates[:, 0]]
            node2_features = x_batched[valid_batch_indices, valid_candidates[:, 1]]
            edge_features = torch.cat([node1_features, node2_features], dim=1)

            # 점수 계산
            scores = self.edge_scorer(edge_features).squeeze(-1)

            # 점수를 원래 위치에 할당
            original_indices = torch.where(flat_candidate_mask)[0]
            batch_indices_2d = original_indices // self.max_candidates
            candidate_indices_2d = original_indices % self.max_candidates
            all_candidate_scores[batch_indices_2d, candidate_indices_2d] = scores

        # 9. 최종 출력
        return self.output_layer(all_candidate_scores)
