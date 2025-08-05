import gymnasium as gym


class GATv2FeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.spaces.Dict, features_dim: int = 0
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.node_space = observation_space["nodes"].shape

        if self.node_space is None:
            raise ValueError(
                "Observation space must contain 'nodes' with a valid shape."
            )

        self.gcn1 = GATv2Conv(self.node_space[1], 64, edge_dim=1)
        self.gcn2 = GATv2Conv(64, 32, edge_dim=1)
        self.gcn3 = GATv2Conv(32, 16, edge_dim=1)

        self.dropout = torch.nn.Dropout(0.1)

        self.feature_liner = torch.nn.Linear(16, features_dim)

    def forward(self, observations) -> torch.Tensor:
        batch_size = observations["nodes"].shape[0]
        nodes_num = observations["nodes_num"].long()
        edge_links_num = observations["edge_links_num"].long()

        # 배치 내 각 그래프에 대해 Data 객체 생성
        data_list = []
        for i in range(batch_size):
            nodes = observations["nodes"][i].float()
            edge_index = observations["edge_links"][i].long()
            edge_attributes = observations["edge_attributes"][i].float()

            filtered_nodes = nodes[: nodes_num[i]]
            filtered_edge_index = edge_index[: edge_links_num[i]]
            filtered_edge_attributes = edge_attributes[: edge_links_num[i]]

            data = Data(
                x=filtered_nodes,
                edge_index=filtered_edge_index.T,
                edge_attr=filtered_edge_attributes,
            )
            data_list.append(data)

        # PyTorch Geometric의 배치 처리
        batch = Batch.from_data_list(data_list)

        if batch.batch.dtype != torch.long:  # type: ignore
            batch.batch = batch.batch.long()  # type: ignore

        # GCN 통과
        x = self.gcn1(batch.x, batch.edge_index, edge_attr=batch.edge_attr)  # type: ignore
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.gcn2(x, batch.edge_index, edge_attr=batch.edge_attr)  # type: ignore
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.gcn3(x, batch.edge_index, edge_attr=batch.edge_attr)  # type: ignore
        x = torch.relu(x)

        # Global mean pooling 적용
        graph_features = global_mean_pool(
            x,
            batch.batch,  # type: ignore
        )

        # 선형 레이어를 통해 최종 특징 차원으로 변환
        graph_features = self.feature_liner(graph_features)  # (batch_size, features_dim

        return graph_features
