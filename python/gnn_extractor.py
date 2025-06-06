"""
GNN Feature Extractor for Drone Delivery Environment
Uses GAT to process K=10 local neighborhood structure
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class LocalK10GNN(BaseFeaturesExtractor):
    """
    GNN-based feature extractor for K=10 local neighborhood
    Transforms observation (118 dims) into embedding via 2-layer GAT
    
    Input structure:
    - obs[:, :6]: global features (battery, payload, flags, recharge, delivery_dist)
    - obs[:, 6:86]: local K=10 features (10 neighbors * 8 features each)
    - obs[:, 86:]: node embedding (if enabled)
    """
    
    def __init__(self, observation_space: gym.Space, 
                 hidden_dim: int = 64, 
                 heads: int = 4,
                 gnn_layers: int = 2,
                 aggregation: str = "mean",
                 features_dim: int = 128):
        
        # Calculate input dimension from observation space
        input_dim = observation_space.shape[0]
        
        super(LocalK10GNN, self).__init__(observation_space, features_dim)
        
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.gnn_layers = gnn_layers
        self.aggregation = aggregation
        self.input_dim = input_dim
        
        # Global features processor (battery, payload, flags, etc.)
        self.global_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # GNN layers for K=10 local neighborhood
        self.gat_layers = nn.ModuleList()
        
        # First GAT layer: 8 features -> hidden_dim
        self.gat_layers.append(
            GATConv(8, hidden_dim // heads, heads=heads, concat=True, dropout=0.1)
        )
        
        # Additional GAT layers
        for _ in range(gnn_layers - 1):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, dropout=0.1)
            )
        
        # Node embedding processor (if present)
        embedding_dim = max(0, input_dim - 86)  # Remaining dims after global + local
        if embedding_dim > 0:
            self.embedding_mlp = nn.Sequential(
                nn.Linear(embedding_dim, 32),
                nn.ReLU()
            )
            embedding_output_dim = 32
        else:
            self.embedding_mlp = None
            embedding_output_dim = 0
        
        # Final fusion network
        fusion_input_dim = hidden_dim + 64 + embedding_output_dim  # GNN + global + embedding
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        print(f"🧠 GNN Extractor initialized:")
        print(f"   Input dim: {input_dim}")
        print(f"   Hidden dim: {hidden_dim}, Heads: {heads}")
        print(f"   GNN layers: {gnn_layers}")
        print(f"   Aggregation: {aggregation}")
        print(f"   Features dim: {features_dim}")
        print(f"   Embedding dim: {embedding_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Process observations through GNN
        
        Args:
            observations: (batch_size, input_dim)
            
        Returns:
            features: (batch_size, features_dim)
        """
        batch_size = observations.size(0)
        device = observations.device
        
        # Split observation components
        global_feat = observations[:, :6]                    # (B, 6): battery, payload, flags
        local_feat = observations[:, 6:86].view(batch_size, 10, 8)  # (B, 10, 8): K=10 neighbors
        
        # Process global features
        global_embed = self.global_mlp(global_feat)          # (B, 64)
        
        # Process node embeddings (if present)
        embedding_embed = None
        if self.embedding_mlp is not None and observations.size(1) > 86:
            embedding_feat = observations[:, 86:]
            embedding_embed = self.embedding_mlp(embedding_feat)  # (B, 32)
        
        # Construct mini-graph for each batch element
        # K=10 nodes: center (idx 0) connected to 9 neighbors (idx 1-9)
        
        # Flatten node features for PyG format
        x = local_feat.reshape(-1, 8)                        # (B*10, 8)
        
        # Create edge connectivity: star topology with center node
        # Center node (idx 0, 10, 20, ...) connects to neighbors (1-9, 11-19, 21-29, ...)
        edge_lists = []
        for b in range(batch_size):
            base_idx = b * 10
            center_idx = base_idx
            neighbor_indices = torch.arange(base_idx + 1, base_idx + 10, device=device)
            
            # Bidirectional edges: center <-> neighbors
            edges_out = torch.stack([
                torch.full((9,), center_idx, device=device),  # center -> neighbors
                neighbor_indices
            ], dim=0)
            
            edges_in = torch.stack([
                neighbor_indices,                              # neighbors -> center
                torch.full((9,), center_idx, device=device)
            ], dim=0)
            
            edge_lists.append(torch.cat([edges_out, edges_in], dim=1))
        
        # Concatenate all edges
        edge_index = torch.cat(edge_lists, dim=1)            # (2, B*18)
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers) - 1:  # No activation after last layer
                x = torch.relu(x)
        
        # Aggregate node features per batch
        # Take only center node features (indices 0, 10, 20, ...)
        if self.aggregation == "mean":
            # Alternative: aggregate all 10 nodes per batch
            batch_indices = torch.arange(batch_size, device=device).repeat_interleave(10)
            gnn_embed = global_mean_pool(x, batch_indices)   # (B, hidden_dim)
        elif self.aggregation == "max":
            batch_indices = torch.arange(batch_size, device=device).repeat_interleave(10)
            gnn_embed = global_max_pool(x, batch_indices)    # (B, hidden_dim)
        else:  # "center" - take only center node features
            center_indices = torch.arange(0, batch_size * 10, 10, device=device)
            gnn_embed = x[center_indices]                    # (B, hidden_dim)
        
        # Fuse all embeddings
        if embedding_embed is not None:
            fused_features = torch.cat([gnn_embed, global_embed, embedding_embed], dim=1)
        else:
            fused_features = torch.cat([gnn_embed, global_embed], dim=1)
        
        # Final processing
        output = self.fusion_net(fused_features)             # (B, features_dim)
        
        return output


class LocalK10GNNSimple(BaseFeaturesExtractor):
    """
    Simplified GNN extractor for faster training
    Single GAT layer + lightweight processing
    """
    
    def __init__(self, observation_space: gym.Space, 
                 hidden_dim: int = 32, 
                 heads: int = 2,
                 features_dim: int = 128):
        
        input_dim = observation_space.shape[0]
        super(LocalK10GNNSimple, self).__init__(observation_space, features_dim)
        
        self.hidden_dim = hidden_dim
        self.heads = heads
        
        # Single GAT layer
        self.gat = GATConv(8, hidden_dim, heads=heads, concat=False, dropout=0.1)
        
        # Global features
        self.global_net = nn.Linear(6, 32)
        
        # Embedding features (if present)
        embedding_dim = max(0, input_dim - 86)
        if embedding_dim > 0:
            self.embedding_net = nn.Linear(embedding_dim, 16)
            total_dim = hidden_dim + 32 + 16
        else:
            self.embedding_net = None
            total_dim = hidden_dim + 32
        
        # Output
        self.output_net = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.ReLU()
        )
        
        print(f"🧠 Simple GNN Extractor: {input_dim} -> {features_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.size(0)
        device = observations.device
        
        # Extract components
        global_feat = observations[:, :6]
        local_feat = observations[:, 6:86].view(batch_size, 10, 8)
        
        # Process global
        global_embed = torch.relu(self.global_net(global_feat))
        
        # Process embeddings
        embedding_embed = None
        if self.embedding_net is not None and observations.size(1) > 86:
            embedding_feat = observations[:, 86:]
            embedding_embed = torch.relu(self.embedding_net(embedding_feat))
        
        # Simple GNN: flatten and create star edges
        x = local_feat.reshape(-1, 8)
        
        # Star topology edges
        center_indices = torch.arange(0, batch_size * 10, 10, device=device)
        neighbor_indices = []
        for b in range(batch_size):
            base = b * 10
            neighbor_indices.extend(range(base + 1, base + 10))
        neighbor_indices = torch.tensor(neighbor_indices, device=device)
        
        # Bidirectional edges
        edge_index = torch.stack([
            torch.cat([center_indices.repeat_interleave(9), neighbor_indices]),
            torch.cat([neighbor_indices, center_indices.repeat_interleave(9)])
        ])
        
        # Single GAT pass
        x = self.gat(x, edge_index)
        x = torch.relu(x)
        
        # Take center nodes only
        gnn_embed = x[center_indices]
        
        # Fuse and output
        if embedding_embed is not None:
            features = torch.cat([gnn_embed, global_embed, embedding_embed], dim=1)
        else:
            features = torch.cat([gnn_embed, global_embed], dim=1)
        
        return self.output_net(features)
