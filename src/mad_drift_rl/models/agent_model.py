import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentTransformer(nn.Module):
    """
        This special transformer layer cross-attends the current latents with the memory latents.
    """
    def __init__(self, feature_dim: int = 128):
        super(RecurrentTransformer, self).__init__()
        # Current-to-memory attention
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(feature_dim)

        # Memory update attention
        self.memory_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.memory_fc = nn.Linear(feature_dim, feature_dim)
        self.memory_gelu = nn.GELU()
        self.memory_layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, current_latents: torch.Tensor, memory_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Cross-attention: current latents attend to memory latents
        # query=current, key=memory, value=memory
        attn_output, _ = self.attention(current_latents, memory_latents, memory_latents)

        # Add & Norm (residual connection)
        x = self.layer_norm(current_latents + attn_output)

        # Feedforward with residual
        x = x + self.gelu(self.fc(x))

        # Memory update: memory attends to current latents
        # query=memory, key=current, value=current
        memory_attn_output, _ = self.memory_attention(memory_latents, current_latents, current_latents)

        # Update memory with residual connection
        updated_memory = self.memory_layer_norm(memory_latents + memory_attn_output)
        updated_memory = updated_memory + self.memory_gelu(self.memory_fc(updated_memory))

        return x, updated_memory

class AgentModel(nn.Module):
    def __init__(self, feature_dim: int = 128, n_latent: int = 16, n_layer: int = 2):
        super(AgentModel, self).__init__()

        self.feature_dim = feature_dim
        self.n_latent = n_latent

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.LazyLinear(feature_dim),
            nn.ReLU()
        )

        self.recurrent_transformer = nn.ModuleList([
            RecurrentTransformer(feature_dim=feature_dim)
            for _ in range(n_layer)
        ])

        self.action_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 3)  # Assuming 3 possible actions
        )

        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )

        # Action interval control (AlphaStar-style)
        self.action_embedding = nn.Embedding(3, feature_dim)  # 3 actions
        self.interval_head = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),  # concat pooled + action_embed
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()  # outputs [0, 1]
        )


    def forward(self, x: torch.Tensor, memory_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, 1, height, width)
        # memory_latents: (batch, n_latent, feature_dim)

        # Encode visual features
        visual_features = self.visual_encoder(x)  # (batch, feature_dim)

        # Reshape to (batch, 1, feature_dim) for transformer
        current_latents = visual_features.unsqueeze(1)  # (batch, 1, feature_dim)

        # Pass through recurrent transformer layers
        updated_memory = memory_latents
        for layer in self.recurrent_transformer:
            current_latents, updated_memory = layer(current_latents, updated_memory)

        # Pool current latents for heads (take the single latent)
        pooled_features = current_latents.squeeze(1)  # (batch, feature_dim)

        # Compute action logits and value
        action_logits = self.action_head(pooled_features)  # (batch, 3)
        value = self.value_head(pooled_features)  # (batch, 1)

        return updated_memory, action_logits, value, pooled_features

    def get_interval(self, pooled_features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute action interval using action embedding (AlphaStar-style).

        Args:
            pooled_features: (batch, feature_dim) - state features
            actions: (batch,) - discrete action indices

        Returns:
            intervals: (batch,) - action intervals in seconds [0.1, 0.5]
        """
        # Get action embeddings
        action_embeds = self.action_embedding(actions)  # (batch, feature_dim)

        # Concatenate state features and action embeddings
        combined = torch.cat([pooled_features, action_embeds], dim=-1)  # (batch, feature_dim * 2)

        # Compute interval: sigmoid [0, 1] → scale to [0, 0.1]
        sigmoid_output = self.interval_head(combined).squeeze(-1)  # (batch,)
        intervals = 0.0 + sigmoid_output * 0.1  # Scale to [0, 0.1] (0ms to 100ms)

        return intervals
    