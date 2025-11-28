import torch
import torch.nn as nn
import math

class ChessTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, num_actions=32):
        super(ChessTransformer, self).__init__()
        
        # Input: [Piece Type (0-2), Color (0-1), Rank (0-7), File (0-7)]
        # We project this 4-dim vector to d_model
        self.embedding = nn.Linear(4, d_model)
        
        # Positional encoding is less critical here as we have explicit rank/file, 
        # but standard transformers use it. We'll skip it for now as our input 
        # explicitly contains "position" (Rank/File).
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output head: Predict Q-values for all possible moves
        # We'll need to map actions to indices. 
        # For now, let's assume a fixed action space size.
        self.fc_out = nn.Linear(d_model, num_actions)

    def forward(self, x):
        # x shape: (batch_size, num_pieces, 4)
        
        # 1. Embed
        x = self.embedding(x.float()) # (batch, num_pieces, d_model)
        
        # 2. Transform
        # We want to get the attention weights for visualization later
        # But nn.TransformerEncoder doesn't return them easily.
        # For the forward pass, we just need the output.
        x = self.transformer_encoder(x) # (batch, num_pieces, d_model)
        
        # 3. Aggregate
        # We can take the mean of the piece embeddings to get a board embedding
        # Or just take the first piece (e.g. White King) as the "CLS" token.
        # Let's try max pooling to capture the most salient features.
        x = torch.max(x, dim=1)[0] # (batch, d_model)
        
        # 4. Predict Q-values
        q_values = self.fc_out(x) # (batch, num_actions)
        
        return q_values

    def get_attention_weights(self, x):
        # Helper to extract attention weights for visualization
        # We'll need to manually run the layers to capture weights
        weights = []
        x = self.embedding(x.float())
        
        for layer in self.transformer_encoder.layers:
            # self_attn returns (attn_output, attn_output_weights) if need_weights=True
            _, attn_weights = layer.self_attn(x, x, x, need_weights=True)
            weights.append(attn_weights)
            x = layer.forward(x) # This is a bit simplified, actual layer has residual + norm
            
        return weights
