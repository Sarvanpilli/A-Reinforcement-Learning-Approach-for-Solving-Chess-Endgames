import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ChessBoard import ChessBoard, King, Rook, Piece
from ChessFormer.Agent import ChessAgent
from ChessFormer.Train import get_state_tensor

def visualize_attention():
    # Load model
    agent = ChessAgent()
    # In a real scenario, we'd load weights here:
    # agent.load('ChessFormer_model_100.pth')
    
    # Create a sample board state
    # White King at E1 (0, 4)
    # White Rook at A1 (0, 0)
    # Black King at E8 (7, 4)
    wk = King(0, 4, Piece.WHITE)
    wr = Rook(0, 0, Piece.WHITE)
    bk = King(7, 4, Piece.BLACK)
    
    board = ChessBoard(wk, wr, bk, white_plays=1)
    state = get_state_tensor(board)
    
    # Get attention weights
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    
    # We need to access the model's internal attention weights
    # The current Model.py has a helper for this
    weights = agent.model.get_attention_weights(state_tensor)
    
    # Visualize the first layer's attention
    # weights[0] shape: (batch, num_heads, seq_len, seq_len)
    # seq_len is 3 (WK, WR, BK)
    
    # Reshape to (3, 3) or (1, 3) depending on what we want to visualize.
    # The error suggests attn_map is (3,). This means we likely grabbed the wrong dimension or the model output is different.
    # Let's inspect weights[0] shape again. It should be (batch, num_heads, seq_len, seq_len).
    # If it is (3,), then we might have indexed it wrong.
    
    # weights[0] is from layer 0.
    # It is a tuple (attn_output, attn_output_weights).
    # We appended `attn_weights` which is (batch, num_heads, target_len, source_len) -> (1, 4, 3, 3)
    
    # Wait, in Model.py:
    # _, attn_weights = layer.self_attn(x, x, x, need_weights=True)
    # x shape is (batch, seq_len, d_model) -> (1, 3, 64)
    # So attn_weights should be (batch, num_heads, seq_len, seq_len) -> (1, 4, 3, 3)
    
    # However, the print output says: [0.20850606 0.35665187 0.5459533 ]
    # This is a 1D array of size 3.
    # This implies attn_map = weights[0][0, 0] is picking a single row?
    
    # Let's force it to be 2D (3, 3) to see the full relationship.
    # weights[0] is the first layer's attention weights.
    # It should be a Tensor.
    
    # Let's try to get the full (3, 3) matrix for Head 0.
    attn_map = weights[0][0, 0].detach().cpu().numpy() # Should be (3, 3)
    
    if len(attn_map.shape) == 1:
        # If it somehow collapsed, reshape or debug.
        # But wait, the error says "got (3, 1) and (3,)".
        # This usually happens when labels don't match data.
        # If data is (3,), it's 1D. Heatmap needs 2D.
        attn_map = attn_map.reshape(1, -1) # Make it (1, 3)
        yticklabels = False # Disable Y labels if 1D
    else:
        yticklabels = labels

    print("Attention Map Shape:", attn_map.shape)
    
    # Plot
    labels = ['WK', 'WR', 'BK']
    plt.figure(figsize=(6, 5))
    sns.heatmap(attn_map, xticklabels=labels, yticklabels=yticklabels, annot=True, cmap='viridis')
    plt.title("ChessFormer Attention Map")
    plt.xlabel("Target Piece")
    plt.ylabel("Source Piece")
    
    # Save plot
    output_path = os.path.join(os.path.dirname(__file__), 'attention_map.png')
    plt.savefig(output_path)
    print(f"Saved attention map to {output_path}")

if __name__ == "__main__":
    visualize_attention()
