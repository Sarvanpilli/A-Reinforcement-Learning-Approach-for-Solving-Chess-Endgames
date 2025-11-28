import sys
import os
import numpy as np
import torch

# Add parent directory to path to import ChessBoard
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ChessBoard import ChessBoard, King, Rook, Piece
from ChessFormer.Agent import ChessAgent

def get_state_tensor(board):
    # Convert board state to tensor [PieceType, Color, Rank, File]
    # PieceType: 0=King, 1=Rook
    # Color: 0=Black, 1=White
    
    pieces_data = []
    
    # Find pieces in board.pieces
    wk = next((p for p in board.pieces if isinstance(p, King) and p.color == Piece.WHITE), None)
    wr = next((p for p in board.pieces if isinstance(p, Rook) and p.color == Piece.WHITE), None)
    bk = next((p for p in board.pieces if isinstance(p, King) and p.color == Piece.BLACK), None)
    
    # White King
    if wk: pieces_data.append([0, 1, wk.row, wk.col])
    else: pieces_data.append([0, 1, -1, -1]) # Should not happen in valid state
        
    # White Rook
    if wr: pieces_data.append([1, 1, wr.row, wr.col])
    else: pieces_data.append([1, 1, -1, -1]) # Captured?
        
    # Black King
    if bk: pieces_data.append([0, 0, bk.row, bk.col])
    else: pieces_data.append([0, 0, -1, -1])
    
    return np.array(pieces_data)

def train():
    agent = ChessAgent()
    episodes = 1000
    batch_size = 32
    
    print("Starting training...")
    
    for e in range(episodes):
        # Initialize board (random state)
        # For simplicity, let's just place pieces randomly for now
        # Ideally we'd use BaseParams to get valid states
        wk = King(np.random.randint(8), np.random.randint(8), Piece.WHITE)
        wr = Rook(np.random.randint(8), np.random.randint(8), Piece.WHITE)
        bk = King(np.random.randint(8), np.random.randint(8), Piece.BLACK)
        
        board = ChessBoard(wk, wr, bk, white_plays=1)
        state = get_state_tensor(board)
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 50:
            # Action selection (simplified: just picking random valid moves for now)
            # In a real implementation, we'd map agent output index to a specific move
            # For this prototype, let's just verify the loop works
            
            action_idx = agent.act(state)
            
            # Simulate a step (random valid move for now as we haven't mapped actions)
            moves = board.get_possible_moves()
            if not moves:
                done = True
                reward = -1 # Stalemate or Checkmate
            else:
                next_board = moves[np.random.randint(len(moves))]
                next_state = get_state_tensor(next_board)
                
                # Reward function (simplified)
                if next_board.state == ChessBoard.BLACK_KING_CHECKMATE:
                    reward = 100
                    done = True
                else:
                    reward = -0.1
                
                agent.remember(state, action_idx, reward, next_state, done)
                state = next_state
                board = next_board
                total_reward += reward
                steps += 1
                
            if done:
                print(f"Episode: {e}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
        if e % 100 == 0:
            agent.save(f"ChessFormer_model_{e}.pth")

if __name__ == "__main__":
    train()
