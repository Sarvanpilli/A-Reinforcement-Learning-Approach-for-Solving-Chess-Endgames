import sys
import os
import torch
import numpy as np
import random

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ChessBoard import ChessBoard, King, Rook, Piece
from ChessFormer.Agent import ChessAgent
from ChessFormer.Train import get_state_tensor

def play_game():
    agent = ChessAgent()
    model_path = 'ChessFormer_model_900.pth'
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        agent.load(model_path)
    else:
        print("Model not found! Playing with random weights.")

    # Initialize board
    wk = King(0, 4, Piece.WHITE)
    wr = Rook(0, 0, Piece.WHITE)
    bk = King(7, 4, Piece.BLACK)
    
    board = ChessBoard(wk, wr, bk, white_plays=1)
    
    print("Starting Game!")
    print("You are watching the AI play against itself (or random opponent logic).")
    
    steps = 0
    max_steps = 50
    
    while steps < max_steps:
        print(f"\n--- Turn {steps + 1} ---")
        board.draw()
        
        if board.state == ChessBoard.BLACK_KING_CHECKMATE:
            print("CHECKMATE! White Wins!")
            break
        if board.state == ChessBoard.DRAW:
            print("DRAW!")
            break
            
        # AI Turn (White)
        if board.turn == Piece.WHITE:
            print("White (AI) is thinking...")
            moves = board.get_possible_moves()
            
            if not moves:
                print("Stalemate/Checkmate!")
                break
                
            # Evaluate all possible moves
            best_score = -float('inf')
            best_move = None
            
            for move_board in moves:
                state = get_state_tensor(move_board)
                # We use the agent to evaluate the state
                # Note: Our current agent outputs 32 values. 
                # For this prototype, let's assume output 0 is the "Value" of the state.
                # In a real Value Network, output would be size 1.
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    values = agent.model(state_tensor)
                    score = values[0][0].item() # Take first output as "Value"
                
                if score > best_score:
                    best_score = score
                    best_move = move_board
            
            print(f"AI chose move with score: {best_score:.4f}")
            board = best_move
            
        else:
            # Black Turn (Random for now)
            print("Black (Random) is moving...")
            moves = board.get_possible_moves()
            if not moves:
                print("Checkmate! White Wins!")
                break
            board = moves[random.randint(0, len(moves) - 1)]
            
        steps += 1
        # input("Press Enter to continue...") # Uncomment for step-by-step

if __name__ == "__main__":
    play_game()
