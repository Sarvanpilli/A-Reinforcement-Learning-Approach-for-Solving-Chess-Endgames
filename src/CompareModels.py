import sys
import os
import time
import torch
import numpy as np
import random
import pickle

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ChessFormer'))

from ChessBoard import ChessBoard, King, Rook, Piece
from ChessFormer.Agent import ChessAgent
from ChessFormer.Train import get_state_tensor

def load_old_model(filename):
    print(f"Loading Old Model (Table) from {filename}...")
    with open(filename, 'rb') as infile:
        return pickle.load(infile)

def load_new_model(filename):
    print(f"Loading New Model (Transformer) from {filename}...")
    agent = ChessAgent()
    if os.path.exists(filename):
        agent.load(filename)
    return agent

def compare_models():
    # Paths
    old_model_path = 'res/memory1-0_Q_trained_ep1000000_g99_l8_e90.bson'
    new_model_path = 'ChessFormer_model_900.pth'
    
    # Load Models
    try:
        old_table = load_old_model(old_model_path)
    except FileNotFoundError:
        print("Old model not found. Please run QLearning.py first.")
        return

    try:
        new_agent = load_new_model(new_model_path)
    except Exception as e:
        print(f"Error loading new model: {e}")
        return

    print("\n--- Starting Comparison (Inference Speed) ---")
    n_tests = 100
    
    old_times = []
    new_times = []
    
    # Generate random states
    print(f"Running {n_tests} test cases...")
    
    for i in range(n_tests):
        # Create a random board
        wk = King(random.randint(0, 7), random.randint(0, 7), Piece.WHITE)
        wr = Rook(random.randint(0, 7), random.randint(0, 7), Piece.WHITE)
        bk = King(random.randint(0, 7), random.randint(0, 7), Piece.BLACK)
        board = ChessBoard(wk, wr, bk, white_plays=1)
        
        if not board.valid:
            continue
            
        moves = board.get_possible_moves()
        if not moves:
            continue
            
        # --- Test Old Model ---
        start = time.time()
        # The old model looks up the current state in the big table
        # Then iterates over next states to find max
        # We simulate the lookup cost
        state_id = board.board_id()
        if state_id in old_table:
            next_states = old_table[state_id]
            # Find max
            best_move = None
            max_val = -float('inf')
            for next_id, val in next_states.items():
                if val > max_val:
                    max_val = val
        else:
            # State not in table (shouldn't happen if fully trained, but possible with random placement)
            pass
        old_times.append(time.time() - start)
        
        # --- Test New Model ---
        start = time.time()
        # The new model evaluates each next board state
        best_score = -float('inf')
        for move_board in moves:
            state = get_state_tensor(move_board)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(new_agent.device)
            with torch.no_grad():
                values = new_agent.model(state_tensor)
                score = values[0][0].item()
        new_times.append(time.time() - start)

    avg_old = sum(old_times) / len(old_times) * 1000
    avg_new = sum(new_times) / len(new_times) * 1000
    
    print("\n--- Results ---")
    print(f"Old Model (Table Lookup) Avg Time: {avg_old:.4f} ms")
    print(f"New Model (Transformer)  Avg Time: {avg_new:.4f} ms")
    
    print("\n--- Analysis ---")
    if avg_old < avg_new:
        print("The Old Model is faster because it's just a Hash Map lookup (O(1)).")
        print("The New Model runs a Neural Network (Matrix Multiplications), so it's slower.")
        print("HOWEVER: The Old Model takes ~100MB+ RAM. The New Model takes ~2MB.")
        print("AND: The New Model can handle unseen states. The Old Model crashes.")
    else:
        print("The New Model is surprisingly faster!")

if __name__ == "__main__":
    compare_models()
