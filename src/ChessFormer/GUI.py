import os
import cocos
import random
import pyglet
from cocos.director import director
import sys
import torch
import numpy as np
from cocos.draw import Line

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ChessBoard import ChessBoard, King, Rook, Piece
from ChessFormer.Agent import ChessAgent
from ChessFormer.Train import get_state_tensor

class Game(cocos.layer.Layer):
    is_event_handler = True

    def __init__(self):
        super(Game, self).__init__()
        self.init_board()
        
        # Load Agent
        self.agent = ChessAgent()
        model_path = os.path.join(os.path.dirname(__file__), 'ChessFormer_model_900.pth')
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.agent.load(model_path)
        else:
            print("Model not found, using random weights")

        self.board = None
        self.wk_sprite = None
        self.wr_sprite = None
        self.bk_sprite = None
        
        self.attention_lines = []
        
        # Labels
        self.info_label = cocos.text.Label('Press SPACE to move',
                                  font_name='Times New Roman',
                                  font_size=20,
                                  anchor_x='center', anchor_y='bottom')
        self.info_label.position = 64*4, 10
        self.add(self.info_label, z=4)

        self.start_new_episode()

    def init_board(self):
        # Path to GUI assets (now local)
        asset_dir = os.path.dirname(os.path.abspath(__file__))
        sprite = cocos.sprite.Sprite(pyglet.image.load(os.path.join(asset_dir, 'chessboard.gif')))
        sprite.position = 64*4, 64*4
        sprite.scale = 1
        self.add(sprite, z=0)

    def convert_pos(self, row, col):
        return 64*col + 32, 64*row + 32

    def start_new_episode(self):
        # Initialize random board
        wk = King(random.randint(0, 7), random.randint(0, 7), Piece.WHITE)
        wr = Rook(random.randint(0, 7), random.randint(0, 7), Piece.WHITE)
        bk = King(random.randint(0, 7), random.randint(0, 7), Piece.BLACK)
        self.board = ChessBoard(wk, wr, bk, white_plays=1)
        
        self.update_sprites()
        self.draw_attention()

    def update_sprites(self):
        # Remove old sprites
        if self.wk_sprite: self.remove(self.wk_sprite)
        if self.wr_sprite: self.remove(self.wr_sprite)
        if self.bk_sprite: self.remove(self.bk_sprite)
        
        # Load images (now local)
        asset_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Find pieces
        wk = next((p for p in self.board.pieces if isinstance(p, King) and p.color == Piece.WHITE), None)
        wr = next((p for p in self.board.pieces if isinstance(p, Rook) and p.color == Piece.WHITE), None)
        bk = next((p for p in self.board.pieces if isinstance(p, King) and p.color == Piece.BLACK), None)

        if wk:
            self.wk_sprite = cocos.sprite.Sprite(pyglet.image.load(os.path.join(asset_dir, 'wking.gif')))
            self.wk_sprite.position = self.convert_pos(wk.row, wk.col)
            self.wk_sprite.scale = 0.1
            self.add(self.wk_sprite, z=2)

        if wr:
            self.wr_sprite = cocos.sprite.Sprite(pyglet.image.load(os.path.join(asset_dir, 'wrook.gif')))
            self.wr_sprite.position = self.convert_pos(wr.row, wr.col)
            self.wr_sprite.scale = 0.1
            self.add(self.wr_sprite, z=2)

        if bk:
            self.bk_sprite = cocos.sprite.Sprite(pyglet.image.load(os.path.join(asset_dir, 'bking.gif')))
            self.bk_sprite.position = self.convert_pos(bk.row, bk.col)
            self.bk_sprite.scale = 0.1
            self.add(self.bk_sprite, z=2)

    def draw_attention(self):
        # Clear old lines
        for line in self.attention_lines:
            self.remove(line)
        self.attention_lines = []
        
        # Get attention weights
        state = get_state_tensor(self.board)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
        weights = self.agent.model.get_attention_weights(state_tensor)
        
        # Layer 0, Head 0
        attn_map = weights[0][0, 0].detach().cpu().numpy() # Shape (3, 3) or (3,)
        
        if len(attn_map.shape) == 1:
             attn_map = attn_map.reshape(1, -1)
             
        # Pieces: 0=WK, 1=WR, 2=BK (Order matters! Must match get_state_tensor)
        wk = next((p for p in self.board.pieces if isinstance(p, King) and p.color == Piece.WHITE), None)
        wr = next((p for p in self.board.pieces if isinstance(p, Rook) and p.color == Piece.WHITE), None)
        bk = next((p for p in self.board.pieces if isinstance(p, King) and p.color == Piece.BLACK), None)
        
        pieces = [wk, wr, bk]
        
        # Draw lines based on attention
        # We only care about strong attention (> 0.1)
        # We draw lines FROM source TO target
        
        # If attn_map is (1, 3), it means Source is "Global" or "CLS" token?
        # Actually in our model we didn't use a CLS token, we just passed 3 pieces.
        # So attn_map (3, 3) means:
        # Row 0: WK attending to [WK, WR, BK]
        # Row 1: WR attending to [WK, WR, BK]
        # Row 2: BK attending to [WK, WR, BK]
        
        # If it collapsed to (1, 3), it might be because we took max pooling in forward pass?
        # No, get_attention_weights returns the raw weights from the layer.
        # It should be (batch, num_heads, seq_len, seq_len).
        
        # Let's assume it is (3, 3) for now. If (1, 3), we treat row 0 as the only source.
        
        threshold = 0.2
        
        for i in range(attn_map.shape[0]): # Source
            for j in range(attn_map.shape[1]): # Target
                weight = attn_map[i, j]
                if pieces[i] and pieces[j] and weight > threshold and i != j: # Don't draw self-attention and check existence
                    start_pos = self.convert_pos(pieces[i].row, pieces[i].col)
                    end_pos = self.convert_pos(pieces[j].row, pieces[j].col)
                    
                    # Draw line
                    # Color: Yellow, Opacity based on weight
                    opacity = int(min(255, weight * 255 * 2)) # Boost visibility
                    line = Line(start_pos, end_pos, (255, 255, 0, opacity), stroke_width=3)
                    self.add(line, z=1)
                    self.attention_lines.append(line)

    def on_key_press(self, key, modifiers):
        if key == pyglet.window.key.SPACE:
            self.play_move()
        elif key == pyglet.window.key.R:
            self.start_new_episode()

    def play_move(self):
        if self.board.state != ChessBoard.NOTHING:
            self.info_label.element.text = f"Game Over: {self.board.state}. Press R to restart."
            return

        if self.board.turn == Piece.WHITE:
            # AI Move
            moves = self.board.get_possible_moves()
            if not moves:
                self.info_label.element.text = "Stalemate/Checkmate!"
                return
                
            best_score = -float('inf')
            best_move = None
            
            for move_board in moves:
                state = get_state_tensor(move_board)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                with torch.no_grad():
                    values = self.agent.model(state_tensor)
                    score = values[0][0].item()
                
                if score > best_score:
                    best_score = score
                    best_move = move_board
            
            self.board = best_move
            self.info_label.element.text = f"AI Score: {best_score:.2f}"
            
        else:
            # Black Move (Random)
            moves = self.board.get_possible_moves()
            if not moves:
                self.info_label.element.text = "Checkmate! White Wins!"
                return
            self.board = moves[random.randint(0, len(moves) - 1)]
            self.info_label.element.text = "Black moved randomly."
            
        self.update_sprites()
        self.draw_attention()

if __name__ == "__main__":
    director.init(width=64*8, height=64*8, caption="ChessFormer Visualization")
    director.run(cocos.scene.Scene(Game()))
