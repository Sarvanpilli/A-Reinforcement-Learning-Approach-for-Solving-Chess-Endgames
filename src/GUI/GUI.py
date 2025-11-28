import os
import cocos
import random
import pickle
import pyglet
from cocos.director import director
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ChessBoard import ChessBoard,King,Rook,Piece


class Game(cocos.layer.Layer):

    is_event_handler = True

    def __init__(self, file_name):
        super(Game, self).__init__()
        self.init_board()
        self.file_name = file_name
        self.R = self.load(file_name)
        self.state_pool = [state for state in self.R.keys() if state[6] == 1]
        self.finish = 0
        self.max_moves = 40
        self.white_moves = 0
        self.result_box = None
        self.result_label = None

        label = cocos.text.Label('Round:',
                                  font_name='Times New Roman',
                                  font_size=26,
                                  anchor_x='left', anchor_y='bottom')
        label.position = 0, 0
        self.add(label,z=4)

        label = cocos.text.Label('Player:',
                                  font_name='Times New Roman',
                                  font_size=26,
                                  anchor_x='left', anchor_y='bottom')
        label.position = 64*4, 0
        self.add(label,z=4)


        asset_dir = os.path.dirname(__file__)

        def load_image(name):
            path = os.path.join(asset_dir, name)
            return pyglet.image.load(path)

        self.wking = cocos.sprite.Sprite(load_image('wking.gif'))
        self.wking.position = self.convert_pos(0,0)
        self.wking.scale = 0.1
        self.add(self.wking, z=1)

        self.wrook = cocos.sprite.Sprite(load_image('wrook.gif'))
        self.wrook.position = self.convert_pos(0,2)
        self.wrook.scale = 0.1
        self.add(self.wrook, z=2)

        self.bking = cocos.sprite.Sprite(load_image('bking.gif'))
        self.bking.position = self.convert_pos(5,4)
        self.bking.scale = 0.1
        self.add(self.bking, z=3)

        self.next_state_id = None

        self.who_plays = cocos.text.Label('WHITE',
                                  font_name='Times New Roman',
                                  font_size=32,
                                  anchor_x='right', anchor_y='bottom')
        self.who_plays.position = 64*8, 0
        self.add(self.who_plays,z=4)

        self.round = cocos.text.Label('0',
                                  font_name='Times New Roman',
                                  font_size=32,
                                  anchor_x='left', anchor_y='bottom')
        self.round.position = 64*2, 0
        self.add(self.round, z=4)

        self.start_new_episode()

    def on_key_press(self, key, modifiers):
        if self.finish == 1:
            self.start_new_episode()
            return

        if self.next_state_id is None:
            return

        self.current_state_id = self.next_state_id
        self.set_position(self.current_state_id)
        self.play()

    def play(self):
        if self.current_state_id[6] == 1:
            self.white_moves += 1
            self.round.element.text = str(self.white_moves)
            self.who_plays.element.text = 'WHITE'
        else:
            self.who_plays.element.text = 'BLACK'

        board = self.get_board(self.current_state_id)

        if board.state is ChessBoard.BLACK_KING_CHECKMATE:
            self._show_checkmate()
            return

        if board.state is ChessBoard.DRAW or self.white_moves >= self.max_moves:
            self._show_draw()
            return

        next_states = self.R[self.current_state_id]

        if self.current_state_id[6] == 0:
            self.next_state_id = self.get_min_state(next_states)
        else:
            self.next_state_id  = self.get_max_state(next_states)

        if self.next_state_id is None:
            print('No valid next move from state:', self.current_state_id)
            self.start_new_episode()
            return

        for i in next_states:
            print (i,'->', next_states[i])
        print('MaxMin:',  self.next_state_id ,'->',next_states[self.next_state_id ])

    def init_board(self):
        y = 64
        for i in range(8):
            x = 0
            for j in range(8):
                if j % 2 == 0:
                    if i % 2 == 0:
                        box = cocos.layer.ColorLayer(255, 215, 97, 255, width=64, height=64)
                    else:
                        box = cocos.layer.ColorLayer(128, 64, 0, 255, width=64, height=64)
                else:
                    if i % 2 == 0:
                       box = cocos.layer.ColorLayer(128, 64, 0, 255, width=64, height=64)
                    else:
                        box = cocos.layer.ColorLayer(255, 215, 97, 255, width=64, height=64)

                box.position = x, y
                self.add(box)
                x += 64
            y += 64

    def set_position(self,state):
        self.wking.position = self.convert_pos(state[0],state[1])
        self.wrook.position = self.convert_pos(state[2],state[3])
        self.bking.position = self.convert_pos(state[4],state[5])
        self.white_plays = state[6]
        pass

    def convert_pos(self,row,col):
        return col * 64 + 32, 64*8 - row * 64 - 32 +64



    def get_board(self,state_id):
         wk_r, wk_c, wr_r, wr_c, bk_r, bk_c, white_plays = state_id
         return ChessBoard(wk=King(wk_r, wk_c, Piece.WHITE),
                            wr=Rook(wr_r, wr_c, Piece.WHITE),
                            bk=King(bk_r, bk_c, Piece.BLACK),
                            white_plays=white_plays,
                            debug=True
                            )
    @staticmethod
    def get_max_state(states):
        if not states:
            return None

        max_state = None
        max_q = None

        for state, value in states.items():
            if max_q is None or value > max_q:
                max_q = value
                max_state = state

        return max_state

    @staticmethod
    def get_min_state(states):
        if not states:
            return None

        min_state = None
        min_q = None

        for state, value in states.items():
            if min_q is None or value < min_q:
                min_q= value
                min_state = state

        return min_state

    def start_new_episode(self):
        self._clear_result_overlay()
        self.bking.z = 3
        self.white_moves = 0
        self.finish = 0
        self.next_state_id = None

        state = self._find_checkmating_state()
        if state is None:
            raise RuntimeError('Unable to find a checkmating line in the provided memory file.')

        self.current_state_id = state
        self.set_position(self.current_state_id)
        self.board = self.get_board(self.current_state_id)
        self.round.element.text = '0'
        self._update_player_label()
        self.play()

    def _update_player_label(self):
        if self.white_plays == 1:
            self.who_plays.element.text = 'WHITE'
        else:
            self.who_plays.element.text = 'BLACK'

    def _show_checkmate(self):
        self.set_position(self.current_state_id)
        self.bking.z = 0
        self.finish = 1

        self._clear_result_overlay()
        self.result_box = cocos.layer.ColorLayer(255, 255, 255, 200, width=64*8, height=72)
        self.result_box.position = 0, 64*8
        self.add(self.result_box, z=3)

        self.result_label = cocos.text.Label('CHECKMATE',
                              font_name='Comic Sans MS',
                              font_size=52,
                              color=(90,135,161, 255),
                              anchor_x='center', anchor_y='top')
        self.result_label.position = 64*4, 64*9
        self.add(self.result_label,z=4)

    def _show_draw(self):
        self.set_position(self.current_state_id)
        self.finish = 1

        self._clear_result_overlay()
        self.result_box = cocos.layer.ColorLayer(255, 255, 255, 200, width=64*8, height=72)
        self.result_box.position = 0, 64*8
        self.add(self.result_box, z=3)

        self.result_label = cocos.text.Label('DRAW',
                              font_name='Comic Sans MS',
                              font_size=52,
                              color=(90,135,161, 255),
                              anchor_x='center', anchor_y='top')
        self.result_label.position = 64*4, 64*9
        self.add(self.result_label,z=4)

    def _clear_result_overlay(self):
        if self.result_label is not None:
            self.remove(self.result_label)
            self.result_label = None
        if self.result_box is not None:
            self.remove(self.result_box)
            self.result_box = None

    def _find_checkmating_state(self):
        candidates = list(self.state_pool)
        random.shuffle(candidates)

        for state in candidates:
            if self._reaches_checkmate(state):
                return state

        return None

    def _reaches_checkmate(self, start_state):
        state = start_state
        visited = set()
        moves = 0

        while moves < self.max_moves:
            board = self.get_board(state)
            if board.state is ChessBoard.BLACK_KING_CHECKMATE:
                return True
            if board.state is ChessBoard.DRAW:
                return False

            next_states = self.R.get(state, {})
            if not next_states:
                return board.state is ChessBoard.BLACK_KING_CHECKMATE

            if state[6] == 1:
                nxt = self.get_max_state(next_states)
            else:
                nxt = self.get_min_state(next_states)

            if nxt is None or nxt in visited:
                return False

            visited.add(state)
            state = nxt

            if state[6] == 1:
                moves += 1

        return False

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as infile:
            params = pickle.load(infile)
            return params

if __name__ == '__main__':
    fp = 'res/memory1-0_Q_trained_ep1000000_g99_l8_e90.bson'

    director.init(width=64*8, height=64*9, caption="Chess Game Engine",resizable=False)
    director.run(cocos.scene.Scene(Game(fp)))
