from Board import Board
import copy

class Game:
    def __init__(self, id, winner, move_string):
        self.id = id
        self.winner = winner
        self.game = self.build_game(move_string)
    
    def build_game(self, move_string):
        moves = []
        for i in range (0, len(move_string), 2):
            moves.append((ord(move_string[i])-97, int(move_string[i+1]) - 1))
        game = []
        board = Board()
        game.append(copy.deepcopy(board.board))
        for i in range(0, len(moves)):
            player = 1 if i%2 == 0 else -1
            board.update(moves[i], player)
            game.append(copy.deepcopy(board.board))
        return game