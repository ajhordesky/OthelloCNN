class Board:
    def __init__(self):
        self.size = 8
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.board[3][3] = -1
        self.board[3][4] = 1
        self.board[4][3] = 1
        self.board[4][4] = -1

    def update(self, move, player):
        self.board[move[1]][move[0]] = player