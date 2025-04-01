import os
from model import OthelloCNN, predict_move  # Import the model class and predict function
import torch
def load_trained_model(model_path="othello_cnn_model.pth"):
        # Initialize the model with the same architecture
        model = OthelloCNN()
        
        # Load the saved state dictionary
        model.load_state_dict(torch.load(model_path))
        
        # Set to evaluation mode
        model.eval()
        
        return model
class Board:
    def __init__(self, copy=None):
        self.size = 8
        if copy is None:
            self.board = None
        else:
            self.board = [row[:] for row in copy] 
        self.WEIGHTED_BOARD = [
                                [100, -20, 10,  5,  5, 10, -20, 100],
                                [-20, -50, -2, -2, -2, -2, -50, -20],
                                [10,  -2,  5,  1,  1,  5,  -2,  10],
                                [5,   -2,  1,  0,  0,  1,  -2,  5],
                                [5,   -2,  1,  0,  0,  1,  -2,  5],
                                [10,  -2,  5,  1,  1,  5,  -2,  10],
                                [-20, -50, -2, -2, -2, -2, -50, -20],
                                [100, -20, 10,  5,  5, 10, -20, 100]
                            ]
    
    def initialize(self):
        self.board = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        self.board[3][3] = 'W'
        self.board[4][4] = 'W'
        self.board[3][4] = 'B'
        self.board[4][3] = 'B'

    def move(self, move, player, available):
        x = move[0]
        y = move[1]
        self.board[y][x] = player
        for flanked in available[(x,y)]:
            self.board[flanked[1]][flanked[0]] = player

    def apply_move(self, move, player, available):
        new_board = Board(self.board)
        x = move[0]
        y = move[1]
        new_board.board[y][x] = player
        for flanked in available[(x,y)]:
            new_board.board[flanked[1]][flanked[0]] = player

        return new_board

    def find_available_moves(self, player):
        available_moves = {}  

        for y in range(0, 8): 
            for x in range(0, 8):  
                if self.board[y][x] == player:  
                    for ydir in range(-1, 2):  
                        for xdir in range(-1, 2):
                            if xdir == 0 and ydir == 0:
                                continue
                            
                            move, flipped = self.search(x, y, xdir, ydir, player)
                            if move and flipped: 
                                if move not in available_moves:
                                    available_moves[move] = set()
                                available_moves[move].update(flipped)  

        return available_moves
    
    def search(self, x, y, xdir, ydir, player):
        included = []

        if player == 'B':
            other = 'W'
        else:
            other = 'B'

        for dist in range(1, 9):
            xCurr = x + dist * xdir
            yCurr = y + dist * ydir

            
            if xCurr < 0 or xCurr > len(self.board) - 1 or yCurr < 0 or yCurr > len(self.board) - 1:
                return None, []

           
            if self.board[yCurr][xCurr] == other:
                included.append((xCurr, yCurr))
            elif self.board[yCurr][xCurr] == player:
                return None, []  
            else: 
                if included:  
                    return (xCurr, yCurr), included
                return None, []

        return None, []
    
    def print_board(self):
        for row in self.board:
            print(' '.join(row))
    
    def get_score(self):
        black_count = 0
        white_count = 0
        for y in range(0,8):
            for x in range(0,8):
                if self.board[y][x] == 'B':
                    black_count += 1
                elif self.board[y][x] == 'W':
                    white_count += 1
        
        return black_count, white_count
    
    def evaluate(self, color, opp_color):
        score = 0
        for y in range(0,8):
            for x in range(0,8):
                if self.board[x][y] == color:
                    score += self.WEIGHTED_BOARD[y][x]
                elif self.board[x][y] == opp_color:
                    score -= self.WEIGHTED_BOARD[y][x]
        return score
    
class Othello:
    def __init__(self):
        self.player_1 = 'B'
        self.black_score = 0
        self.player_2 = 'W'
        self.white_score = 0
        self.board = Board()
        self.board.initialize()

    def get_choice(self, options, prompt):
        self.clear_console()
        print(prompt)
        for i, option in enumerate(options):
            print(f"{i + 1}. {option}")

        while True:
            try:
                choice = int(input("Enter the number of your choice: "))
                if 1 <= choice <= len(options):
                    return options[choice - 1]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def start(self):
        modes = ["CNN Black", "CNN White"]
        prompt = "Pick a game mode: "
        game_mode = self.get_choice(modes, prompt)
        if game_mode == "CNN Black":
            self.CVC(2, 'B')
        elif game_mode == "CNN White":
            self.CVC(2, 'W')

    def CVC(self, minimax_depth, CNN_move):
        count = 60
        model = load_trained_model()
        while count >= 0:
            color = getattr(self, f"player_{(count%2) + 1}")
            if color == 'W':
                opp_color = 'B'
            else:
                opp_color = 'W'
            available = self.board.find_available_moves(color)
            if color != CNN_move and len(available) > 0:
                self.clear_console()
                self.board.print_board()
                self.print_score()
                print(f'{color} is thinking...')
                move, _ = self.minimax(self.board, minimax_depth, True, float('-inf'), float('inf'), color, opp_color)
                self.board.move(move, color, available)
                count -= 1
            elif color == CNN_move and len(available) > 0:
                temp_board = [[1 if col == 'B' else -1 if col == 'W' else 0 for col in row] for row in self.board.board]
                new_board = torch.tensor(temp_board, dtype=torch.float32)
                move = predict_move(model, new_board, player=1 if CNN_move == 'B' else -1)  # Black's turn
                move = (move[1],move[0])
                self.clear_console()
                self.board.print_board()
                self.print_score()
                self.board.move(move, color, available)
                count -= 1
            else:
                count -= 1
        self.clear_console()
        self.board.print_board()
        self.print_score()
        if self.black_score == self.white_score:
            print("Tie")
        elif self.black_score > self.white_score:
            print("Player 1 wins")
        else:
            print("Player 2 wins")

    def minimax(self, board, depth, is_maximizing, alpha, beta, color, opp_color):
        available = board.find_available_moves(color)
        if depth == 0 or not available:
            return None, board.evaluate(color, opp_color)

        best_move = next(iter(available))

        if is_maximizing:
            max_eval = float('-inf')
            for move in available:
                new_board = board.apply_move(move, color, available)
                _, eval = self.minimax(new_board, depth - 1, False, alpha, beta, color, opp_color)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if alpha >= beta:
                    break
            return best_move, max_eval
        else:
            min_eval = float('inf')
            for move in available:
                new_board = board.apply_move(move, opp_color, available)
                _, eval = self.minimax(new_board, depth - 1, True, alpha, beta, color, opp_color)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if alpha >= beta:
                    break
            return best_move, min_eval

    def clear_console(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_score(self):
        self.black_score, self.white_score = self.board.get_score()
        print(f"Black Score: {self.black_score}\nWhite Score: {self.white_score}")

game = Othello()
game.start()
