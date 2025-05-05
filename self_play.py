import torch
import random
import csv
import os
from datetime import datetime
from sptools import Board, Othello
from model import predict_move, OthelloCNN

class SelfPlay:
    def __init__(self):
        self.board = Board()
        self.othello = Othello()
        self.device = "cpu"
        
        # Initialize model
        self.model = OthelloCNN().to(self.device)
        try:
            self.model.load_state_dict(torch.load('othello_cnn.pth'))
            print("Loaded existing model")
        except:
            print("Initializing new model")
        
        # File setup
        self.results_file = 'self_play_results.csv'
        self._init_results_file()

    def _init_results_file(self):
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['game_id', 'winner', 'game_moves'])

    def _save_game_result(self, winner, moves):
        # Save game result to CSV file
        game_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        # Convert moves
        move_string = ''.join([f"{chr(97+x)}{y+1}" for x, y in moves])
        
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([game_id, winner, move_string])

    def get_random_move(self, available_moves):
        # Random move function
        return random.choice(list(available_moves.keys())) if available_moves else None

    def play_game(self, opponent_type='random', cnn_player='both', minimax_depth=3):
        # Play game against specified opponent
        self.board.initialize()
        moves = []
        current_player = 'B'
        
        for _ in range(60):  # Max moves in Othello
            available = self.board.find_available_moves(current_player)
            if not available:
                current_player = 'W' if current_player == 'B' else 'B'
                available = self.board.find_available_moves(current_player)
                if not available:
                    break  # Game over
            
            # Convert board to PyTorch tensor
            board_tensor = torch.tensor(
                [[1 if cell == 'B' else -1 if cell == 'W' else 0 for cell in row] 
                 for row in self.board.board],
                dtype=torch.float32, device=self.device
            )
            player_tensor = 1 if current_player == 'B' else -1
            
            if opponent_type == 'random':
                move = self.get_random_move(available)
            elif opponent_type == 'minimax':
                if (cnn_player == 'both' or 
                    (cnn_player == 'black' and current_player == 'B') or 
                    (cnn_player == 'white' and current_player == 'W')):
                    # CNN move
                    move = predict_move(self.model, board_tensor, player_tensor)
                    if move:  # Convert from (row, col) to (x, y) format
                        move = (move[1], move[0])
                else:
                    # Minimax move
                    move, _ = self.othello.minimax(
                        self.board, minimax_depth, True, 
                        float('-inf'), float('inf'), 
                        current_player, 
                        'B' if current_player == 'W' else 'W'
                    )
            else:
                raise ValueError("Invalid opponent type")
            
            if move is None:
                break
                
            moves.append(move)
            self.board.move(move, current_player, available)
            current_player = 'W' if current_player == 'B' else 'B'
        
        # Specify winner
        black, white = self.board.get_score()
        winner = 1 if black > white else -1 if white > black else 0
        return winner, moves

    def run(self, num_random_games=100):
        # Data generation
        print(f"Generating training data against random bot for {num_random_games} games")
        
        # Play against random move, alternating between black and white
        print(f"Playing {num_random_games} random games...")
        for i in range(num_random_games):
            # Alternate colors for proper coverage
            color = 'black' if i % 2 == 0 else 'white'
            winner, moves = self.play_game(opponent_type='random', cnn_player=color)
            self._save_game_result(winner, moves)
            print(f"Generated {i+1}/{num_random_games} random games "
                    f"(CNN playing {color}): Winner is {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}")

        # Play against minimax, alternating colors and increasing ply depth
        print("\nGenerating training data against minimax bot with increasing depth ranging 1 - 10")
        for depth in range(1, 11):
            for color in ['black', 'white']:
                winner, moves = self.play_game(
                    opponent_type='minimax',
                    cnn_player=color,
                    minimax_depth=depth
                )
                self._save_game_result(winner, moves)
                print(f"Generated game vs minimax (depth {depth}) CNN playing as {color}: "
                      f"Winner is {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}")

        print("Completed")
        print(f"Results saved to {self.results_file}")


generator = SelfPlay()
generator.run(num_random_games=100)