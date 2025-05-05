import torch
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
        game_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        move_string = ''.join([f"{chr(97+x)}{y+1}" for x, y in moves])
        
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([game_id, winner, move_string])

    def play_game(self, temperature=1.0, cnn_player='both', minimax_depth=None):
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
            
            if minimax_depth is None:
                # Self-play with temperature
                move = predict_move(
                    self.model, 
                    board_tensor,  # Pass the board tensor
                    temperature=temperature,
                    available_moves=list(available.keys())
                )
                if move:  # Convert from (row, col) to (x, y) format
                    move = (move[1], move[0])
            else:
                # Minimax opponent logic
                if (cnn_player == 'both' or 
                    (cnn_player == 'black' and current_player == 'B') or 
                    (cnn_player == 'white' and current_player == 'W')):
                    move = predict_move(
                        self.model,
                        board_tensor,
                        temperature=0.1,
                        available_moves=list(available.keys())
                    )
                    if move:
                        move = (move[1], move[0])
                else:
                    move, _ = self.othello.minimax(
                        self.board, minimax_depth, True, 
                        float('-inf'), float('inf'), 
                        current_player, 
                        'B' if current_player == 'W' else 'W'
                    )
            
            if move is None:
                break
                
            moves.append(move)
            self.board.move(move, current_player, available)
            current_player = 'W' if current_player == 'B' else 'B'
        
        # Game result
        black, white = self.board.get_score()
        winner = 1 if black > white else -1 if white > black else 0
        return winner, moves

    def run(self, num_self_play_games=100):
        # Self-play with varying temperature
        print(f"Generating training data through self-play for {num_self_play_games} games")
        
        for i in range(num_self_play_games):
            # Cycle temperature between 0.5 (more deterministic) and 1.5 (more random)
            temperature = 0.5 + (i % 3) * 0.5  # Values: 0.5, 1.0, 1.5
            winner, moves = self.play_game(temperature=temperature)
            self._save_game_result(winner, moves)
            print(f"Generated {i+1}/{num_self_play_games} self-play games "
                  f"(temperature {temperature:.1f}): Winner is "
                  f"{'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}")

        # Minimax phase
        print("\nGenerating training data against minimax bot (depth 1-10)")
        for depth in range(1, 11):
            for color in ['black', 'white']:
                winner, moves = self.play_game(
                    cnn_player=color,
                    minimax_depth=depth,
                    temperature=0.1  # Low temp for minimax phase
                )
                self._save_game_result(winner, moves)
                print(f"Generated game vs minimax (depth {depth}) CNN as {color}: "
                      f"Winner is {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}")

        print("Completed")
        print(f"Results saved to {self.results_file}")


generator = SelfPlay()
generator.run(num_self_play_games=100)
