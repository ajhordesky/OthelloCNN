import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from model import OthelloCNN, weighted_loss

def load_games_from_csv(file_path):
    data = pd.read_csv(file_path)
    white_games = []
    black_games = []
    tie_games = []

    for _, row in data.iterrows():
        winner = int(row.iloc[1])
        board_states = ast.literal_eval(row.iloc[2])

        game = {
            "winner": winner,
            "boards": [torch.tensor(board, dtype=torch.float32).unsqueeze(0) for board in board_states]
        }
        if winner == 1:
            black_games.append(game)
        elif winner == -1:
            white_games.append(game)
        else:
            tie_games.append(game)
    
    games = black_games[0:5000] + white_games[0:5000] + tie_games[0:500]

    return games

class OthelloDataset(Dataset):
    def __init__(self, games):
        self.data = []
        
        for game in games:
            winner = game['winner'] 
            for i, board in enumerate(game['boards'][:-1]):
                next_board = game['boards'][i + 1]
                board_tensor = board.clone().detach()
                next_board_tensor = next_board.clone().detach()

                
                player = 1 if i % 2 == 0 else -1 

                move_target = (next_board_tensor != board_tensor).nonzero(as_tuple=True)
                if move_target[0].nelement() > 0:
                    move_index = move_target[1][0] * 8 + move_target[2][0]
                    player_layer = torch.full_like(board_tensor, player) 
                    input_tensor = torch.cat([board_tensor, player_layer], dim=0)
                    self.data.append((input_tensor, move_index, winner))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


file_path = 'cleaned_games.csv'
games = load_games_from_csv(file_path)
dataset = OthelloDataset(games)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print("Data set up")


model = OthelloCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Model set up")


epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for input_tensor, move, winner in train_loader:
        optimizer.zero_grad()

        player_color = input_tensor[:, 1, 0, 0]

        output = model(input_tensor)
        
        loss = weighted_loss(output, move, winner, player_color)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")


torch.save(model.state_dict(), 'othello_cnn_model.pth')