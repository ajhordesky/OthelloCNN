import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from model import OthelloCNN, weighted_loss
import time

def load_games_from_csv(file_path):
    data = pd.read_csv(file_path, nrows=5000)
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

    games = black_games + white_games + tie_games

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

print("Started loading data...")
file_path = 'cleaned_self_play.csv'
games = load_games_from_csv(file_path)


train_size = int(0.8 * len(games))
val_size = len(games) - train_size
train_games, val_games = torch.utils.data.random_split(games, [train_size, val_size])

train_dataset = OthelloDataset(train_games)
val_dataset = OthelloDataset(val_games)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print("Data set up complete")

model = OthelloCNN()
model.load_state_dict(torch.load('othello_cnn_final_model.pth'))
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Model set up complete")


best_val_loss = float('inf')
epochs_no_improve = 0
max_epochs_no_improve = 3
min_delta = 0.001

epochs = 100
start_time = time.perf_counter()

try:
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        # Training phase
        for input_tensor, move, winner in train_loader:
            optimizer.zero_grad()
            player_color = input_tensor[:, 1, 0, 0]
            output = model(input_tensor)
            loss = weighted_loss(output, move, winner, player_color)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for input_tensor, move, winner in val_loader:
                player_color = input_tensor[:, 1, 0, 0]
                output = model(input_tensor)
                loss = weighted_loss(output, move, winner, player_color)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
       
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'othello_cnn_best_model.pth')
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")
            
           
            if epochs_no_improve >= max_epochs_no_improve:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        
        if avg_train_loss < avg_val_loss * 0.7:
            print("Warning: Potential overfitting detected")
            
            
except KeyboardInterrupt:
    print("Training interrupted")

end_time = time.perf_counter()
elapsed_time = end_time - start_time

# Save final model
torch.save(model.state_dict(), 'othello_cnn_final_model.pth')
print(f'Total train time: {elapsed_time:.2f} seconds')
