import torch
import torch.nn as nn
import torch.nn.functional as F

class OthelloCNN(nn.Module):
    def __init__(self):
        super(OthelloCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        
        x = x.view(-1, 256 * 8 * 8) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model_path="othello_cnn_model.pth"):
    model = OthelloCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def is_valid_move(board, row, col, player):
    if row < 0 or row >= 8 or col < 0 or col >= 8 or board[row][col] != 0:
        return False

    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]

    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == -player:
            r += dr
            c += dc
            while 0 <= r < 8 and 0 <= c < 8:
                if board[r][c] == player:
                    return True
                if board[r][c] == 0:
                    break
                r += dr
                c += dc
    return False

def predict_move(model, board, player):
    model.eval()
    with torch.no_grad():
        
        player_color = torch.full((8, 8), player, dtype=torch.float32)
        input_tensor = torch.stack([board, player_color], dim=0)
        input_tensor = input_tensor.unsqueeze(0)

        
        output = model(input_tensor)
        move_index = torch.argmax(output).item()
        row, col = divmod(move_index, 8)

        
        if is_valid_move(board, row, col, player):
            return row, col
        else:
           
            sorted_moves = torch.argsort(output, descending=True)
            for move in sorted_moves[0]:
                row, col = divmod(move.item(), 8)
                if is_valid_move(board, row, col, player):
                    return row, col
            return None 

def weighted_loss(output, target, winner, player_color):
   
    base_loss = F.cross_entropy(output, target, reduction='none')
    
   
    weights = torch.where(winner == player_color, 2.0, 1.0) 
    weighted_loss = (base_loss * weights).mean()
    
    return weighted_loss