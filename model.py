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
    available = find_available_moves(board, player)
    if (col, row) in available:
        return True
    else:
        return False

def find_available_moves(board, player):
    available_moves = {}  

    for y in range(0, 8): 
        for x in range(0, 8):  
            if board[y][x] == player:  
                for ydir in range(-1, 2):  
                    for xdir in range(-1, 2):
                        if xdir == 0 and ydir == 0:
                            continue
                        
                        move, flipped = search(x, y, xdir, ydir, board, player)
                        if move and flipped: 
                            if move not in available_moves:
                                available_moves[move] = set()
                            available_moves[move].update(flipped)  

    return available_moves

def search(x, y, xdir, ydir, board, player):
    included = []

    for dist in range(1, 9):
        xCurr = x + dist * xdir
        yCurr = y + dist * ydir

        
        if xCurr < 0 or xCurr > len(board) - 1 or yCurr < 0 or yCurr > len(board) - 1:
            return None, []

        
        if board[yCurr][xCurr] == -player:
            included.append((xCurr, yCurr))
        elif board[yCurr][xCurr] == player:
            return None, []  
        else: 
            if included:  
                return (xCurr, yCurr), included
            return None, []

    return None, []



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
