import torch
import torch.nn as nn
import torch.nn.functional as F

class OthelloCNN(nn.Module):
    def __init__(self):
        super(OthelloCNN, self).__init__()
        # Add batch normalization for better training stability
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Add residual connections
        self.residual = nn.Conv2d(2, 256, kernel_size=1)  # For shortcut connection
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Improved fully connected layers with dropout
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)

    def forward(self, x):
        residual = self.residual(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Add attention
        attention = self.attention(x)
        x = x * attention
        
        # Add residual connection
        x += residual
        
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(model_path="othello_cnn_final_model.pth"):
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



def predict_move(model, board_tensor, temperature=1.0, available_moves=None):
    model.eval()
    with torch.no_grad():
        # Create player channel (Black=1, White=-1)
        player_tensor = torch.ones_like(board_tensor) * (1 if model.training else -1)
        
        # Create 2-channel input [board, player]
        input_tensor = torch.stack([board_tensor, player_tensor]).unsqueeze(0)
        
        # Get model logits
        logits = model(input_tensor).squeeze(0)  # Shape: [64]
        
        # Handle valid moves
        if available_moves is None:
            # Convert to numpy for move validation
            board_np = board_tensor.cpu().numpy()
            available_moves = find_available_moves(
                [[1 if x == 1 else -1 if x == -1 else 0 for x in row] 
                 for row in board_np],
                1 if model.training else -1
            )
            available_moves = list(available_moves.keys()) if available_moves else []
        
        # Create mask for valid moves
        valid_indices = [y * 8 + x for (x, y) in available_moves]
        mask = torch.full((64,), -float('inf'), device=logits.device)
        mask[valid_indices] = 0
        masked_logits = logits + mask
        
        # Apply temperature
        if temperature != 1.0:
            masked_logits = masked_logits / temperature
        
        # Sample move
        if len(valid_indices) == 0:
            return None
            
        if temperature == 0.0:  # Greedy selection
            move_idx = torch.argmax(masked_logits).item()
        else:  # Temperature-based sampling
            probs = torch.softmax(masked_logits, dim=-1)
            move_idx = torch.multinomial(probs, 1).item()
        
        return divmod(move_idx, 8)  # (row, col)

def weighted_loss(output, target, winner, player_color):
   
    base_loss = F.cross_entropy(output, target, reduction='none')
    
   
    weights = torch.where(winner == player_color, 2.0, 1.0) 
    weighted_loss = (base_loss * weights).mean()
    
    return weighted_loss
