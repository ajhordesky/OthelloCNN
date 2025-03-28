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

# Load the trained model
model = load_trained_model()

# Example board (replace with your actual board)
new_board = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=torch.float32)

# Get predictions for both players
black_move = predict_move(model, new_board, player=-1)  # Black's turn

print(f"Black's best move: {black_move}")