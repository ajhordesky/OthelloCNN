import pandas as pd
from Game import Game
data = pd.read_csv('othello_dataset.csv')
data = data.dropna()
games = []
for _,row in data.iterrows():
    games.append(Game(row.iloc[0], row.iloc[1], row.iloc[2]))

cleaned_data = []
for game in games:
    game_data = {
        'game_id': game.id,
        'winner': game.winner,
        'game_moves': game.game
    }
    cleaned_data.append(game_data)
df = pd.DataFrame(cleaned_data)
df.to_csv("cleaned_games.csv", index=False)