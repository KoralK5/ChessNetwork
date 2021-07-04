import numpy as np
import os
from tensorflow.keras.models import load_model
from IPython.display import clear_output
import chess

def convert(fen):
    # [color, pawn, knight, bishop, rook, queen, king]

    mapped = {
    ' ': [0, 0, 0, 0, 0, 0, 0],
    'P': [1, 1, 0, 0, 0, 0, 0],
    'p': [0, 1, 0, 0, 0, 0, 0],
    'N': [1, 0, 1, 0, 0, 0, 0],
    'n': [0, 0, 1, 0, 0, 0, 0],
    'B': [1, 0, 0, 1, 0, 0, 0],
    'b': [0, 0, 0, 1, 0, 0, 0],
    'R': [1, 0, 0, 0, 1, 0, 0],
    'r': [0, 0, 0, 0, 1, 0, 0],
    'Q': [1, 0, 0, 0, 0, 1, 0],
    'q': [0, 0, 0, 0, 0, 1, 0],
    'K': [1, 0, 0, 0, 0, 0, 1],
    'k': [0, 0, 0, 0, 0, 0, 1]
    }

    inted = []
    for row in fen:
        if row == ' ':
            break
        elif row != '/':
            if row in mapped:
                inted.append(mapped[row])
            else:
                for counter in range(0, int(row)):
                    inted.append(mapped[' '])

    return np.array_split(inted, 8)
  
  def engine(model, board, color):
    moves = {}
    for row in list(board.legal_moves):
        moves[row] = model.predict(np.array([convert(board.fen())]))[0][0]
    move = min(moves, key=moves.get) if color == 'white' else max(moves, key=moves.get)
    evaluation = min(moves.values()) if color == 'white' else max(moves.values())
    return move, evaluation*2-1
  
  def show(board, evaluation):
    print('FEN:', board.fen(), '\n')
    print('Evaluation:', str(evaluation))
    print('White' if evaluation > 0 else 'Black', 'is winning!\n\n')
    print(board)

def play(color, model):
    board = chess.Board()
    evaluation = 0
    while not(board.is_game_over()):
        show(board, evaluation)
        if color == 'white':
            while True:
                try: board.push_san(input('\nMove: ')); break
                except: clear_output(); show(board, evaluation)
            move, evaluation = engine(model, board, color)
            board.push(move)
        elif color == 'black':
            move, evaluation = engine(model, board, color)
            board.push(move)
            while True:
                try: board.push_san(input('\nMove: ')); break
                except: clear_output(); show(board, evaluation)
        clear_output(wait=True)
    return board.outcome()
  
path = os.path.dirname(__file__)
model = load_model(path + '\\Model')

outcome = play('white', model)
