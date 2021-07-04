import numpy as np
import random
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

random.seed(1)
np.random.seed(1)

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

def evalScore(evaluation):
    evaluation = np.array(evaluation)
    evaluation = np.core.defchararray.replace(evaluation, '#+', '1000000')
    evaluation = np.core.defchararray.replace(evaluation, '#-', '-1000000')
    return 1/(1 + np.exp(-evaluation.astype(np.uint8)))/0.5 + 1
  
  def formatData(fen, evaluation):
    vecConvert = np.vectorize(convert)
    vecEvalScore = np.vectorize(evalScore)
    return np.array([vecConvert(row) for row in fen]), vecEvalScore(evaluation)
  
path = 'D:\\Users\\Koral Kulacoglu\\Coding\\python\\AI\\evalbot'

dataPath = os.path.join(os.path.dirname(__file__), 'Data/chessData.csv')
df = pd.read_csv(dataPath)
fen = df['FEN']
evaluation = df['Evaluation']

x, y = formatData(fen, evaluation)

model = keras.Sequential(name="eval_bot_1.0")
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=x[0].shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Flatten(data_format=None))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mae',
)

earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=250, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

history = model.fit(x, y, batch_size=64, epochs=100, callbacks=[earlystop])
model.save(path + '\\Model')

plt.plot(history.history['loss'])
plt.title('Eval Bot')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

model.save(os.path.dirname(__file__), 'Model')
