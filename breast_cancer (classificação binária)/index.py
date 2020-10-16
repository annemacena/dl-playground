import pandas as pd

previsores = pd.read_csv('entradas_breast.csv')
classes = pd.read_csv('saidas_breast.csv')

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classes_treinamento, classes_teste = train_test_split(previsores, classes, test_size=0.25)

from keras.models import Sequential
# Camadas densas => cada um dos neurônios é ligado a cada um neurônio da camada subsequente
# rede neural fully connected
from keras.layers import Dense

classificador = Sequential()

# units => (no. de entradas + no. de saídas) / 2, nesse caso foi arredondado pra 16 
classificador.add(Dense(units = 16,
                        activation = 'relu',
                        kernel_initializer = 'random_uniform',
                        input_dim = 30))

# possibilidade de adicionar mais camadas ocultas
# classificador.add(Dense(units = 16,
#                         activation = 'relu',
#                         kernel_initializer = 'random_uniform'))

#  função de ativação sigmoid por ser um problema binário
classificador.add(Dense(units = 1,
                        activation = 'sigmoid'))

classificador.compile(optimizer = 'adam',
                      loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

classificador.fit(previsores_treinamento,
                  classes_treinamento,
                  batch_size = 10,
                  epochs = 100)

resultado = classificador.evaluate(previsores_teste, classes_teste)