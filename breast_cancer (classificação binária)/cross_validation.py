import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas_breast.csv')
classes = pd.read_csv('saidas_breast.csv')

def criarRede():
    classificador = Sequential()

    # units => (no. de entradas + no. de saídas) / 2, nesse caso foi arredondado pra 16 
    classificador.add(Dense(units = 16,
                            activation = 'relu',
                            kernel_initializer = 'random_uniform',
                            input_dim = 30))
    # Camada de Dropout zera alguns dados randomicamente para não influenciar o
    # resultado final (recomendável 20 ~ 30% para não haver underfitting),
    # para evitar overfitting.
    classificador.add(Dropout(0.2))
    
    # possibilidade de adicionar mais camadas ocultas
    # classificador.add(Dense(units = 16,
    #                         activation = 'relu',
    #                         kernel_initializer = 'random_uniform'))
    
    # classificador.add(Dropout(0.2))
    
    #  função de ativação sigmoid por ser um problema binário
    classificador.add(Dense(units = 1,
                            activation = 'sigmoid'))
    
    classificador.compile(optimizer = 'adam',
                          loss = 'binary_crossentropy',
                          metrics = ['binary_accuracy'])
    
    return classificador
    
classificador = KerasClassifier(build_fn = criarRede,
                              epochs = 100,
                              batch_size = 10)

# cv => quantas vezes fará o teste, x vezes de divisões da base de dados
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classes,
                             cv = 10, scoring = 'accuracy')

media = resultados.mean()
# quão longe/perto estão da média
# quanto maior desvio padrão, maior a chance da rede ter overfitting
# (overfitting acontece quando a rede se adapta demais à uma base de dados)
desvio = resultados.std()