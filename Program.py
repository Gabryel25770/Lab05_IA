import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def estatisticas(resultados):
    return np.mean(resultados), np.std(resultados)

def obter_dados():
    dados = np.load('teste2.npy')
    return dados[0], np.ravel(dados[1])

def executar_mlp(x, y):
    modelo = MLPRegressor(hidden_layer_sizes=(10),
                          max_iter=1000,
                          activation='relu',
                          solver='adam',
                          learning_rate='adaptive',
                          n_iter_no_change=50)
    
    modelo.fit(x, y)
    predicoes = modelo.predict(x)
    erro_medio = np.mean((predicoes - y) ** 2)
    
    plt.figure(figsize=[14, 7])
    plt.subplot(1, 3, 1)
    plt.plot(x, y)
    plt.subplot(1, 3, 2)
    plt.plot(modelo.loss_curve_)
    plt.subplot(1, 3, 3)
    plt.plot(x, y, linewidth=1, color='red')
    plt.plot(x, predicoes, linewidth=2)
    plt.show()
    
    return erro_medio

def principal():
    erros = []
    for _ in range(10):
        x, y = obter_dados()
        erros.append(executar_mlp(x, y))
    
    media, desvio = estatisticas(erros)
    print("Erro médio:", media)
    print("Desvio padrão:", desvio)

if __name__ == "__main__":
    principal()
