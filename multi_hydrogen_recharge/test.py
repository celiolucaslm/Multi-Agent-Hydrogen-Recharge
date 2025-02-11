import numpy as np

def gerar_numeros_poisson(lambda_val):
    """
    Gera números inteiros baseados na distribuição de Poisson.
    
    Parâmetros:
    - lambda_val: taxa \( \lambda \) da distribuição de Poisson.
    - tamanho: número de valores a serem gerados.
    
    Retorna:
    - Uma lista de números inteiros gerados.
    """
    numeros = np.random.poisson(lambda_val)
    return numeros

# Exemplo de uso
lambda_val = 5  # Taxa média (λ)
tamanho = 10    # Número de valores a gerar
numeros_poisson = gerar_numeros_poisson(lambda_val)

print("Números gerados:", numeros_poisson)

import matplotlib.pyplot as plt

# Visualizando os dados
plt.hist(numeros_poisson, bins=range(min(numeros_poisson), max(numeros_poisson) + 2), alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribuição de Poisson')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.grid(axis='y')
plt.show()