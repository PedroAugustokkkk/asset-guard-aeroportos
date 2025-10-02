import numpy as np # trabalhar com números e listas
from sklearn.cluster import KMeans # nossa IA do projeto
import matplotlib.pyplot as plt # gerar gráficos e visualização dos dados

# PARTE 1: SIMULAÇÃO DE DADOS DOS SENSORES

np.random.seed(42) # Define as sementes aleatórias (pontos hipotéticos no aeroporto) para que os resultados sejam sempre os mesmos, facilitando os testes.

# Define as coordenadas centrais (latitude, longitude) das "zonas normais" de operação no aeroporto.
# Zona 1: Pátio de Bagagens
centroide_bagagens = np.array([-12.91, -38.33])
# Zona 2: Área de Manutenção
centroide_manutencao = np.array([-12.90, -38.32])

# Gera 100 pontos de GPS em torno da zona de bagagens, simulando a operação normal dos veículos.
dados_bagagens = centroide_bagagens + np.random.randn(100, 2) * 0.005
# o mesmo para a área de manutenção.
dados_manutencao = centroide_manutencao + np.random.randn(100, 2) * 0.005

# Combina os dados das duas zonas normais em um único conjunto de dados.
dados_normais = np.vstack([dados_bagagens, dados_manutencao])

# Gera 5 pontos de GPS "anômalos", que estão longe das zonas normais. Esses 5 são só pra exemplificar o protótipo mesmo.
dados_anomalos = np.array([
    [-12.93, -38.34], # Ponto 1: Perto da pista de pouso
    [-12.89, -38.35], # Ponto 2: Perto do terminal de passageiros
    [-12.92, -38.31], # Ponto 3: Perto do portão de serviço
    [-12.88, -38.30], # Ponto 4: Fora do perímetro do aeroporto
    [-12.94, -38.32]  # Ponto 5: Área de carga restrita
])

# Combina todos os dados (normais e anômalos) para a visualização.
todos_os_dados = np.vstack([dados_normais, dados_anomalos])

print("--- Dados Simulados Gerados com Sucesso ---")
# Imprime o formato do array de dados normais (linhas, colunas).
print(f"Formato dos dados normais: {dados_normais.shape}")
# O mesmo para os dados anômalos
print(f"Formato dos dados anômalos: {dados_anomalos.shape}")
print("\n")


# PARTE 2: VISUALIZAÇÃO DOS NOSSOS DADOS

plt.figure(figsize=(10, 8)) # criação da figura do gráfico com tamanh específico

plt.scatter(dados_normais[:, 1], dados_normais[:, 0], c='blue', label='Operação Normal') # plota os dados normais com azuis

plt.scatter(dados_anomalos[:, 1], dados_anomalos[:, 0], c='red', marker='x', s=100, label='Anomalias') # plota os dados anômalos como vermelhos

plt.title('Mapa Simulado da Área Técnica do Aeroporto') # título do gráfico

plt.xlabel('Longitude') # define o eixo x como longitude

plt.ylabel('Latitude') # define o eixo y como longitude

plt.legend() # legenda para identificas os pontos (legenda de gráfico)

plt.grid(True) # grade para facilitar a visualização humana

plt.savefig('mapa_simulado_aeroporto.png') # aqui é importante, salvamos esse gráfico como arquivo .png

# Mostra o gráfico na tela.

print("--- Gráfico dos Dados Gerado e Salvo como 'mapa_simulado_aeroporto.png' ---")
print("\n")


# PARTE 3: IMPLEMENTAÇÃO E TREINAMENTO DO K-MEANS

numero_de_clusters = 2 # definimos o númeo de clusters (zonas normais) para o algoritmo aprender
kmeans = KMeans(n_clusters=numero_de_clusters, random_state=42, n_init=10) # criação de uma instância do modelo KMeans e especificamos o número de clusters.

# Treina o modelo USANDO APENAS OS DADOS NORMAIS. O modelo deve aprender o que é "normal". Podemos pensar depois nas "anomalias"
kmeans.fit(dados_normais)

# Obtém as coordenadas dos centros dos clusters que o modelo aprendeu.
centroides_aprendidos = kmeans.cluster_centers_

print("--- Modelo K-Means Treinado com Sucesso ---")
# Imprime as coordenadas dos centroides que o modelo encontrou.
print(f"Centroides aprendidos pelo modelo:\n{centroides_aprendidos}")
print("\n")


# PARTE 4: LÓGICA DE DETECÇÃO DE ANOMALIAS

# Calcula o "limiar de anomalia". A ideia é encontrar a distância máxima de um ponto normal ao seu centroide.
# Qualquer ponto que esteja mais longe que isso será considerado uma anomalia.
# Para cada ponto normal, calcula a distância para o centro do cluster ao qual ele pertence.
distancias = np.min(kmeans.transform(dados_normais), axis=1)
# Define o limiar como a maior dessas distâncias, com uma pequena margem de segurança (multiplicando por 1.1).
limiar_anomalia = np.max(distancias) * 1.1

print("--- Lógica de Detecção de Anomalias Pronta ---")
# Imprime o valor do limiar calculado.
print(f"Limiar de anomalia calculado: {limiar_anomalia:.5f}")
print("\n")

# Cria uma função para verificar se um novo ponto é uma anomalia.
def verificar_anomalia(ponto, modelo_kmeans, limiar):
    # Converte o ponto para o formato que o NumPy e o scikit-learn esperam.
    ponto_array = np.array([ponto])
    # Calcula a distância do novo ponto para TODOS os centroides aprendidos.
    distancias_aos_centroides = modelo_kmeans.transform(ponto_array)
    # Pega a distância para o centroide MAIS PRÓXIMO.
    distancia_minima = np.min(distancias_aos_centroides)

    # Compara a distância mínima com o nosso limiar.
    if distancia_minima > limiar:
        return True, distancia_minima
    else:
        return False, distancia_minima

# TESTANDO A LÓGICA

print("--- Testando a Lógica de Detecção ---")
# Pega um ponto de teste que sabemos que é NORMAL.
ponto_de_teste_normal = [-12.911, -38.331]
# Pega um ponto de teste que sabemos que é uma ANOMALIA.
ponto_de_teste_anomalo = [-12.93, -38.34]

# Verifica o ponto normal.
e_anomalia_normal, dist_normal = verificar_anomalia(ponto_de_teste_normal, kmeans, limiar_anomalia)
# Imprime o resultado para o ponto normal.
print(f"Verificando ponto NORMAL {ponto_de_teste_normal}:")
print(f"Distância ao centroide mais próximo: {dist_normal:.5f}")
print(f"É uma anomalia? {'SIM' if e_anomalia_normal else 'NÃO'}") # Usa um if ternário para uma saída amigável.

print("-" * 20)

# Verifica o ponto anômalo.
e_anomalia_anomalo, dist_anomalo = verificar_anomalia(ponto_de_teste_anomalo, kmeans, limiar_anomalia)
# Imprime o resultado para o ponto anômalo.
print(f"Verificando ponto ANÔMALO {ponto_de_teste_anomalo}:")
print(f"  Distância ao centroide mais próximo: {dist_anomalo:.5f}")
print(f"  É uma anomalia? {'SIM' if e_anomalia_anomalo else 'NÃO'}") # Usa um if ternário para uma saída amigável.