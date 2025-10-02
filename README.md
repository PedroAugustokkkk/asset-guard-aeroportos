# AssetGuard: Uma Demonstração Prática de Detecção de Anomalias com Aprendizado Não Supervisionado

## 1. Introdução

Este repositório apresenta um protótipo acadêmico, desenvolvido no âmbito da unidade curricular de Sistemas Computacionais, que demonstra a aplicação do algoritmo de clusterização **K-Means** para a detecção de anomalias. O objetivo deste trabalho é ilustrar, de forma prática, um pipeline de Machine Learning, desde a criação de um cenário de dados controlado até o treinamento de um modelo e a implementação de uma lógica de decisão para identificar desvios de um padrão estabelecido.

O projeto simula a movimentação de veículos em uma área operacional restrita, como um aeroporto, e implementa um "vigia virtual" capaz de discernir entre atividades normais e anômalas.

## 2. Metodologia Aplicada: Como o Sistema Funciona

A lógica do protótipo é dividida em quatro etapas fundamentais, que representam o ciclo de vida da solução de detecção.

### Etapa 1: Criação do Cenário (Simulação de Dados)

O primeiro passo consiste em gerar um ambiente de dados controlado para validar a metodologia. Como a coleta de dados reais está fora do escopo deste protótipo, utilizamos a biblioteca `numpy` para simular um cenário realista.

-   **Analogia:** Desenhamos um mapa de um pátio de aeroporto e definimos duas áreas de operação principais (ex: Pátio de Bagagens e Área de Manutenção).
-   **Implementação:**
    1.  Dois centróides são definidos para representar os centros dessas áreas.
    2.  Nuvens de pontos de dados (`dados_normais`) são geradas em torno desses centros, simulando a movimentação normal de veículos.
    3.  Um pequeno conjunto de pontos (`dados_anomalos`) é deliberadamente posicionado em locais distantes e não autorizados, representando as anomalias que desejamos detectar.
-   **Reprodutibilidade:** A semente do gerador de números aleatórios (`np.random.seed(42)`) é fixada para garantir que os resultados sejam sempre os mesmos a cada execução, uma prática essencial para a validação consistente de modelos em trabalhos acadêmicos.

### Etapa 2: Treinamento do "Vigia Virtual" (Modelo K-Means)

Nesta fase, o algoritmo K-Means da biblioteca `scikit-learn` é treinado para "aprender" o que constitui um comportamento normal.

-   **Analogia:** Apresentamos a um novo "guarda de segurança" um mapa contendo apenas as rotas e localizações normais, para que ele memorize as zonas de trabalho padrão.
-   **Implementação:** O modelo é treinado utilizando **exclusivamente** o conjunto de `dados_normais`. O algoritmo, então, calcula e armazena as coordenadas dos centros matemáticos (centróides) que melhor representam os agrupamentos de dados normais. Ao final desta etapa, o modelo "sabe" onde as operações padrão ocorrem, mas não tem nenhum conhecimento sobre o que é uma anomalia.

### Etapa 3: Definição da Lógica de Decisão (Limiar de Anomalia)

Com o modelo treinado, o próximo passo é definir uma regra clara e quantitativa para a detecção.

-   **Analogia:** Damos uma ordem ao "guarda": "Se você avistar um veículo muito longe do centro da área de trabalho mais próxima, soe o alarme!".
-   **Implementação:** O script calcula um **limiar de anomalia**. Ele analisa todos os pontos de treinamento (os dados normais) e mede a distância de cada um ao seu centróide correspondente. O limiar é definido como a maior dessas distâncias, acrescida de uma pequena margem de segurança. Esse valor representa a "distância máxima permitida" para uma operação ser considerada normal.

### Etapa 4: Validação do Modelo

A etapa final consiste em testar a lógica com pontos de dados conhecidos para validar sua eficácia.

-   **Analogia:** Testamos o "guarda" mostrando a ele um veículo em uma localização normal e outro em uma localização suspeita, e verificamos se ele reage corretamente em ambos os casos.
-   **Implementação:** Uma função `verificar_anomalia` é criada. Ela recebe um novo ponto, calcula sua distância ao centróide mais próximo e compara essa distância com o limiar. O script então executa essa função com um ponto normal e um anômalo conhecidos, imprimindo o resultado da classificação e a distância medida, validando o funcionamento do sistema.

## 3. Visualização dos Resultados

Para auxiliar na compreensão do cenário e dos resultados, o script utiliza a biblioteca `matplotlib` para gerar uma visualização gráfica dos dados, que é salva no arquivo `mapa_simulado_aeroporto.png`.

![Mapa Simulado do Aeroporto](mapa_simulado_aeroporto.png)
*Figura 1: Representação visual do cenário de dados. Os clusters de operação normal são mostrados em azul, enquanto as anomalias de teste são destacadas com um "X" vermelho.*

## 4. Como Reproduzir este Estudo

Para executar este protótipo e reproduzir os resultados apresentados, siga os passos abaixo.

### 4.1. Instalação das Dependências

1.  Crie um arquivo chamado `requirements.txt` com o seguinte conteúdo:
    ```txt
    numpy
    scikit-learn
    matplotlib
    ```
2.  Instale as bibliotecas via pip:
    ```bash
    pip install -r requirements.txt
    ```

### 4.2. Execução

Execute o script principal em um terminal:
```bash
python nome_do_script.py
