# Inteli - Instituto de Tecnologia e Liderança 

<p align="center">
<a href= "https://www.inteli.edu.br/"><img src="assets\logos\inteli.png" alt="Inteli - Instituto de Tecnologia e Liderança" border="0"></a>
</p>

# Nome do projeto

## GaIA

## :student: Integrantes: 
- <a href="https://www.linkedin.com/in/carol-pascarelli/">Carolina Pascarelli</a>
- <a href="https://www.linkedin.com/in/felipe-elgenneni/">Felipe de Melo Elgennei</a>
- <a href="https://www.linkedin.com/in/jo%C3%A3o-victor-wandermurem-de-oliveira/">João Victor Wandermurem de Oliveira</a> 
- <a href="https://www.linkedin.com/in/laura-rodrigues31/">Laura Rodrigues</a>
- <a href="https://www.linkedin.com/in/larissa-martins-304644245/">Larissa Martins</a> 
- <a href="https://www.linkedin.com/in/lucas-guerra-vicente-47a28927a/">Lucas Guerra Vicente</a>
- <a href="https://www.linkedin.com/in/ryan-botelho-gartlan/">Ryan Gartlan</a>

## :teacher: Professores:
### Orientador(a) 
- <a href="https://www.linkedin.com/in/marcelo-gon%C3%A7alves-phd-a550652/">Marcelo Gonçalves</a>
### Instrutores
- <a href="https://www.linkedin.com/in/jefferson-o-silva/">Jefferson Silva</a>
- <a href="https://www.linkedin.com/in/filipe-gon%C3%A7alves-08a55015b/">Filipe Gonçalves</a>
- <a href="https://www.linkedin.com/in/fillipe-resina-b2211a22/">Fillipe Resina</a> 
- <a href="https://www.linkedin.com/in/francisco-escobar/">Francisco Escobar</a> 
- <a href="https://www.linkedin.com/in/renato-penha/">Renato Penha</a> 
- <a href="https://www.linkedin.com/in/ricardo-missori/">Ricardo José Missori</a> 

## 📝 Descrição

Atualmente, a definição de metas de venda de publicidade na Rede Gazeta é baseada principalmente no desempenho do ano anterior, sem considerar de forma abrangente fatores como sazonalidade e eventos extraordinários. Essa abordagem pode resultar em metas desalinhadas com as reais condições do mercado, impactando a motivação da equipe comercial, a alocação eficiente de recursos e, consequentemente, a rentabilidade da empresa.

Para solucionar essa questão, a Rede Gazeta, com o apoio do Inteli (Instituto de Tecnologia e Liderança), está desenvolvendo um modelo preditivo de receitas publicitárias baseado em séries temporais. Esse modelo busca integrar a sazonalidade, a análise de tendências e padrões dos anos anteriores, a fim de gerar metas mais precisas e realistas. A utilização de técnicas de análise de dados permitirá que a empresa considere variáveis contextuais e eventos excepcionais que influenciam o desempenho futuro.

Essa solução proporcionará uma ferramenta de apoio para a tomada de decisões estratégicas, ajustando metas financeiras e otimizando os recursos da empresa. Com isso, a Rede Gazeta assegurará maior competitividade e eficiência, fortalecendo sua capacidade de responder às demandas do mercado.

## 📁 Estrutura de pastas

Dentre os arquivos presentes na raiz do projeto, definem-se:

- <b>readme.md</b>: arquivo que serve como guia e explicação geral sobre o projeto (o mesmo que você está lendo agora).

- <b>assets</b>: nesta pasta estão armazenadas todas as imagens e mídias que são utilizadas nos notebooks e na documentação, como gráficos, logos, mapas e personas. 

    - <b>grafico:</b> armazena gráficos de suporte ao projeto.
    - <b>logos:</b> Logos utilizadas.
    - <b>mapa-jornada:</b> Mídias relacionadas a jornadas do usuário.
    - <b>personas:</b> Informações visuais sobre personas.

- <b>documents</b>: Pasta onde ficam armazenados os documentos principais do projeto, como o DMP.md (Data Management Plan), que documenta o plano de gerenciamento de dados, e seu equivalente em PDF.

- <b>notebooks</b>: Todos os Jupyter Notebooks usados no desenvolvimento e análise do projeto são armazenados aqui. Dentro desta pasta, existe uma subpasta <b>data</b> onde os arquivos de dados a serem usados nos notebooks devem ser colocados.


## 💻 Execução dos projetos

### Execução dos Notebooks Localmente (VS Code com Python)

#### Requisitos:

- Baixe e instale o Visual Studio Code em https://code.visualstudio.com/.<p>

- Baixe e instale a versão mais recente do Python em https://www.python.org/downloads/.<p>
Durante a instalação, certifique-se de marcar a opção "Add Python to PATH".<p>

- Baixe e instale o GIT em https://git-scm.com/downloads<p>

- Instalação de Dependências.<p>
Após instalar o Python, abra o terminal do VS Code e execute os seguintes comandos para instalar as bibliotecas necessárias:

```
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install shap
pip install xgboost
pip install statsmodels
pip install tensorflow
pip install keras
```

#### Passos para Executar Localmente:
- Abra o VS Code e crie um novo arquivo .ipynb<p>
- No VS Code, vá até a aba de extensões (ícone de quadrado à esquerda) e procure por "Jupyter". Instale a extensão "Jupyter" para rodar notebooks diretamente no VS Code.<p>
- Abra o terminal VS Code e digite:
````
git clone https://github.com/Inteli-College/2024-2A-T12-IN03-G01.git
````
- Abra o arquivo .ipynb e execute o código célula por célula, clicando no ícone de "Run" ao lado de cada célula.


### Passos para Executar no Colab:

#### Instalação das Dependências:

- No Google Colab, execute as células que contêm as instruções para instalar as bibliotecas.<p>

```
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install scikit-learn
!pip install shap
!pip install xgboost
!pip install statsmodels
!pip install tensorflow
!pip install keras
```
#### Salvando o Notebook:

- Se o utilizador não salvar uma cópia do notebook no Google Drive próprio, não será possível salvar as alterações realizadas no arquivo. Para garantir isso, clique em "File" > "Save a copy in Drive" antes de começar a trabalhar no projeto.
Executando o Código:

- Após a configuração, execute as células com os comandos clicando no botão "Play" ao lado de cada célula.

## 🗃 Histórico de lançamentos

* 1.0.0 - 10/10/2024
    * [sprint 5] Lançamento da primeira versão do modelo preditivo com documentação.
        - 4.5.1.Modelo preditivo final
        - 4.5.2.Atendimento dos Requisitos
        - 4.5.3.Plano para caso de Falha do Modelo
        - 4.5.4.Explicabilidade
        - 4.5.5.Hipóteses sobre o Modelo Final
        - 4.5.6.Descrição das Ferramentas e Plataformas Utilizadas
        - 4.5.7.Conclusão
        - 5.1.Principais Resultados
        - 5.2.Recomendações ao Usuário
        - 5.3.Métricas alcançadas pelo Modelo
        - 6.Referências
        - 7.Anexos
        - Versão final [notebook.ipynb](notebooks/notebook.ipynb)
        - Modelo Final [Sarimax.ipynb](notebooks/sarimax.ipynb)

* 0.6.0 - 27/09/2024
    * [sprint 4] Comparação de modelos preditivos
        - 4.4.1 Modelo XGboost
        - 4.4.2 Rede Neural Artificial com Keras
        - 4.4.3 Random Forest Regressor
        - 4.4.4 Sarimax
        - 4.4.5 Comparação das Métricas
        - 3ª versão [notebook.ipynb](notebooks/notebook.ipynb)


* 0.3.1 - 13/09/2024
    * [sprint 3] Preparação de dados e modelo preditivo preliminar
        - 3.1.1 Tabela de Faturamento
        - 3.1.2 Tabela de Audiência
        - 3.1.3 Tabela Agosto
        - 3.1.4 Divisão dos dados
        - 3.2 Entendimento dos dados
        - 3.3 Preparação dos dados
        - 3.4 Modelagem
        - 3.5 Avaliação do Modelo
        - 3.6 Implementação
        - 4.3.1. Seleção de Features e Construção do Modelo
        - 4.3.2. Avaliação do Modelo e Discussão de Resultados
        - 4.3.3. Conclusão
        - 2ª versão [notebook.ipynb](notebooks/notebook.ipynb)

* 0.2.7 - 30/08/2024
    * [sprint 2] Análise exploratória e levantamento de hipóteses
        - 4.1.7.User Stories
        - 4.2.Compreensão dos Dados
        - 4.2.1.Exploração de dados
        - 4.2.2.Pré-processamento dos dados
        - 4.2.3.Hipóteses
        - 1ª versão [notebook.ipynb](notebooks/notebook.ipynb)

* 0.1.3 - 16/08/2024
    * [sprint 1] Documentação de entendimento do negócio
        - 1.Introdução
        - 2.Objetivos e Justificativa
        - 4.1.1.Contexto da indústria
        - 4.1.2.Análise SWOT
        - 4.1.3.Planejamento Geral da Solução
        - 4.1.4.Value Proposition Canvas
        - 4.1.5.Matriz de Riscos
        - 4.1.6. Personas
        - 4.1.9.Política de Privacidade

## 📋 Licença/License

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/Inteli-College/2024-2A-T12-IN03-G01?tab=readme-ov-file">GaIA</a> by <span property="cc:attributionName">Inteli, Carolina Pascarelli, Felipe de Melo Elgennei, João Victor Wandermurem de Oliveira, Laura Rodrigues, Larissa Martins, Lucas Guerra Vicente, Ryan Gartlan</span> is licensed under <a href="https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""></a></p>
