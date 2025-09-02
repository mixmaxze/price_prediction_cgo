# Tutorial: Rodando a API de Previsão de Preços de Casas

Este tutorial explica como instalar dependências e executar a API de forma simples usando Python e pip.

O modelo utilizado para as predições é resultado dos experimentos realizados no notebook `experiments.ipynb`.

---

## 1. Pré-requisitos

- Python 3.10 ou superior instalado.
- pip instalado (verifique com `python --version` e `pip --version`).

Se o pip não estiver instalado, consulte a documentação oficial [aqui](https://pip.pypa.io/en/stable/installation/).


## 2. Instalando dependências da API

Com o pip instalado, execute no terminal o comando:

```sh
pip install -r requirements.txt
```

Com isso, as dependências da API contida em `main.py` serão instaladas.

## 3. Executando API

Para iniciar a API de predições de preços, execute no terminal o comando:

```sh
uvicorn main:app --reload
```

Com o comando executado, aparecerá no terminal o link da API (algo como `Uvicorn running on http://127.0.0.1:8000`). Clique no link e acesse a página inicial no seu navegador.

Após isso, basta enviar o arquivo CSV utilizando o botão de `Escolher arquivo` e depois clicar no botão `Enviar`.

Dessa forma, a API retornará um JSON com todas as predições realizadas, em que cada item dentro da lista de `predictions` contém o `PID` da entrada junto ao `SalePrice` predito pelo modelo.