# DocSearcher

DocSearcher é uma ferramenta para extrair e pesquisar conteúdo de documentos no formato .docx. Ele utiliza técnicas de embedding para gerar representações semânticas dos documentos, permitindo pesquisas com consultas.

## Instalação

Para instalar o DocSearcher, você pode clonar este repositório:

```bash
git clone https://github.com/on-ferreira/DocSearcher.git
```

Em seguida, instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

## Uso

Para usar o DocSearcher, siga estas etapas:

1. Execute o script `doc_searcher.py` com os argumentos igual mostrado abaixo.
2. Caso não informe os argumentos, eles serão perguntados por onde estiverem rodando o código.

Exemplo:

```bash
python doc_searcher.py "path para a pasta" "consulta aqui"
```

O resultado será o conteúdo do documento mais relevante para a consulta fornecida utilizando a GeminiAPI para trazer a resposta da consulta de forma mais orgânica e natural..


## Jupyter Notebook

Caso deseje mais detalhes da busca, buscar por top 3 arquivos mais significativos ou outras informações, utilize o jupyter notebook para isso.


## Exemplo

O arquivo [Doc_Searcher_example](Doc_Searcher_example.ipynb) possui um exemplo utilizando o jupyter notebook e pedaços de páginas da WikiPédia.


## Configuração

Antes de usar o DocSearcher, você precisará configurar sua chave de API do Google. Para fazer isso, crie um arquivo `google_api_key.py` na raiz do projeto e defina sua chave como `my_api_key`.

```python
my_api_key = "sua_chave_de_api_aqui"
```

## Licença

Este projeto está licenciado sob a Licença Pública Geral GNU v3.0 (GPLv3). Consulte o arquivo [LICENSE](LICENSE) para obter mais detalhes.

