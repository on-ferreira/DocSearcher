import numpy as np
import pandas as pd
import google.generativeai as genai
from docx2python import docx2python
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from google_api_key import my_api_key

GOOGLE_API_KEY=my_api_key
genai.configure(api_key=GOOGLE_API_KEY)
model = "models/embedding-001"

def extract_docx_contents(docs_path):
    """
    Extracts the content of .docx documents from a specific directory.

    Args:
        docs_path (str): The path to the directory containing the .docx documents.

    Returns:
        list: A list of dictionaries, where each dictionary contains the title and content of a document.

    Example:
        >>> docs_path = "/path/to/your/docs"
        >>> documents = extract_docx_contents(docs_path)
    """

    documents = []

    for doc in os.listdir(docs_path):
        if doc.endswith(".docx"):
            path_file = os.path.join(docs_path, doc)
            doc_title = os.path.splitext(doc)[0]
            docx_content = docx2python(path_file)
           
            if docx_content.body and len(docx_content.body) > 1:
                text = ''.join([''.join(subsublist) for sublist in docx_content.body[1] for subsublist in sublist])
            elif docx_content.body:
                text = docx_content.text
            else:
                text = ''

            document = {
                    "Title": doc_title,
                    "Content": text
                }
            documents.append(document)
            

    return documents

def embed_fn(title, text):
  """
    Generates an embedding for the provided title and text using a specific model.

    This function utilizes Google Generative AI to generate an embedding for the provided title and text.
    The resulting embedding is useful for document retrieval tasks, where documents can be compared
    based on the similarity of their embeddings.

    Args:
        title (str): The title of the document.
        text (str): The text of the document.

    Returns:
        numpy.ndarray: The embedding generated for the provided title and text.

    Example:
        >>> title = "Example Title"
        >>> text = "This is an example text."
        >>> embedding = embed_fn(title, text)
    """

  return genai.embed_content(model=model,
                                 content=text,
                                 title=title,
                                 task_type="RETRIEVAL_DOCUMENT")["embedding"]


def generate_and_search_query(query, database, model):
    """
    Generates and searches a query within a given database using a specified model.

    This function generates an embedding for the provided query using the specified model.
    It then compares this embedding with the embeddings of documents in the database
    to find the most relevant document.

    Args:
        query (str): The query to be searched within the database.
        database (DataFrame): The DataFrame containing the database of documents.
        model (str): The model to be used for generating embeddings.

    Returns:
        str: The content of the most relevant document to the query.

    Example:
        >>> query = "Example query"
        >>> relevant_document = generate_and_search_query(query, df, "models/embedding-001")
    """
    embedding_of_query = genai.embed_content(model=model,
                                             content=query,
                                             task_type="RETRIEVAL_QUERY")["embedding"]

    dot_products = np.dot(np.stack(database["Embeddings"]), embedding_of_query)
    index = np.argmax(dot_products)

    return database.iloc[index]["Content"]

def generate_and_search_top_3_docs(query, database, model, similarity_threshold=0.65):
    """
    Generates and searches a query within a given database using a specified model, returning the top 3 most relevant documents with a similarity score equal to or greater than the specified threshold.

    This function generates an embedding for the provided query using the specified model.
    It then compares this embedding with the embeddings of documents in the database
    to find the top 3 most relevant documents with a similarity score equal to or greater than the specified threshold.

    Args:
        query (str): The query to be searched within the database.
        database (DataFrame): The DataFrame containing the database of documents.
        model (str): The model to be used for generating embeddings.
        similarity_threshold (float, optional): The minimum similarity score required for a document to be considered relevant. Defaults to 0.65.

    Returns:
        list: A list of tuples containing the content of the top 3 most relevant documents to the query, along with their similarity scores.

    Example:
        >>> query = "Example query"
        >>> top_3_documents = generate_and_search_top_3_docs(query, df, "models/embedding-001")
    """
    embedding_of_query = genai.embed_content(model=model,
                                             content=query,
                                             task_type="RETRIEVAL_QUERY")["embedding"]

    embeddings_of_database = np.stack(database["Embeddings"])
    similarities = cosine_similarity(embeddings_of_database, np.array(embedding_of_query).reshape(1, -1)).flatten()
    relevant_indices = np.where(similarities >= similarity_threshold)[0]
    sorted_indices = np.argsort(similarities[relevant_indices])[::-1][:3]

    top_relevant_documents = [(database.iloc[relevant_indices[index]]["Content"], similarities[relevant_indices[index]]) for index in sorted_indices]

    return top_relevant_documents


if len(sys.argv) > 1:
    docs_path = sys.argv[1]
else:
    docs_path = input("Digite o path da pasta: ")

documents = extract_docx_contents(docs_path)
df = pd.DataFrame(documents)
df.columns = ["Title", "Content"]
df["Content"] = df["Content"].str.replace(r"\t|\n", "", regex=True)
df["Embeddings"] = df.apply(lambda row: embed_fn(row["Title"], row["Content"]), axis=1)

if len(sys.argv) > 2:
    query = sys.argv[2]
else:
    query = input("Digite a consulta: ")

top_document = generate_and_search_query(query, df, model)

generation_config = {
  "temperature": 0,
  "candidate_count": 1
}

prompt_br = f"{query}. Não adicione nenhuma informação extra ao seguinte trecho. Trecho: {top_document}"

model_2 = genai.GenerativeModel("gemini-1.0-pro",
                                generation_config=generation_config)
response = model_2.generate_content(prompt_br)
print(response.text)