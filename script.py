
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import faiss 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import openai  # Correct import
import faiss  # Import de la bibliothèque Faiss

load_dotenv()

# Clé API OpenAI
OPENAI_API_KEY = os.environ.get('OPENAI_API_Key')

# Charger le fichier JSONL
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

# Chemin vers votre fichier
file_path = 'meta.jsonl'
product_data = load_jsonl(file_path)

#print(f"Nombre de produits chargés : {len(product_data)}")
#print("Exemple de produit :", product_data[0])

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Extraire les descriptions de chaque produit
descriptions = [product.get("description", "") for product in product_data if "description" in product]

# Configurer le séparateur
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # Taille de chaque chunk
    chunk_overlap=50  # Chevauchement entre les chunks
)

# Diviser les descriptions en chunks
segmented_texts = []
for description in descriptions:
    if description:  # Si la description existe
        chunks = text_splitter.split_text(" ".join(description))
        segmented_texts.extend(chunks)

print(f"Nombre total de chunks créés : {len(segmented_texts)}")
print("Exemple de chunk :", segmented_texts[:2])


# Générer des embeddings pour chaque chunk
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = embedding_model.encode(segmented_texts)


# Créer un index FAISS pour les embeddings
dimension = embeddings[0].shape[0]  # Dimension des vecteurs d'embeddings
index = faiss.IndexFlatL2(dimension)  # Index basé sur la distance euclidienne

# Ajouter les embeddings à l'index
index.add(np.array(embeddings))

# Créer les documents
documents = [Document(page_content=text) for text in segmented_texts]

# Créer un mapping d'index pour les documents
docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
index_to_docstore_id = {i: str(i) for i in range(len(documents))}



# Créer un vector store avec FAISS
vector_store = FAISS(
    embedding_function=embedding_model.encode,  # Passez la fonction d'encodage
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id  # Mapping corrigé
)

print(f"Nombre de vecteurs indexés : {index.ntotal}")


# Configurer un retriever pour rechercher les descriptions pertinentes
retriever = vector_store.as_retriever()

# Exemple de requête utilisateur
query = "OnePLus"
retrieved_docs = retriever.get_relevant_documents(query)

# Afficher les résultats
print("Résultats de la recherche :")
for i, doc in enumerate(retrieved_docs):
    print(f"Document {i+1}: {doc.page_content}")






# document1 = Document(page_content = "iphones are great")

# vector_store.add_documents(documents = [document1])
# ########### 4. Création d’une base de données vectorielle ###############
# import chromadb

# # Étape 1 : Initialiser ChromaDB
# client = chromadb.Client()

# # Créer une collection pour stocker uniquement les embeddings
# collection = client.create_collection(name="product_embeddings")

# # Étape 2 : Ajouter uniquement les embeddings à la collection
# # Les IDs sont utilisés pour identifier chaque vecteur de manière unique

# #for i, embedding in index:
# #   collection.add(
# #       ids=[f"vector_{i}"],  # ID unique pour chaque vecteur
# #       index=[embedding],  # Embedding uniquement
# #       metadatas=[{"vector_id": i}]  # Identifier chaque vecteur avec un ID dans les métadonnées
# #    )

# #print(collection [0])
# # Étape 3 : Vérifier le nombre total de vecteurs stockés
# print(f"Nombre total de vecteurs dans la collection : {collection.count()}")


# ############# 4

# # Fonction pour configurer un retriever
# def retrieve_descriptions(query, n_results=10):
#     """
#     Rechercher des descriptions pertinentes dans la base vectorielle.

#     Args:
#     - query (str): La requête utilisateur.
#     - n_results (int): Nombre de résultats à retourner.

#     Returns:
#     - Résultats de la recherche.
#     """
#     # Convertir la requête utilisateur en embedding
#     query_embedding = embedding_model.encode([query])[0]

#     print(query_embedding)
#     # Effectuer la recherche dans la base vectorielle
#     results = collection.query(
#         query_texts=[query],
#         n_results=n_results
#     )
    
#     return results



# # Étape 3 : Exemple d'utilisation
# query = "OnePlus"
# retrieved_results = retrieve_descriptions(query)

# # Afficher les résultats pertinents
# print("Requête utilisateur :", query)
# if retrieved_results['documents']:
#     print("\nDescriptions les plus pertinentes :")
#     for doc in retrieved_results['documents'][0]:
#         print(f"- {doc}")
# else:
#     print("Aucun document pertinent trouvé.")



