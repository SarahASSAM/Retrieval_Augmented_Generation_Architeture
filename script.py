import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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

print(f"Nombre de vecteurs indexés : {index.ntotal}")


########### 4. Création d’une base de données vectorielle ###############
import chromadb

# Étape 1 : Initialiser ChromaDB
client = chromadb.Client()

# Créer une collection pour stocker uniquement les embeddings
collection = client.create_collection(name="product_embeddings")

# Étape 2 : Ajouter uniquement les embeddings à la collection
# Les IDs sont utilisés pour identifier chaque vecteur de manière unique
for i, embedding in enumerate(embeddings):
    collection.add(
        ids=[f"vector_{i}"],  # ID unique pour chaque vecteur
        embeddings=[embedding],  # Embedding uniquement
        metadatas=[{"vector_id": i}]  # Identifier chaque vecteur avec un ID dans les métadonnées
    )

# Étape 3 : Vérifier le nombre total de vecteurs stockés
print(f"Nombre total de vecteurs dans la collection : {collection.count()}")
