import json

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
