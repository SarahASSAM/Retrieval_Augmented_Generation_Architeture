import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
 
# Désactiver le parallélisme des tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
 
# Charger les données et préparer les embeddings
@st.cache_resource
def load_and_prepare_data(file_path, model_name="all-MiniLM-L6-v2"):
    with open(file_path, 'r', encoding='utf-8') as file:
        product_data = [json.loads(line) for line in file]
    
    descriptions = [product.get("description", "") for product in product_data if "description" in product]
    
    # Segmentation des textes
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    segmented_texts = []
    for description in descriptions:
        if isinstance(description, list):
            description = " ".join(description)
        if description:
            chunks = text_splitter.split_text(description)
            segmented_texts.extend(chunks)
    
    # Générer des embeddings
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    embeddings = embedding_model.embed_documents(segmented_texts)
    
    # Créer un index FAISS
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    # Créer un docstore
    documents = [Document(page_content=text) for text in segmented_texts]
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
    
    # Créer un vector store
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    
    return vector_store
 
# Charger un modèle LLM (comme BLOOM) pour générer des réponses
@st.cache_resource
def load_llm_model(model_name="bigscience/bloomz-560m"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model
 
# Extraire et vérifier uniquement les informations pertinentes
def get_flexible_answer(query, retriever, max_results=3):
    # Récupérer les documents pertinents
    results = retriever.get_relevant_documents(query)
    
    if not results:
        return "Je ne connais pas la réponse."  # Aucun document trouvé
 
    # Rassembler les informations pertinentes
    matched_contexts = []
    query_words = set(query.lower().split())
    for result in results[:max_results]:
        document_words = set(result.page_content.lower().split())
        if query_words & document_words:  # Vérifie s'il y a des mots en commun
            matched_contexts.append(result.page_content)
 
    if not matched_contexts:
        return "Je ne connais pas la réponse."  # Aucun document ne correspond directement
 
    # Retourner les informations pertinentes trouvées
    return "\n".join(matched_contexts).strip()
 
# Générer une réponse concise avec LLM
def generate_response(query, context, tokenizer, model, temperature, top_p):
    # Réduction du contexte pour éviter des prompts trop longs
    truncated_context = context[:1000]  # Limite à 1000 caractères
 
    # Formulation d'un prompt clair
    prompt = f"""
    Voici les informations pertinentes extraites des documents :
    {truncated_context}
 
    Répondez uniquement à la question suivante :
    {query}
 
    Si aucune réponse ne peut être trouvée dans ces informations, répondez : "Je ne connais pas la réponse."
    """
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,  # Limite la longueur de la réponse générée
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=1
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()
 
# Interface utilisateur avec Streamlit
def main():
    st.title("Architecture Retrieval Augmented Generation sur des descriptions de produits Amazon")
    st.write("Posez une question pour obtenir des réponses basées sur les informations fournies.")
    
    file_path = 'meta.jsonl'  # Remplacez par le chemin réel de votre fichier JSON
    vector_store = load_and_prepare_data(file_path)
    retriever = vector_store.as_retriever()
 
    tokenizer, model = load_llm_model()
 
    # Entrée utilisateur
    query = st.text_input("Votre question :")
    
    # Paramètres de test
    temperature = st.slider("Température (créativité)", 0.1, 2.0, 1.0, step=0.1)
    top_p = st.slider("Top-p (probabilité cumulative)", 0.1, 1.0, 0.9, step=0.1)
 
    if st.button("Soumettre"):
        if query.strip():
            # Obtenir les informations pertinentes
            context = get_flexible_answer(query, retriever)
            
            if context == "Je ne connais pas la réponse.":
                st.write("### Réponse :")
                st.write(context)
            else:
                # Générer une réponse concise avec les paramètres choisis
                response = generate_response(query, context, tokenizer, model, temperature, top_p)
                st.write("### Réponse :")
                st.write(response)  # On affiche uniquement la réponse générée
        else:
            st.write("Veuillez entrer une question.")
 
if __name__ == "__main__":
    main()
 
