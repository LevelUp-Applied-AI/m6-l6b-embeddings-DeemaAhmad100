"""
Module 6 Week B — Lab: Embeddings Comparison

Compare three text representation methods — TF-IDF, GloVe, and
DistilBERT — on the BBC News corpus (5 categories).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def build_tfidf(texts):
    """Build TF-IDF representations for a list of texts.

    Returns (tfidf_matrix, vectorizer).
    """
    vectorizer = TfidfVectorizer()
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    return tfidf_matrix, vectorizer


def compute_tfidf_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity from a TF-IDF matrix.

    Returns a numpy array of shape (n, n).
    """
    similarity_matrix = sklearn_cosine(tfidf_matrix)
    
    return similarity_matrix


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Returns a dict mapping each word to a numpy array.
    """
    embeddings = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:

            parts = line.strip().split()
            word = parts[0]  
            vector = np.array([float(x) for x in parts[1:]])  
            embeddings[word] = vector
    
    return embeddings


def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text.

    Skip out-of-vocabulary words. If every word is OOV, return a zero
    vector of shape (50,).
    """
    words = text.lower().split()
    
    valid_vectors = []
    for word in words:
        if word in embeddings:
            valid_vectors.append(embeddings[word])
    
    if len(valid_vectors) == 0:
        return np.zeros(50)
    
    return np.mean(valid_vectors, axis=0)


def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT.

    Returns a numpy array of shape (768,).
    """
    import torch
    
    inputs = tokenizer(text, truncation=True, max_length=512, 
                      return_tensors='pt', padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_state = outputs.last_hidden_state  # shape: (1, sequence_length, 768)
    
    attention_mask = inputs['attention_mask']  # shape: (1, sequence_length)
    
    masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
    
    summed = torch.sum(masked_embeddings, dim=1)  # shape: (1, 768)
    counts = torch.sum(attention_mask, dim=1, keepdim=True)  # shape: (1, 1)
    mean_pooled = summed / counts  # shape: (1, 768)
    
    return mean_pooled.squeeze(0).numpy()  # shape: (768,)


def compare_similarities(texts, queries, tfidf_sim, glove_embeddings,
                         bert_model, bert_tokenizer):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT.

    For each query, find the top-3 most similar texts under each method,
    excluding the query itself. Return:

        {query_text: {"tfidf": [(text, score), ...],
                      "glove": [(text, score), ...],
                      "bert":  [(text, score), ...]}}
    """
    import torch
    
    results = {}
    
    for query in queries:

        query_idx = texts.index(query)
        
        tfidf_scores = tfidf_sim[query_idx]
        tfidf_results = []
        for i, score in enumerate(tfidf_scores):
            if i != query_idx:  
                tfidf_results.append((texts[i], score))

        tfidf_results.sort(key=lambda x: x[1], reverse=True)
        
        query_glove = text_to_glove(query, glove_embeddings)
        glove_results = []
        for i, text in enumerate(texts):
            if i != query_idx:  
                text_glove = text_to_glove(text, glove_embeddings)

                similarity = np.dot(query_glove, text_glove) / (
                    np.linalg.norm(query_glove) * np.linalg.norm(text_glove)
                )
                glove_results.append((text, similarity))
        glove_results.sort(key=lambda x: x[1], reverse=True)
        


        query_bert = extract_bert_embedding(query, bert_tokenizer, bert_model)
        bert_results = []
        for i, text in enumerate(texts):
            if i != query_idx:  
                text_bert = extract_bert_embedding(text, bert_tokenizer, bert_model)

                similarity = np.dot(query_bert, text_bert) / (
                    np.linalg.norm(query_bert) * np.linalg.norm(text_bert)
                )
                bert_results.append((text, similarity))
        bert_results.sort(key=lambda x: x[1], reverse=True)
        
        results[query] = {
            "tfidf": tfidf_results[:3],
            "glove": glove_results[:3], 
            "bert": bert_results[:3]
        }
    
    return results


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModel

    # Load data
    df = pd.read_csv("data/bbc_news.csv")
    texts = df["text"].tolist()  # Use all texts for full analysis
    print(f"Loaded {len(texts)} texts")

    # Task 1: TF-IDF
    result = build_tfidf(texts)
    if result:
        tfidf_matrix, vectorizer = result
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        tfidf_sim = compute_tfidf_similarity(tfidf_matrix)
        if tfidf_sim is not None:
            print(f"TF-IDF similarity matrix shape: {tfidf_sim.shape}")

    # Task 2: GloVe
    glove = load_glove("data/glove_50k_50d.txt")
    if glove:
        print(f"Loaded {len(glove)} GloVe vectors")
        sample_emb = text_to_glove(texts[0], glove)
        if sample_emb is not None:
            print(f"Sample GloVe text embedding shape: {sample_emb.shape}")

    # Task 3: DistilBERT
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    sample_bert = extract_bert_embedding(texts[0], tokenizer, model)
    if sample_bert is not None:
        print(f"Sample BERT embedding shape: {sample_bert.shape}")

    # Task 4: Compare — pick one query per category so the cross-method
    # ranking comparison is not degenerate (the CSV is sorted by category,
    # so texts[:5] would all be from the same one).
    if result and glove and tfidf_sim is not None:
        queries = [df[df["category"] == cat]["text"].iloc[0]
                   for cat in df["category"].unique()]
        comparison = compare_similarities(
            texts, queries, tfidf_sim, glove, model, tokenizer
        )
        if comparison:
            for q in list(comparison.keys()):  # Show all queries
                print(f"\nQuery: {q[:80]}...")
                for method in ["tfidf", "glove", "bert"]:
                    top = comparison[q].get(method, [])
                    print(f"  {method}: {[t[:40] for t, _ in top[:3]]}")
