"""
Task 5 Analysis Script — Generates comparison table and OOV stats.
Uses full corpus for TF-IDF/GloVe; 20 texts/category (100 total) for BERT.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import torch
from transformers import AutoTokenizer, AutoModel
from embeddings_lab import (
    build_tfidf, compute_tfidf_similarity,
    load_glove, text_to_glove, extract_bert_embedding
)

# ─── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/bbc_news.csv")
print(f"Full corpus: {len(df)} texts | {df['category'].value_counts().to_dict()}")

# Balanced subset: 20 texts per category = 100 total (for BERT)
subset_df = df.groupby("category").head(20).reset_index(drop=True)
texts = subset_df["text"].tolist()
print(f"Working subset: {len(texts)} texts (20 per category)\n")

# One query per category (first article of each)
queries = [subset_df[subset_df["category"] == cat]["text"].iloc[0]
           for cat in subset_df["category"].unique()]
query_categories = list(subset_df["category"].unique())

# ─── OOV Rate (on full corpus for accuracy) ───────────────────────────────────
print("[OOV] Loading GloVe & computing OOV rate on full corpus...")
glove = load_glove("data/glove_50k_50d.txt")
all_words, oov_words = [], []
for text in df["text"].tolist():
    words = text.lower().split()
    all_words.extend(words)
    oov_words.extend(w for w in words if w not in glove)

from collections import Counter
oov_rate = len(oov_words) / len(all_words) * 100
oov_counter = Counter(oov_words)
print(f"  Total words : {len(all_words):,}")
print(f"  OOV words   : {len(oov_words):,}  ({oov_rate:.1f}%)")
print(f"  Top-20 OOV  : {oov_counter.most_common(20)}\n")

# ─── Task 1: TF-IDF (subset) ──────────────────────────────────────────────────
print("[1/3] Building TF-IDF on subset...")
tfidf_matrix, _ = build_tfidf(texts)
tfidf_sim = compute_tfidf_similarity(tfidf_matrix)

# ─── Task 2: GloVe (subset) ───────────────────────────────────────────────────
print("[2/3] Computing GloVe embeddings on subset...")
glove_embeddings = np.array([text_to_glove(t, glove) for t in texts])
glove_sim = sklearn_cosine(glove_embeddings)

# ─── Task 3: BERT (subset — 100 texts is fast) ────────────────────────────────
print("[3/3] Computing BERT embeddings (100 texts)...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()

bert_embeddings = []
for i, text in enumerate(texts):
    if i % 20 == 0:
        print(f"      {i}/{len(texts)}...")
    bert_embeddings.append(extract_bert_embedding(text, tokenizer, model))
bert_embeddings = np.array(bert_embeddings)
bert_sim = sklearn_cosine(bert_embeddings)
print("  Done!\n")

# ─── Comparison Table ─────────────────────────────────────────────────────────
def get_top3(sim_matrix, query_idx, texts_list, df_ref):
    scores = sim_matrix[query_idx].copy()
    scores[query_idx] = -1
    top_idx = np.argsort(scores)[::-1][:3]
    results = []
    for i in top_idx:
        cat = df_ref[df_ref["text"] == texts_list[i]]["category"].values[0]
        results.append((texts_list[i], float(scores[i]), cat))
    return results

print("=" * 80)
print("COMPARISON TABLE — Top-3 Similar Texts per Method (5 Queries)")
print("=" * 80)

for query, cat in zip(queries, query_categories):
    q_idx = texts.index(query)
    tfidf_top3 = get_top3(tfidf_sim, q_idx, texts, subset_df)
    glove_top3  = get_top3(glove_sim,  q_idx, texts, subset_df)
    bert_top3   = get_top3(bert_sim,   q_idx, texts, subset_df)

    print(f"\n{'─'*80}")
    print(f"Query [{cat.upper()}]: {query[:85]}...")
    print(f"{'─'*80}")
    for method, top3 in [("TF-IDF", tfidf_top3), ("GloVe", glove_top3), ("BERT", bert_top3)]:
        print(f"  {method}:")
        for rank, (text, score, result_cat) in enumerate(top3, 1):
            match = "✅" if result_cat == cat else "⚠️"
            print(f"    {rank}. {match} [{result_cat:15s}] {score:.3f} | {text[:55]}...")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for query, cat in zip(queries, query_categories):
    q_idx = texts.index(query)
    tfidf_top3 = get_top3(tfidf_sim, q_idx, texts, subset_df)
    glove_top3  = get_top3(glove_sim,  q_idx, texts, subset_df)
    bert_top3   = get_top3(bert_sim,   q_idx, texts, subset_df)
    t_cats = [c for _, _, c in tfidf_top3]
    g_cats = [c for _, _, c in glove_top3]
    b_cats = [c for _, _, c in bert_top3]
    all_agree = set(t_cats) == set(g_cats) == set(b_cats) == {cat}
    print(f"[{cat:15s}] TF-IDF:{t_cats} | GloVe:{g_cats} | BERT:{b_cats}  {'ALL AGREE ✅' if all_agree else 'DISAGREE ⚠️'}")

print(f"\nOOV Rate: {oov_rate:.1f}% ({len(oov_words):,}/{len(all_words):,} words)")
print(f"Top OOV: {oov_counter.most_common(10)}")


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import torch
from transformers import AutoTokenizer, AutoModel
from embeddings_lab import (
    build_tfidf, compute_tfidf_similarity,
    load_glove, text_to_glove, extract_bert_embedding
)

# ─── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/bbc_news.csv")
texts = df["text"].tolist()
print(f"Loaded {len(texts)} texts across categories: {df['category'].value_counts().to_dict()}")

# One query per category (first article of each)
queries = [df[df["category"] == cat]["text"].iloc[0]
           for cat in df["category"].unique()]
query_categories = list(df["category"].unique())

# ─── Task 1: TF-IDF ──────────────────────────────────────────────────────────
print("\n[1/4] Building TF-IDF...")
tfidf_matrix, _ = build_tfidf(texts)
tfidf_sim = compute_tfidf_similarity(tfidf_matrix)
print(f"      TF-IDF similarity matrix: {tfidf_sim.shape}")

# ─── Task 2: GloVe ───────────────────────────────────────────────────────────
print("[2/4] Loading GloVe & computing embeddings...")
glove = load_glove("data/glove_50k_50d.txt")
glove_embeddings = np.array([text_to_glove(t, glove) for t in texts])
glove_sim = sklearn_cosine(glove_embeddings)
print(f"      GloVe similarity matrix: {glove_sim.shape}")

# ─── OOV Rate ────────────────────────────────────────────────────────────────
print("[2b] Computing OOV rate...")
all_words = []
oov_words = []
for text in texts:
    words = text.lower().split()
    all_words.extend(words)
    oov_words.extend([w for w in words if w not in glove])

oov_rate = len(oov_words) / len(all_words) * 100
print(f"     Total words: {len(all_words):,} | OOV: {len(oov_words):,} | Rate: {oov_rate:.1f}%")

# Show most common OOV word types
from collections import Counter
oov_counter = Counter(oov_words)
print(f"     Top 20 OOV words: {oov_counter.most_common(20)}")

# ─── Task 3: BERT (precompute all) ───────────────────────────────────────────
print("[3/4] Loading DistilBERT & computing embeddings (this takes a few minutes)...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()

bert_embeddings = []
for i, text in enumerate(texts):
    if i % 100 == 0:
        print(f"      BERT: {i}/{len(texts)}...")
    bert_embeddings.append(extract_bert_embedding(text, tokenizer, model))

bert_embeddings = np.array(bert_embeddings)
bert_sim = sklearn_cosine(bert_embeddings)
print(f"      BERT similarity matrix: {bert_sim.shape}")

# ─── Task 4: Comparison Table ────────────────────────────────────────────────
print("\n[4/4] Building comparison table...")

def get_top3(sim_matrix, query_idx, texts):
    scores = sim_matrix[query_idx].copy()
    scores[query_idx] = -1  # exclude self
    top_indices = np.argsort(scores)[::-1][:3]
    return [(texts[i], scores[i]) for i in top_indices]

print("\n" + "=" * 80)
print("COMPARISON TABLE — Top-3 Similar Texts per Method per Query")
print("=" * 80)

results_summary = []

for q_idx, (query, cat) in enumerate(zip(queries, query_categories)):
    query_text_idx = texts.index(query)
    
    tfidf_top3 = get_top3(tfidf_sim, query_text_idx, texts)
    glove_top3  = get_top3(glove_sim,  query_text_idx, texts)
    bert_top3   = get_top3(bert_sim,   query_text_idx, texts)
    
    print(f"\n{'─'*80}")
    print(f"Query [{cat.upper()}]: {query[:90]}...")
    print(f"{'─'*80}")
    
    for rank, (method, top3) in enumerate([("TF-IDF", tfidf_top3), ("GloVe", glove_top3), ("BERT", bert_top3)]):
        print(f"\n  {method}:")
        for i, (text, score) in enumerate(top3):
            # Find the category of this result
            try:
                text_cat = df[df["text"] == text]["category"].values[0]
            except Exception:
                text_cat = "?"
            print(f"    {i+1}. [{text_cat:15s}] score={score:.3f} | {text[:60]}...")
    
    # Check agreement
    tfidf_cats = [df[df["text"] == t]["category"].values[0] for t, _ in tfidf_top3]
    glove_cats  = [df[df["text"] == t]["category"].values[0] for t, _ in glove_top3]
    bert_cats   = [df[df["text"] == t]["category"].values[0] for t, _ in bert_top3]
    
    all_same_cat = set(tfidf_cats) == set(glove_cats) == set(bert_cats) == {cat}
    results_summary.append({
        "category": cat,
        "tfidf_cats": tfidf_cats,
        "glove_cats":  glove_cats,
        "bert_cats":   bert_cats,
        "all_agree":   all_same_cat,
    })

# ─── Analysis Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

print("\n1. Category Agreement:")
for r in results_summary:
    agree = "ALL AGREE ✅" if r["all_agree"] else "DISAGREE ⚠️"
    print(f"   [{r['category']:15s}] TF-IDF:{r['tfidf_cats']} | GloVe:{r['glove_cats']} | BERT:{r['bert_cats']}  → {agree}")

print(f"\n2. OOV Rate: {oov_rate:.1f}%")
print(f"   Total corpus words: {len(all_words):,}")
print(f"   OOV words: {len(oov_words):,}")
print(f"   Top OOV types: {oov_counter.most_common(10)}")

print("\nAnalysis complete!")
