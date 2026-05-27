"""Stretch 6B-S1: Embedding space visualization for words and documents.

Creates:
- plots/word_embeddings_tsne.png
- plots/document_embeddings_tsne.png

The script uses one dimensionality reduction method (t-SNE by default)
for both word-level and document-level embeddings so the comparison is
method-consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer

from embeddings_lab import extract_bert_embedding, load_glove


GLOVE_PATH = Path("data/glove_50k_50d.txt")
BBC_PATH = Path("data/bbc_news.csv")
PLOTS_DIR = Path("plots")

REDUCTION_METHOD = "tsne"
TSNE_RANDOM_STATE = 42
WORD_TSNE_PERPLEXITY = 25
DOC_TSNE_PERPLEXITY = 8


WORD_CANDIDATES: Dict[str, List[str]] = {
    "countries": [
        "america", "canada", "mexico", "brazil", "argentina", "chile", "peru", "colombia",
        "england", "france", "germany", "italy", "spain", "portugal", "russia", "ukraine",
        "sweden", "norway", "finland", "denmark", "poland", "greece", "turkey", "egypt",
        "morocco", "nigeria", "kenya", "ethiopia", "india", "china", "japan", "korea",
        "thailand", "vietnam", "indonesia", "australia", "zealand", "iran", "iraq", "israel",
        "syria", "pakistan", "afghanistan", "ireland", "switzerland", "austria", "romania", "hungary",
    ],
    "sports": [
        "football", "soccer", "basketball", "baseball", "tennis", "cricket", "rugby", "hockey",
        "golf", "boxing", "cycling", "swimming", "running", "athlete", "coach", "stadium",
        "league", "tournament", "championship", "match", "score", "victory", "defeat", "goal",
        "season", "playoff", "referee", "captain", "striker", "midfielder", "defender", "goalkeeper",
        "training", "fitness", "medal", "olympic", "record", "competition", "fans", "crowd",
        "team", "player", "club", "manager", "final", "winner", "loser", "draw",
    ],
    "finance": [
        "market", "stock", "shares", "bond", "bank", "credit", "loan", "debt",
        "money", "cash", "profit", "loss", "revenue", "income", "salary", "wages",
        "tax", "budget", "economy", "trade", "growth", "inflation", "recession", "investment",
        "investor", "currency", "dollar", "euro", "pound", "yen", "fund", "capital",
        "portfolio", "dividend", "insurance", "mortgage", "savings", "spending", "cost", "price",
        "sales", "demand", "supply", "corporate", "business", "industry", "finance", "accounting",
    ],
    "technology": [
        "computer", "software", "hardware", "internet", "network", "database", "server", "cloud",
        "mobile", "phone", "device", "digital", "online", "website", "search", "engine",
        "algorithm", "data", "robot", "ai", "machine", "learning", "model", "code",
        "program", "python", "java", "linux", "windows", "security", "encryption", "privacy",
        "chip", "processor", "memory", "storage", "sensor", "camera", "signal", "wireless",
        "satellite", "innovation", "startup", "platform", "app", "automation", "technology", "cyber",
    ],
    "emotions": [
        "happy", "sad", "angry", "fear", "joy", "love", "hate", "hope",
        "anxiety", "calm", "stress", "proud", "shame", "guilt", "trust", "surprise",
        "grief", "relief", "excited", "bored", "lonely", "confident", "nervous", "depressed",
        "smile", "laugh", "cry", "tears", "pain", "pleasure", "kind", "gentle",
        "friendly", "hostile", "passion", "emotion", "feeling", "mood", "heart", "spirit",
        "sorrow", "delight", "frustration", "optimism", "pessimism", "compassion", "empathy", "sympathy",
    ],
}


@dataclass
class ProjectionResult:
    points_2d: np.ndarray
    labels: List[str]
    categories: List[str]


def reduce_to_2d(vectors: np.ndarray, *, perplexity: int) -> np.ndarray:
    """Apply the selected reduction method and return 2D coordinates."""
    if REDUCTION_METHOD != "tsne":
        raise ValueError(f"Unsupported reduction method: {REDUCTION_METHOD}")

    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=TSNE_RANDOM_STATE,
    )
    return model.fit_transform(vectors)


def select_glove_words(
    glove: Dict[str, np.ndarray],
    *,
    candidates_by_category: Dict[str, Sequence[str]],
    per_category: int = 40,
) -> Tuple[List[str], List[str], np.ndarray]:
    """Select a balanced set of GloVe words by semantic category."""
    selected_words: List[str] = []
    selected_categories: List[str] = []
    selected_vectors: List[np.ndarray] = []

    for category, candidates in candidates_by_category.items():
        available = [w for w in candidates if w in glove]
        chosen = available[:per_category]
        if len(chosen) < per_category:
            raise ValueError(
                f"Category '{category}' has only {len(chosen)} available words in GloVe; "
                f"need {per_category}."
            )

        for word in chosen:
            selected_words.append(word)
            selected_categories.append(category)
            selected_vectors.append(glove[word])

    return selected_words, selected_categories, np.vstack(selected_vectors)


def sample_bbc_articles(df: pd.DataFrame, *, per_category: int = 4) -> pd.DataFrame:
    """Sample 20 BBC articles with balanced categories (4 x 5 categories)."""
    required_categories = ["business", "entertainment", "politics", "sport", "tech"]
    rows: List[pd.DataFrame] = []

    for category in required_categories:
        subset = df[df["category"] == category].head(per_category)
        if len(subset) < per_category:
            raise ValueError(
                f"Category '{category}' has only {len(subset)} rows; need {per_category}."
            )
        rows.append(subset)

    sampled = pd.concat(rows, ignore_index=True)
    return sampled


def plot_projection(
    projection: ProjectionResult,
    *,
    title: str,
    output_path: Path,
    annotations: Dict[str, str] | None = None,
) -> None:
    """Create a category-colored scatter plot and save it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique_categories = sorted(set(projection.categories))
    cmap = plt.get_cmap("tab10")
    colors = {cat: cmap(i) for i, cat in enumerate(unique_categories)}

    plt.figure(figsize=(12, 9), dpi=160)
    for category in unique_categories:
        idx = [i for i, c in enumerate(projection.categories) if c == category]
        points = projection.points_2d[idx]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=48,
            alpha=0.85,
            color=colors[category],
            edgecolors="black",
            linewidths=0.3,
            label=category,
        )

    if annotations:
        for label, text in annotations.items():
            if label not in projection.labels:
                continue
            i = projection.labels.index(label)
            x, y = projection.points_2d[i]
            plt.annotate(text, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")

    plt.title(title, fontsize=13, pad=12)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Category", frameon=True)
    plt.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_word_projection(glove: Dict[str, np.ndarray]) -> ProjectionResult:
    labels, categories, vectors = select_glove_words(
        glove,
        candidates_by_category=WORD_CANDIDATES,
        per_category=40,
    )
    points_2d = reduce_to_2d(vectors, perplexity=WORD_TSNE_PERPLEXITY)
    return ProjectionResult(points_2d=points_2d, labels=labels, categories=categories)


def build_document_projection(sampled_articles: pd.DataFrame) -> ProjectionResult:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    labels = sampled_articles["text"].tolist()
    categories = sampled_articles["category"].tolist()

    vectors = [extract_bert_embedding(text, tokenizer, model) for text in labels]
    points_2d = reduce_to_2d(np.vstack(vectors), perplexity=DOC_TSNE_PERPLEXITY)

    return ProjectionResult(points_2d=points_2d, labels=labels, categories=categories)


def summarize_category_compactness(projection: ProjectionResult) -> Dict[str, float]:
    """Return mean distance to category centroid in 2D (lower means tighter)."""
    result: Dict[str, float] = {}
    for category in sorted(set(projection.categories)):
        idx = [i for i, c in enumerate(projection.categories) if c == category]
        points = projection.points_2d[idx]
        centroid = points.mean(axis=0)
        mean_dist = np.linalg.norm(points - centroid, axis=1).mean()
        result[category] = float(mean_dist)
    return result


def main() -> None:
    print("Loading GloVe vectors...")
    glove = load_glove(str(GLOVE_PATH))

    print("Projecting 200 GloVe words to 2D...")
    word_projection = build_word_projection(glove)
    word_annotations = {
        "football": "football",
        "basketball": "basketball",
        "bank": "bank",
        "stock": "stock",
        "internet": "internet",
        "ai": "ai",
        "happy": "happy",
        "angry": "angry",
        "france": "france",
        "japan": "japan",
        "emotion": "emotion",
        "technology": "technology",
    }
    word_plot = PLOTS_DIR / "word_embeddings_tsne.png"
    plot_projection(
        word_projection,
        title="GloVe Word Embeddings (200 words) reduced with t-SNE",
        output_path=word_plot,
        annotations=word_annotations,
    )

    print("Sampling BBC articles and projecting DistilBERT embeddings...")
    df = pd.read_csv(BBC_PATH)
    sampled_articles = sample_bbc_articles(df, per_category=4)
    doc_projection = build_document_projection(sampled_articles)

    doc_annotations = {
        text: f"{cat}: {text[:28]}..."
        for text, cat in zip(sampled_articles["text"], sampled_articles["category"])
    }
    doc_plot = PLOTS_DIR / "document_embeddings_tsne.png"
    plot_projection(
        doc_projection,
        title="BBC DistilBERT Document Embeddings (20 docs) reduced with t-SNE",
        output_path=doc_plot,
        annotations=doc_annotations,
    )

    print("\nSaved plots:")
    print(f"- {word_plot}")
    print(f"- {doc_plot}")

    word_compactness = summarize_category_compactness(word_projection)
    doc_compactness = summarize_category_compactness(doc_projection)

    print("\nCategory compactness (mean distance to centroid in 2D):")
    print("Words:", {k: round(v, 3) for k, v in word_compactness.items()})
    print("Docs:", {k: round(v, 3) for k, v in doc_compactness.items()})


if __name__ == "__main__":
    main()
