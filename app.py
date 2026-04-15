from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Chargement du modèle une seule fois au démarrage
model = SentenceTransformer("all-MiniLM-L6-v2")

def score_semantique(texte1: str, texte2: str) -> float:
    """Calcule la similarité sémantique entre deux textes (0 à 100)"""
    if not texte1 or not texte2:
        return 0.0
    emb1 = model.encode([texte1])
    emb2 = model.encode([texte2])
    score = cosine_similarity(emb1, emb2)[0][0]
    return round(float(score) * 100, 2)

def score_competences_exact(candidat: str, offre: str) -> float:
    """Overlap exact entre compétences (0 à 100)"""
    co = set(map(str.strip, offre.lower().split(',')))
    cc = set(map(str.strip, candidat.lower().split(',')))
    if not co:
        return 0.0
    communs = co & cc
    return round(len(communs) / len(co) * 100, 2)

@app.route('/score-competences', methods=['POST'])
def score_competences():
    data = request.json
    comp_candidat = data.get('competences_candidat', '')
    comp_offre    = data.get('competences_offre', '')

    # Score sémantique (NLP)
    sem = score_semantique(comp_candidat, comp_offre)

    # Score exact (overlap)
    exact = score_competences_exact(comp_candidat, comp_offre)

    # Score final pondéré (sur 40 pts comme votre Laravel)
    final = round((0.6 * sem + 0.4 * exact) / 100 * 40, 2)

    return jsonify({
        "score":           final,        # ← utilisé par Laravel
        "semantic_score":  sem,          # ← bonus pour debug
        "exact_score":     exact,        # ← bonus pour debug
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(port=5000, debug=True)