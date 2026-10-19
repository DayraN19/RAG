import os
import pandas as pd
import json
import pypdf
import docx
import bm25s


# 1. Fonctions pour extraire le texte selon l'extension
def extract_text_from_pdf(filepath):
    try:
        reader = pypdf.PdfReader(filepath)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception:
        return ""

def extract_text_from_docx(filepath):
    try:
        doc = docx.Document(filepath)
        return " ".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""


# 2. Parcourir le dossier et centraliser les documents
documents = []
folder_path = "vllm-0.10.1"

for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    ext = filename.split('.')[-1].lower()

    if ext == 'txt':
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            documents.append({"text": f.read(), "source": filename})

    elif ext == 'pdf':
        text = extract_text_from_pdf(filepath)
        if text: documents.append({"text": text, "source": filename})

    elif ext == 'docx':
        text = extract_text_from_docx(filepath)
        if text: documents.append({"text": text, "source": filename})

    elif ext == 'csv':
        df = pd.read_csv(filepath)
        # Combine toutes les lignes et colonnes textuelles en chaînes de caractères
        for _, row in df.dropna(how='all').iterrows():
            row_text = " ".join(row.astype(str))
            documents.append({"text": row_text, "source": filename})
            
    elif ext in ['json', 'jsonl']:
        with open(filepath, 'r', encoding='utf-8') as f:
            # S'adapte si c'est un JSON classique ou du JSON Lines
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data: documents.append({"text": str(item), "source": filename})
                else:
                    documents.append({"text": str(data), "source": filename})
            except json.JSONDecodeError:
                f.seek(0)
                for line in f:
                    if line.strip(): documents.append({"text": line.strip(), "source": filename})

# 3. Tokenisation et Indexation avec BM25S
# Extraction de la liste brute de textes pour l'indexation
corpus_texts = [doc["text"] for doc in documents]

# Tokenisation (séparation par espaces, minuscules)
corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")

# Création du moteur de recherche
retriever = bm25s.BM25()
retriever.index(corpus_tokens)

# 4. Exemple de recherche
query = "comment installer pip"
query_tokens = bm25s.tokenize([query])
results, scores = retriever.retrieve(query_tokens, k=1)

# Affichage des résultats avec leur fichier d'origine
for i in range(results.shape[1]):
    doc_index = results[0, i]
    print(f"Trouvé dans : {documents[doc_index]['source']} (Score: {scores[0, i]:.2f})")
    print(f"Extrait : {corpus_texts[doc_index][:2000]}...\n")