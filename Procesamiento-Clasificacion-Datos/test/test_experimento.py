# test/test_experimento.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.pipeline              import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm                   import SVC
from sklearn.naive_bayes           import MultinomialNB
from sklearn.preprocessing         import LabelEncoder
from sklearn.model_selection       import StratifiedKFold
from sklearn.metrics               import (accuracy_score, f1_score,
                                           precision_score, recall_score,
                                           classification_report,
                                           confusion_matrix)
import numpy as np
import pandas as pd
import time

from experiments.experiment import TextClassificationExperiment

# ─────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────
base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir   = os.path.join(base_dir, "data", "processed")
output_dir = os.path.join(base_dir, "data", "processed")

# ─────────────────────────────────────────
# INICIAR EXPERIMENTO
# ─────────────────────────────────────────
exp = TextClassificationExperiment(data_dir, output_dir)
exp.load_data()

resultados         = []
confusion_matrices = {}

# ─────────────────────────────────────────
# EXPERIMENTO 1 — Naive Bayes (baseline)
# ─────────────────────────────────────────
pipe_nb = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features = 5000,
        ngram_range  = (1, 2),
        sublinear_tf = True
    )),
    ("clf", MultinomialNB(alpha=1.0))
])

res_nb = exp.evaluate_classic("Naive Bayes", pipe_nb)
resultados.append(res_nb)

# ─────────────────────────────────────────
# EXPERIMENTO 2 — SVM lineal
# ─────────────────────────────────────────
pipe_svm_lin = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features = 5000,
        ngram_range  = (1, 2),
        sublinear_tf = True
    )),
    ("clf", SVC(kernel="linear", C=1.0, probability=True))
])

res_svm_lin = exp.evaluate_classic("SVM Lineal C=1", pipe_svm_lin)
resultados.append(res_svm_lin)

# ─────────────────────────────────────────
# EXPERIMENTO 3 — SVM RBF (hiperparametro diferente)
# ─────────────────────────────────────────
pipe_svm_rbf = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features = 5000,
        ngram_range  = (1, 2),
        sublinear_tf = True
    )),
    ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True))
])

res_svm_rbf = exp.evaluate_classic("SVM RBF C=10", pipe_svm_rbf)
resultados.append(res_svm_rbf)

# ─────────────────────────────────────────
# EXPERIMENTO 4 — robertuito (ground truth)
# ─────────────────────────────────────────
res_robertuito = exp.evaluate_transformer(
    model_name = "robertuito",
    model_id   = "pysentimiento/robertuito-sentiment-analysis",
    label_map  = {"POS": "POS", "NEG": "NEG", "NEU": "NEU"}
)
resultados.append(res_robertuito)

# ─────────────────────────────────────────
# EXPERIMENTO 5 — BETO
# ─────────────────────────────────────────
res_beto = exp.evaluate_transformer(
    model_name = "BETO",
    model_id   = "dccuchile/bert-base-spanish-wwm-cased",
    label_map  = {"LABEL_0": "NEG", "LABEL_1": "NEU", "LABEL_2": "POS"}
)
resultados.append(res_beto)

# ─────────────────────────────────────────
# EXPERIMENTO 6 — XLM-RoBERTa
# ─────────────────────────────────────────
res_xlm = exp.evaluate_transformer(
    model_name = "XLM-RoBERTa",
    model_id   = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    label_map  = {"Negative": "NEG", "Neutral": "NEU", "Positive": "POS"}
)
resultados.append(res_xlm)

# ─────────────────────────────────────────
# MATRICES DE CONFUSION (ultimo fold de cada modelo)
# ─────────────────────────────────────────
print("\n" + "="*60)
print("GENERANDO MATRICES DE CONFUSION")
print("="*60)

le    = LabelEncoder()
y_enc = le.fit_transform(exp.y)
cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(exp.X, exp.y))
_, test_idx = folds[-1]

X_test = [exp.X[i] for i in test_idx]
y_test = [exp.y[i] for i in test_idx]

# Matriz para SVM Lineal
pipe_svm_lin.fit(
    [exp.X[i] for i in folds[-1][0]],
    le.transform([exp.y[i] for i in folds[-1][0]])
)
y_pred_svm = le.inverse_transform(pipe_svm_lin.predict(X_test))
confusion_matrices["SVM_Lineal"] = exp.full_report("SVM Lineal", y_test, y_pred_svm)

# ─────────────────────────────────────────
# GUARDAR RESULTADOS
# ─────────────────────────────────────────
exp.save_results(resultados, confusion_matrices)

# ─────────────────────────────────────────
# RANKING FINAL EN CONSOLA
# ─────────────────────────────────────────
print("\n" + "="*60)
print("RANKING FINAL DE MODELOS")
print("="*60)

df_rank = pd.DataFrame(resultados).sort_values("f1_macro_avg", ascending=False)
df_rank.insert(0, "ranking", range(1, len(df_rank)+1))

print(df_rank[["ranking", "modelo", "f1_macro_avg",
               "accuracy_avg", "tiempo_seg"]].to_string(index=False))