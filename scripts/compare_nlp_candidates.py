from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
REPORTS_DIR = ROOT / 'reports'
LABELS = ['LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK']
SEED = 42

train_df = pd.read_csv(DATA_DIR / 'nlp_train.csv')
val_df = pd.read_csv(DATA_DIR / 'nlp_validation.csv')
test_df = pd.read_csv(DATA_DIR / 'nlp_test.csv')
unseen_df = pd.read_csv(DATA_DIR / 'nlp_final_unseen_test.csv')
ood_df = pd.read_csv(DATA_DIR / 'nlp_ood_test.csv')
adversarial_df = pd.read_csv(DATA_DIR / 'nlp_adversarial_test.csv')


current_model = joblib.load(ROOT / 'models' / 'nlp_payment_risk.joblib')
current_thresholds = json.loads((REPORTS_DIR / 'nlp_thresholds.json').read_text())


def choose_thresholds(y_true, y_prob):
    best_f1 = -1.0
    best_thresholds = {'high': 0.50, 'medium': 0.45}
    for high_t in np.linspace(0.30, 0.85, 12):
        for medium_t in np.linspace(0.20, 0.75, 12):
            preds = []
            for row in y_prob:
                probs = dict(zip(LABELS, row))
                if probs.get('HIGH_RISK', 0.0) >= high_t:
                    preds.append('HIGH_RISK')
                elif probs.get('MEDIUM_RISK', 0.0) >= medium_t:
                    preds.append('MEDIUM_RISK')
                else:
                    preds.append('LOW_RISK')
            score = f1_score(y_true, preds, labels=LABELS, average='macro', zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresholds = {'high': float(high_t), 'medium': float(medium_t)}
    return best_thresholds


def compute_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    result = {
        'macro_f1': float(f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)),
    }
    for label, p, r, f, s in zip(LABELS, precision, recall, f1, support):
        result[f'{label.lower()}_precision'] = float(p)
        result[f'{label.lower()}_recall'] = float(r)
        result[f'{label.lower()}_f1'] = float(f)
    return result


def apply_thresholds(probs, thresholds, class_order):
    preds = []
    for row in probs:
        prob_map = {label: float(row[idx]) for idx, label in enumerate(class_order)}
        if prob_map.get('HIGH_RISK', 0.0) >= thresholds['high']:
            preds.append('HIGH_RISK')
        elif prob_map.get('MEDIUM_RISK', 0.0) >= thresholds['medium']:
            preds.append('MEDIUM_RISK')
        else:
            preds.append('LOW_RISK')
    return preds


def evaluate_candidate(name, model):
    class_order = [str(v) for v in model.named_steps['classifier'].classes_]
    thresholds = choose_thresholds(val_df['label'].tolist(), model.predict_proba(val_df['text'].tolist()))
    results = {}
    for partition_name, frame in {
        'train': train_df,
        'validation': val_df,
        'test': test_df,
        'final_unseen': unseen_df,
        'ood': ood_df,
        'adversarial': adversarial_df,
    }.items():
        probs = model.predict_proba(frame['text'].tolist())
        preds = apply_thresholds(probs, thresholds, class_order)
        metrics = compute_metrics(frame['label'].tolist(), preds)
        results[partition_name] = metrics
    return {
        'name': name,
        'thresholds': thresholds,
        'train': results['train'],
        'validation': results['validation'],
        'test': results['test'],
        'final_unseen': results['final_unseen'],
        'ood': results['ood'],
        'adversarial': results['adversarial'],
    }


def build_word_char_model(C):
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents='unicode',
        norm='l2',
        min_df=1,
        max_features=5000,
    )
    char = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        sublinear_tf=True,
        strip_accents='unicode',
        norm='l2',
        min_df=1,
        max_features=4000,
    )
    vec = FeatureUnion(transformer_list=[('word', word), ('char', char)])
    return Pipeline([('vectorizer', vec), ('classifier', LogisticRegression(C=C, class_weight='balanced', max_iter=5000, random_state=SEED))])


candidate_specs = {
    'Current saved model': current_model,
    'Word+Char C=0.7': build_word_char_model(0.7),
    'Word+Char C=0.4': build_word_char_model(0.4),
    'Word+Char C=0.2': build_word_char_model(0.2),
    'Char 3-5 C=1.0': Pipeline([
        ('vectorizer', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, strip_accents='unicode', norm='l2', min_df=1, max_features=4000)),
        ('classifier', LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000, random_state=SEED))
    ]),
    'Char 3-5 C=0.4': Pipeline([
        ('vectorizer', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, strip_accents='unicode', norm='l2', min_df=1, max_features=4000)),
        ('classifier', LogisticRegression(C=0.4, class_weight='balanced', max_iter=5000, random_state=SEED))
    ]),
}


for name, model in candidate_specs.items():
    print(f'=== {name} ===')
    result = evaluate_candidate(name, model)
    print(json.dumps({
        'train_f1': result['train']['macro_f1'],
        'validation_f1': result['validation']['macro_f1'],
        'test_f1': result['test']['macro_f1'],
        'final_unseen_f1': result['final_unseen']['macro_f1'],
        'ood_f1': result['ood']['macro_f1'],
        'adversarial_f1': result['adversarial']['macro_f1'],
        'high_risk_precision': result['test']['high_risk_precision'],
        'high_risk_recall': result['test']['high_risk_recall'],
        'high_risk_f1': result['test']['high_risk_f1'],
        'gap_train_validation': result['train']['macro_f1'] - result['validation']['macro_f1'],
        'gap_train_test': result['train']['macro_f1'] - result['test']['macro_f1'],
        'gap_train_unseen': result['train']['macro_f1'] - result['final_unseen']['macro_f1'],
        'gap_train_ood': result['train']['macro_f1'] - result['ood']['macro_f1'],
        'gap_train_adversarial': result['train']['macro_f1'] - result['adversarial']['macro_f1'],
        'thresholds': result['thresholds'],
    }, indent=2))
    print()
