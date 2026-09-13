from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
REPORTS_DIR = ROOT / 'reports'

LABELS = ['LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK']
SEED = 42

train_df = pd.read_csv(DATA_DIR / 'nlp_train.csv')
val_df = pd.read_csv(DATA_DIR / 'nlp_validation.csv')
combined_df = pd.concat([train_df, val_df], ignore_index=True)


def build_candidate_specs():
    specs = []
    char_ranges = [
        ('3-5', (3, 5)),
        ('3-6', (3, 6)),
        ('4-6', (4, 6)),
        ('4-7', (4, 7)),
    ]
    max_features_opts = [4000]
    for label, ngram_range in char_ranges:
        for C in [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
            vec = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=ngram_range,
                sublinear_tf=True,
                strip_accents='unicode',
                norm='l2',
                min_df=1,
                max_features=max_features_opts[0],
            )
            specs.append(
                {
                    'model_name': f'Char {label} C={C}',
                    'kind': 'char',
                    'vectorizer': vec,
                    'C': C,
                    'notes': f'char_wb {label}',
                }
            )

    for C in [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
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
        specs.append(
            {
                'model_name': f'Word+Char C={C}',
                'kind': 'word_char',
                'vectorizer': vec,
                'C': C,
                'notes': 'word+char',
            }
        )

    return specs


def choose_thresholds(y_true, y_prob, class_order):
    best_f1 = -1.0
    best_thresholds = {'high': 0.50, 'medium': 0.45}
    for high_t in np.linspace(0.30, 0.85, 12):
        for medium_t in np.linspace(0.20, 0.75, 12):
            preds = []
            for row in y_prob:
                prob_map = {label: float(row[idx]) for idx, label in enumerate(class_order)}
                if prob_map.get('HIGH_RISK', 0.0) >= high_t:
                    preds.append('HIGH_RISK')
                elif prob_map.get('MEDIUM_RISK', 0.0) >= medium_t:
                    preds.append('MEDIUM_RISK')
                else:
                    preds.append('LOW_RISK')
            score = f1_score(y_true, preds, labels=LABELS, average='macro', zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresholds = {'high': float(high_t), 'medium': float(medium_t)}
    return best_thresholds, best_f1


def apply_thresholds(y_prob, thresholds, class_order):
    preds = []
    for row in y_prob:
        prob_map = {label: float(row[idx]) for idx, label in enumerate(class_order)}
        if prob_map.get('HIGH_RISK', 0.0) >= thresholds['high']:
            preds.append('HIGH_RISK')
        elif prob_map.get('MEDIUM_RISK', 0.0) >= thresholds['medium']:
            preds.append('MEDIUM_RISK')
        else:
            preds.append('LOW_RISK')
    return preds


def metric_summary(y_true, y_pred, class_order):
    metrics = {}
    for label in LABELS:
        if label not in class_order:
            continue
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    class_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    metrics['precision'] = {label: float(precision[class_to_idx[label]]) for label in LABELS}
    metrics['recall'] = {label: float(recall[class_to_idx[label]]) for label in LABELS}
    metrics['f1'] = {label: float(f1[class_to_idx[label]]) for label in LABELS}
    metrics['support'] = {label: int(support[class_to_idx[label]]) for label in LABELS}
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0))
    return metrics


def evaluate_candidate(spec):
    model = Pipeline([
        ('vectorizer', spec['vectorizer']),
        ('classifier', LogisticRegression(C=spec['C'], class_weight='balanced', max_iter=5000, random_state=SEED))
    ])
    model.fit(train_df['text'].tolist(), train_df['label'].tolist())

    class_order = [str(v) for v in model.named_steps['classifier'].classes_]
    val_prob = model.predict_proba(val_df['text'].tolist())
    thresholds, _ = choose_thresholds(val_df['label'].tolist(), val_prob, class_order)
    val_preds = apply_thresholds(val_prob, thresholds, class_order)

    train_prob = model.predict_proba(train_df['text'].tolist())
    train_preds = apply_thresholds(train_prob, thresholds, class_order)

    train_metrics = metric_summary(train_df['label'].tolist(), train_preds, class_order)
    val_metrics = metric_summary(val_df['label'].tolist(), val_preds, class_order)

    return {
        'model_name': spec['model_name'],
        'kind': spec['kind'],
        'C': spec['C'],
        'notes': spec['notes'],
        'train_f1': train_metrics['macro_f1'],
        'validation_f1': val_metrics['macro_f1'],
        'gap': train_metrics['macro_f1'] - val_metrics['macro_f1'],
        'high_risk_precision': val_metrics['precision']['HIGH_RISK'],
        'high_risk_recall': val_metrics['recall']['HIGH_RISK'],
        'high_risk_f1': val_metrics['f1']['HIGH_RISK'],
        'thresholds': thresholds,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
    }


candidates = build_candidate_specs()
results = [evaluate_candidate(spec) for spec in candidates]

for result in sorted(results, key=lambda x: (x['validation_f1'], -x['gap'], x['high_risk_recall'], -x['high_risk_f1']), reverse=True):
    print(
        json.dumps(
            {
                'model': result['model_name'],
                'train_f1': round(result['train_f1'], 6),
                'validation_f1': round(result['validation_f1'], 6),
                'gap': round(result['gap'], 6),
                'high_risk_recall': round(result['high_risk_recall'], 6),
                'high_risk_f1': round(result['high_risk_f1'], 6),
                'thresholds': result['thresholds'],
            },
            indent=2,
        )
    )
    print('---')

comparison_rows = []
for result in results:
    comparison_rows.append(
        {
            'model': result['model_name'],
            'kind': result['kind'],
            'regularization_C': result['C'],
            'notes': result['notes'],
            'train_f1': round(result['train_f1'], 6),
            'validation_f1': round(result['validation_f1'], 6),
            'train_validation_gap': round(result['gap'], 6),
            'high_risk_precision': round(result['high_risk_precision'], 6),
            'high_risk_recall': round(result['high_risk_recall'], 6),
            'high_risk_f1': round(result['high_risk_f1'], 6),
            'thresholds_high': result['thresholds']['high'],
            'thresholds_medium': result['thresholds']['medium'],
        }
    )

comparison_df = pd.DataFrame(comparison_rows)
comparison_df = comparison_df.sort_values(['validation_f1', 'train_validation_gap', 'high_risk_recall'], ascending=[False, True, False])
comparison_df.to_csv(REPORTS_DIR / 'nlp_optimization_comparison.csv', index=False)
print(f'Wrote {REPORTS_DIR / "nlp_optimization_comparison.csv"}')
