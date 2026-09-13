from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.pipeline import choose_path, read_table, infer_map, build_supervised, add_features, make_preprocessor

MD = ROOT / 'models'
RP = ROOT / 'reports'
MD.mkdir(exist_ok=True)
RP.mkdir(exist_ok=True)
DOCS = ROOT / 'docs'
DOCS.mkdir(exist_ok=True)

def split(df):
    # Splits chronologically: 50% Train, 20% Val, 15% Test, 15% OOT
    n = len(df)
    train_end = int(n * 0.50)
    val_end = int(n * 0.70)
    test_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:test_end], df.iloc[test_end:]

def evaluate_classification(y_true, y_pred, y_prob, name="Model"):
    metrics = {
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
        'PR-AUC': float(average_precision_score(y_true, y_prob)),
        'ROC-AUC': float(roc_auc_score(y_true, y_prob)),
        'Brier_Score': float(brier_score_loss(y_true, y_prob))
    }
    print(f"[{name}] " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    return metrics

def format_markdown_table(data_dict, title):
    lines = [f"### {title}"]
    if not data_dict:
        return ""
    headers = ["Metric", "Train", "Validation", "Test", "OOT"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    
    keys = list(data_dict.get('Test', {}).keys())
    for k in keys:
        row = [k]
        for split_name in ["Train", "Validation", "Test", "OOT"]:
            val = data_dict.get(split_name, {}).get(k, "N/A")
            row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n\n"

def build_nlp_demo():
    # Controlled, diverse payment-communication examples for an optional NLP component.
    texts_pos = [
      'We expect a short delay while the payment receives internal approval.',
      'Our finance team will process the transfer next week, after approval.',
      'Payment may be delayed because the invoice is still under review.',
      'We are waiting for treasury approval before releasing the payment.',
      'The transfer is scheduled after our internal reconciliation is completed.',
      'We anticipate paying shortly, but the current processing cycle may take longer.',
      'There is a temporary cash flow constraint and the payment will be late.',
      'We cannot complete the payment by the due date; we expect to settle it soon.',
      'The remittance is pending and may miss the agreed payment date.',
      'Our accounting department requested additional time to complete payment.'
    ]
    texts_neg = [
      'Payment has been scheduled and will be completed on the agreed date.',
      'The invoice was approved and the bank transfer is ready for release.',
      'We have completed the payment and attached the remittance advice.',
      'Funds have been transferred successfully for this invoice.',
      'Our finance team confirmed payment for the original due date.',
      'The invoice is approved with no expected payment issues.',
      'Remittance advice is available; the transfer will settle as planned.',
      'We have no changes to the agreed payment schedule.',
      'Payment processing is complete and the transaction is confirmed.',
      'The outstanding invoice will be paid according to the contract terms.'
    ]
    variants = ['today', 'tomorrow', 'this week', 'this month', 'as agreed', 'after approval', 'after reconciliation']
    rows = []
    for y, base in [(1, texts_pos), (0, texts_neg)]:
        for i, t in enumerate(base):
            for v in variants:
                rows.append((f'{t} We expect completion {v}.', y))
    
    df = pd.DataFrame(rows, columns=['text', 'label']).sample(frac=1, random_state=42).reset_index(drop=True)
    tr, va, te, oot = split(df)
    
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True, max_features=5000)),
        ('model', CalibratedClassifierCV(LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42), cv=3))
    ])
    pipe.fit(tr.text, tr.label)
    
    metrics = {
        "Train": evaluate_classification(tr.label, pipe.predict(tr.text), pipe.predict_proba(tr.text)[:, 1], "NLP_Train"),
        "Validation": evaluate_classification(va.label, pipe.predict(va.text), pipe.predict_proba(va.text)[:, 1], "NLP_Val"),
        "Test": evaluate_classification(te.label, pipe.predict(te.text), pipe.predict_proba(te.text)[:, 1], "NLP_Test"),
        "OOT": evaluate_classification(oot.label, pipe.predict(oot.text), pipe.predict_proba(oot.text)[:, 1], "NLP_OOT")
    }
    
    joblib.dump(pipe, MD / 'nlp_payment_risk.joblib')
    (MD / 'nlp_payment_risk_metadata.json').write_text(json.dumps({"model_name": "TF-IDF + Calibrated LogisticRegression", "metrics": metrics}, indent=2))
    return format_markdown_table(metrics, "NLP Payment Risk Prediction (Text Classification)")


def build_tabular_variants():
    p = choose_path('demo')
    raw = read_table(p)
    m = infer_map(raw)
    x = build_supervised(raw, m)
    x, num, cat, features = add_features(x, m)
    
    tr, va, te, oot = split(x)
    Xtr, ytr = tr[features], tr.late_payment
    Xv, yv = va[features], va.late_payment
    Xt, yt = te[features], te.late_payment
    Xo, yo = oot[features], oot.late_payment
    
    models = {
        'DummyBaseline': Pipeline([('prep', make_preprocessor(num, cat)), ('model', DummyClassifier(strategy="prior"))]),
        'logistic_regression': Pipeline([('prep', make_preprocessor(num, cat)), ('model', LogisticRegression(max_iter=3000, class_weight='balanced'))]),
        'random_forest': Pipeline([('prep', make_preprocessor(num, cat)), ('model', RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', random_state=42, n_jobs=-1))]),
        'hist_gradient_boosting': Pipeline([('prep', make_preprocessor(num, cat)), ('model', HistGradientBoostingClassifier(random_state=42))])
    }
    
    best_model = None
    best_score = -1
    best_name = ""
    best_metrics = {}
    
    for name, pipe in models.items():
        if name != 'DummyBaseline':
            # Calibrate pipeline estimator using prefit
            pipe.fit(Xtr, ytr)
            calibrated_model = CalibratedClassifierCV(estimator=pipe.named_steps['model'], method='isotonic', cv="prefit")
            calibrated_model.fit(pipe.named_steps['prep'].transform(Xv), yv)
            # Reconstruct pipeline
            pipe = Pipeline([('prep', pipe.named_steps['prep']), ('model', calibrated_model)])
        else:
            pipe.fit(Xtr, ytr)
            
        y_prob = pipe.predict_proba(Xv)[:, 1]
        score = average_precision_score(yv, y_prob)
        print(f"{name} PR-AUC on val: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = pipe
            best_name = name
            
            best_metrics = {
                "Train": evaluate_classification(ytr, pipe.predict(Xtr), pipe.predict_proba(Xtr)[:, 1], name=name + "_Train"),
                "Validation": evaluate_classification(yv, pipe.predict(Xv), y_prob, name=name + "_Val"),
                "Test": evaluate_classification(yt, pipe.predict(Xt), pipe.predict_proba(Xt)[:, 1], name=name + "_Test"),
                "OOT": evaluate_classification(yo, pipe.predict(Xo), pipe.predict_proba(Xo)[:, 1], name=name + "_OOT")
            }
            
    print(f"Best tabular model: {best_name} with Val PR-AUC: {best_score:.4f}")
    
    joblib.dump(best_model, MD / 'classifier.joblib')
    # Save legacy models for backward compatibility if needed, but we select best
    for name, pipe in models.items():
        if name != 'DummyBaseline':
            joblib.dump(pipe, MD / f'{name}.joblib')
            
    metadata = {
        "model_name": best_name,
        "features": features,
        "num_features": num,
        "cat_features": cat,
        "metrics": best_metrics
    }
    (MD / 'classifier_metadata.json').write_text(json.dumps(metadata, indent=2))
    
    return format_markdown_table(best_metrics, f"Invoice Payment Risk Model: {best_name}")


if __name__ == '__main__':
    print("="*50 + "\nTASK 3: INVOICE PAYMENT RISK (Tabular)\n" + "="*50)
    tabular_report = build_tabular_variants()
    
    print("\n" + "="*50 + "\nTASK 4: NLP PAYMENT RISK\n" + "="*50)
    nlp_report = build_nlp_demo()
    
    # Append to report
    report_path = DOCS / "model_quality_report.md"
    content = ""
    if report_path.exists():
        content = report_path.read_text()
        
    content += "\n\n" + tabular_report + "\n\n" + nlp_report
    report_path.write_text(content)
    print("\nAppended Invoice ML models metrics to model_quality_report.md")

