# NLP Final Validation

## Dataset integrity

- Training examples: 1350
- Validation examples: 450
- Final unseen examples: 1500
- OOD examples: 1500
- Adversarial examples: 1500
- Exact duplicate count: 0
- Near duplicate count: 0

## Model comparison

| Model | Train F1 | Validation F1 | Final Test F1 | Final Unseen F1 | OOD F1 | Adversarial F1 | High-Risk Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Word TF-IDF + Logistic Regression | 1.0000 | 0.9212 | 0.8459 | 0.8606 | 0.8088 | 0.8050 | 0.8333 |
| Character TF-IDF + Logistic Regression | 1.0000 | 0.9072 | 0.8191 | 0.9347 | 0.8548 | 0.8646 | 0.7600 |
| Word + Character TF-IDF + Logistic Regression | 1.0000 | 0.9196 | 0.8604 | 0.8786 | 0.7691 | 0.8377 | 0.9067 |
| Linear SVM (calibrated) | 1.0000 | 0.9255 | 0.8658 | 0.8940 | 0.7397 | 0.8456 | 0.8600 |

## Overfitting analysis

- Train -> Validation gap: 0.0928
- Train -> Final unseen gap: 0.0653
- Train -> OOD gap: 0.1452
- Train -> Adversarial gap: 0.1354
- Overfitting label: MODERATE

## Final recommendation

The safest choice is Character TF-IDF + Logistic Regression because it preserves the strongest OOD and adversarial generalization while remaining competitive on final unseen and test performance.