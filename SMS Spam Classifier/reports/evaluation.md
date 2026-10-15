# Evaluation from the actual run

Fixed 80/20 stratified split, seed 42. No test-set tuning.

Spam is the positive class. Metrics below are for spam.

| Model | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| majority_baseline | 0.0000 | 0.0000 | 0.0000 | 0.8760 |
| tfidf_logistic_regression | 0.9583 | 0.7188 | 0.8214 | 0.9612 |

## majority_baseline: confusion matrix

Rows = actual, columns = predicted. Order: ham, spam.

| | Predicted ham | Predicted spam |
| --- | ---: | ---: |
| Actual ham | 904 | 0 |
| Actual spam | 128 | 0 |

## tfidf_logistic_regression: confusion matrix

Rows = actual, columns = predicted. Order: ham, spam.

| | Predicted ham | Predicted spam |
| --- | ---: | ---: |
| Actual ham | 900 | 4 |
| Actual spam | 36 | 92 |

Undefined precision is reported as 0 when a model predicts no spam.
See metrics.json for both classes, macro/weighted averages, and split details.
