# Main interview questions

## 1. Explain your project in a minute

"I built a small API that labels SMS messages as ham or spam. I used the original
UCI SMS Spam Collection, audited missing values and class balance, flagged
conflicting labels, and removed duplicate messages before an 80/20 stratified
split. A Scikit-learn pipeline learns TF-IDF features and logistic regression
on training data only. I compared it against a majority baseline, reported
precision, recall, F1, and confusion matrices on the same held-out test set,
and saved the complete pipeline for FastAPI inference. A simple browser interface
and tests cover the data and request contracts."

## 2. Why logistic regression instead of a large language model?

This is a small labeled text classification task. A linear model is cheap to
train, runs locally, and provides an understandable baseline. A large model adds
complexity that this first version does not need. We have not demonstrated that
logistic regression is the best possible model; we established a fair starting point.

## 3. What is TF-IDF? How does it differ from embeddings?

TF-IDF assigns numeric weights to vocabulary terms using their frequency within
messages and across the training collection. It creates a sparse representation
based on words, usually without order. Dense embeddings can capture semantic
relationships between different words; that is a different representation, with
different dependencies and tradeoffs. This project uses no embeddings or RAG.

## 4. Why a pipeline?

It connects text transformation and prediction behind one `fit`/`predict`
interface. Only training data fits the vocabulary, IDF, and classifier. The same
fitted transformations are reused for test data and API requests. Saving it as
one artifact reduces training/inference inconsistencies.

## 5. What is data leakage in your project?

Examples include fitting TF-IDF on all messages before splitting, allowing an
identical SMS into both sets, or selecting a threshold based on test results.
We clean/deduplicate before splitting, fit the pipeline on training data only,
and use predetermined settings. Near duplicates may still cross the split;
campaign-aware grouping would be a later improvement.

## 6. How did you handle duplicates and label conflicts?

Message identity ignores surrounding whitespace and preserves internal text/case.
Messages assigned both labels are written to a conflict report and excluded
entirely; we don't arbitrarily choose their label. Other repeated messages keep
one row before splitting. This downloaded corpus had 414 repeated rows and no
conflicting messages. Synthetic tests verify the conflict behavior nonetheless.

## 7. Why stratify? Why a fixed seed?

Stratification preserves approximately the same class proportions in training and
test sets, which matters for an imbalanced dataset. A fixed seed reproduces split
membership. The seed is not a parameter to search for a favorable result; we
also record the dataset checksum and dependency versions.

## 8. Why isn't accuracy enough?

The test set has 904 ham and 128 spam messages. Always saying ham achieves 87.60%
accuracy while finding zero spam. Spam precision and recall expose this failure.
Always include a simple baseline to show that the trained model adds value.

## 9. Explain precision, recall, and your confusion matrix

Spam is positive. The model has TN=900, FP=4, FN=36, TP=92. Precision is
`92 / (92 + 4) = 95.83%`; recall is `92 / (92 + 36) = 71.88%`; F1 is 82.14%.
Rows are actual labels and columns are predictions, ordered ham then spam.
Precision asks whether flags are trustworthy; recall asks how much spam was found.

## 10. Which matters more: precision or recall?

It depends on the cost of mistakes. Automatically hiding an important legitimate
message makes false positives expensive, so precision may be critical. Letting
harmful spam through makes false negatives expensive, so recall may matter more.
The project's scores show the tradeoff; they do not establish a universally
correct operating point.

## 11. How would you improve recall without corrupting the evaluation?

Consider threshold changes, class weighting, TF-IDF variants, or another simple
classifier using validation data/cross-validation within the training partition.
Keep transformations inside the pipeline in each fold. Choose a metric and error
cost in advance. Because this test set's results are already known, reserve a
new untouched final test set for later experiments. We did not do these searches in v1.

## 12. Is the spam probability a confidence guarantee?

No. It is the logistic model's probability estimate under its learned features
and training distribution. We have not evaluated calibration, and unfamiliar
languages or current spam may behave differently. If probabilities drive a
product decision, evaluate calibration on suitable validation data.

## 13. What happens to a word the model has never seen?

The vectorizer ignores it. If all tokens are unknown, the numeric vector is zero
and the classifier relies on its intercept. Accepting Unicode input does not
prove good classification performance in every language.

## 14. Why save the vectorizer together with the classifier?

Each model coefficient belongs to a specific vocabulary position. Rebuilding
the vocabulary could assign that position to a different word. We serialize the
complete fitted pipeline, reload once at API startup, and test that predictions
and probabilities survive the round trip. Only trusted artifacts are loaded.

## 15. What input validation did you add?

`message` must be a string with 1–5,000 characters and contain non-whitespace.
Missing, null, numeric, extra-field, oversized, and malformed JSON requests get
422 responses. The server validates independently of the interface. The API
returns a ham/spam label and a numeric spam probability.

## 16. What are the limitations and next steps?

The corpus is older and mainly English. A random message split does not fully
measure performance on new campaigns or future messages. Exact deduplication
does not remove all near duplicates. This model missed 36 of 128 test spam
messages. Later work could collect representative current labeled examples,
use campaign/time-aware evaluation, tune on training-only validation data, and
monitor drift and real error costs. Production hosting would also require
decisions about authentication, request limits, privacy, and observability.

## 17. What does regularization do? What does `C` mean?

Regularization discourages overly large model weights, helping reduce overfitting.
For Scikit-learn logistic regression, `C` controls inverse regularization strength:
smaller `C` means stronger regularization. This version fixes `C=1.0` rather than
selecting it using test scores.

## 18. How do you test an ML API without brittle score assertions?

Test contracts and failure modes: no duplicates across sets, train-only vocabulary,
correct confusion-matrix orientation, serialization equivalence, loaded-model/API
agreement, validation errors, and missing-model startup behavior. Keep evaluation
reports separate from unit tests; a test should not pretend synthetic data proves
real-world accuracy.

