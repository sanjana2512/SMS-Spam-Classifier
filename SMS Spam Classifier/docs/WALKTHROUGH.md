# Build and understand it one stage at a time

You know how to build HTTP services and React interfaces. The new part is learning
a rule from examples instead of writing all the rules yourself. Read one stage,
run its command, and inspect its output before moving on.

## Stage 0: just enough Python

Python uses indentation where C++/JavaScript use braces. Variables don't need
declarations, but values still have types. A function starts with `def`:

```python
def greet(name: str) -> str:
    return f"Hello, {name}"
```

`name: str` and `-> str` are type hints, roughly like TypeScript annotations. Hints
usually do not enforce types at runtime. The API's Pydantic model does enforce
its declared validation rules when requests arrive.

| Familiar concept | Python in this project |
| --- | --- |
| `const labels = ["ham", "spam"]` | `labels = ["ham", "spam"]` (a list) |
| JavaScript object | `{"message": "hello"}` (a dictionary) |
| `import ... from ...` | `from sklearn.pipeline import Pipeline` |
| Function | `def clean_data(raw):` followed by an indented body |
| Class | `class PredictRequest(BaseModel):` |
| `null` | `None` |
| Boolean | `True` / `False` |
| `throw new Error(...)` | `raise ValueError(...)` |
| Template literal | `f"Saved: {path}"` |
| `package.json` dependencies | `requirements.txt` / `requirements-lock.txt` |

`.venv` is an isolated place for this project's Python interpreter and packages.
`python -m spam_classifier.train` runs a module inside our package. A folder with
`__init__.py` is a Python package. The `if __name__ == "__main__":` block runs a
file's command entry point without running it when another file imports it.

`Path` represents filesystem paths. `with ...:` closes resources or performs
cleanup even if an exception happens. `paths.py` calculates paths from its own
location so the API can find the same model reliably.

## Stage 1: understand the dataset

Read `spam_classifier/download.py` and `data/SOURCE.md`, then run:

```powershell
.\.venv\Scripts\python.exe -m spam_classifier.download
```

We download the original UCI archive, not an unrelated cleaned CSV. Each line
contains a known label and a message. These known answers make this **supervised
learning**. The task is **binary classification** because there are two labels.

The downloader keeps the original README and writes `source.json`. SHA-256 is
a fingerprint of a file: it lets you identify which exact dataset produced a
report. Downloading does not train a model.

## Stage 2: inspect and clean

Read `spam_classifier/data.py`, then run:

```powershell
.\.venv\Scripts\python.exe -m spam_classifier.train --inspect-only
```

Pandas gives us a `DataFrame`, which you can think of as an in-memory table.
`data["message"]` is one column, called a `Series`. Here are the main operations:

- `isna().sum()` counts missing values in each column.
- `value_counts()` counts messages of each label.
- `groupby("message")["label"].nunique()` counts different labels for each message.
- `drop_duplicates(subset="message")` keeps one row per message.
- `reset_index(drop=True)` gives the cleaned table simple sequential row numbers.

Missing values are different from blank strings, so we inspect both. We remove
invalid rows rather than inventing an SMS or its answer. Unexpected labels cause
an error. A message tagged both ham and spam is ambiguous; we flag every such row
in `reports/conflicts.csv` and exclude the message entirely.

Our duplicate identity rule strips surrounding whitespace, preserving the rest
of the message. Deduplication happens **before splitting**: otherwise the same
message could appear in training and testing, making memorization look like
generalization. This does not catch near duplicates or repeated campaigns.

Look at `reports/data_audit.json`: the raw corpus is about 13.40% spam; after
deduplication it is about 12.44% spam. This is **class imbalance**. A model saying
ham for everything can already look good if you only measure accuracy.

## Stage 3: make a fair train/test split

Read `split_data()` in `spam_classifier/train.py`:

```python
train_test_split(
    data["message"], data["label"],
    test_size=0.20, random_state=42, stratify=data["label"],
)
```

`X` means inputs (messages) and `y` means targets (labels), by convention.
The training set teaches the model; the test set estimates performance on unseen
messages. Here 80% is for training and 20% is held out.

`stratify` keeps roughly the same ham/spam proportions in each set. `random_state`
is a fixed random seed, so the same cleaned data and environment give the same
split. Reproducibility is not proof that the model works; it makes comparisons
auditable. The report saves split indices, dependency versions, and the data hash.

## Stage 4: turn text into numbers and learn a boundary

Read `build_pipeline()` in `spam_classifier/train.py`:

```python
Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(
        C=1.0, solver="liblinear", max_iter=1000, random_state=42
    )),
])
```

**TF-IDF** turns a message into a numeric vector with one position per learned
word. It considers word frequency within a message and how widespread the word
is across training messages. A word appearing in almost every message is less
distinctive than one concentrated in particular messages. Scikit-learn applies
smoothing and normalizes the vectors by default. This is a sparse bag-of-words
representation: most positions are zero, and word order is not preserved.

The vectorizer learns its vocabulary and IDF values from training text only.
Its defaults lowercase text and tokenize words of at least two characters.
We do not add stemming, stop-word lists, or hand-written feature rules in v1.
Unknown words at prediction time are ignored; a message with no recognized
tokens effectively relies on the classifier's intercept.

**Logistic regression** learns one weight per feature plus an intercept. A
weighted sum passes through a sigmoid function, producing an estimated spam
probability. Despite its name, logistic regression is a classifier. It can learn
that certain words are associated with spam without us hard-coding those words.
Its regularization discourages overly large weights; smaller `C` means stronger
regularization. `max_iter` caps optimization steps; it is not a search over models.

Calling `pipeline.fit(X_train, y_train)` learns both the text representation and
the classification weights. Calling `pipeline.predict(X_test)` transforms using
the existing vocabulary/IDF and applies the existing weights. It does not learn
from the test messages.

Fitting TF-IDF on all messages before splitting is **data leakage**, even though
TF-IDF does not read labels. It has learned statistics about held-out inputs.
Scikit-learn [recommends fitting transforms only on training data and using pipelines](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

This is different from RAG: there is no retrieval index, prompt, or generation
step. Here, `fit` learns feature statistics and coefficients from labeled examples.

## Stage 5: compare and evaluate

Run training once you understand the earlier stages:

```powershell
.\.venv\Scripts\python.exe -m spam_classifier.train
```

The majority baseline uses `DummyClassifier(strategy="most_frequent")`. It learns
the most common training label, then always returns it. Both models are evaluated
on exactly the same test messages.

Treat spam as the positive class:

| Term | Meaning | Actual model count |
| --- | --- | ---: |
| True positive (TP) | Spam correctly flagged | 92 |
| False positive (FP) | Legitimate message flagged as spam | 4 |
| False negative (FN) | Spam allowed through as ham | 36 |
| True negative (TN) | Legitimate message allowed through | 900 |

- **Precision = TP / (TP + FP)**: of flagged messages, how many really are spam?
  Here, `92 / 96 = 95.83%`.
- **Recall = TP / (TP + FN)**: of all spam, how much did we catch?
  Here, `92 / 128 = 71.88%`.
- **F1 = 2 × precision × recall / (precision + recall)**: combines the two,
  penalizing a very low value on either side. Here, `82.14%`.
- **Accuracy = (TP + TN) / all messages**: overall fraction correct.
  Here, `992 / 1032 = 96.12%`.
- **Support**: number of actual examples of a class in the test set.

Our confusion matrix order is `[ham, spam]`, with actual labels in rows and
predictions in columns. Always state this order; otherwise a matrix is ambiguous.
The JSON report includes both classes, **macro average** (equal weight per class)
and **weighted average** (weight by support). These answer different questions.

Always predicting ham gets 87.60% accuracy and catches zero spam. Its spam precision
has no defined denominator; `zero_division=0` reports zero without claiming a
meaningful precision estimate. The comparison shows why a majority baseline matters.

We chose the settings before looking at the test scores. Choosing settings after
seeing these results would turn this test set into a validation set. Future
experiments should tune using training-only validation or cross-validation,
keeping a new final test set untouched. Re-running the same evaluation is not
independent new evidence.

## Stage 6: save and reload the whole learned transformation

Training ends with:

```python
joblib.dump(pipeline, MODEL_PATH)
```

The API starts with:

```python
pipeline = joblib.load(path)
```

The file contains the fitted vocabulary, IDF, model coefficients, and class order.
Saving just the classifier would lose the meaning of each numeric input position.
Never fit a new vectorizer per request: its positions would mean different words.
Keep the dependency versions used for training and only load trusted artifacts.

## Stage 7: expose a familiar HTTP contract

Read `spam_classifier/api.py`. This is similar to Express:

| Express/Node concept | This API |
| --- | --- |
| `app.post('/predict', handler)` | `@app.post("/predict")` above a function |
| `req.body` | `payload: PredictRequest` |
| Zod/Joi validation | Pydantic fields and validators |
| Startup initialization | FastAPI lifespan context |
| `res.json(...)` | Return a `PredictResponse` |

The model is loaded once at startup, rather than on every request. The endpoint
validates the JSON, strips surrounding whitespace, predicts a label, and returns
the estimated spam probability. It finds the spam column through `classes_`
instead of assuming the order.

The request has a strict string type, a length limit, a non-whitespace check,
and rejects unexpected fields. Invalid requests get HTTP 422 automatically;
FastAPI [documents typed request bodies](https://fastapi.tiangolo.com/tutorial/body/)
and generates the interactive `/docs` page.

The prediction route is an ordinary `def` function because model inference is
synchronous work. FastAPI runs synchronous handlers in its thread pool. An
`async def` wrapper alone would not make CPU work nonblocking.

## Stage 8: interface and tests

`static/index.html` uses the same `fetch` you know from React. It posts JSON to
`/predict`, handles loading and errors, and displays the response using `textContent`.
Example buttons fill the editor without submitting; the character counter updates
as you type. A result card shows the returned spam probability as text and a bar.
Editing or clearing the message resets the previous result so it cannot be confused
with a prediction for the new text. Ctrl+Enter also submits the form.
The API serves the page on the same origin, so no separate frontend server or
CORS configuration is needed. The UI's checks help the user; server validation
is still required because anyone can call an API directly.

Tests use pytest (roughly the role Jest plays in JavaScript) and FastAPI's
`TestClient`. Temporary files isolate saved models. Tiny synthetic examples
verify mechanics without downloading data or asserting made-up UCI scores.
Persistence tests compare predictions and probabilities before and after loading.

## What to understand before adding features

Be able to trace one SMS through input validation, the fitted vectorizer, model
probability, predicted label, and JSON response. Explain why duplicates are removed
before splitting, why TF-IDF is learned on training data, and why a model with
high accuracy can still miss every spam message. That is the foundation of this v1.
