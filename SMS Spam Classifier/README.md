# SMS Spam Classification API

A small learning project: **raw SMS → TF-IDF → logistic regression → FastAPI**.
No paid APIs, language models, deep learning, or Node build. `ham` means legitimate SMS;
`spam` means unsolicited/promotional spam as labeled in this dataset.

Start with the commands below, then read [the staged walkthrough](docs/WALKTHROUGH.md)
and [interview questions](docs/INTERVIEW.md). The walkthrough connects Python to
JavaScript/C++ and explains what each file does.

## 1. Set up Python

Verified on **Windows with Python 3.14.2**. Use Python 3.14 for the same environment.
Run these commands in PowerShell from this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

An environment is already created in this workspace. Re-running installation is safe.
The explicit executable paths avoid PowerShell activation-policy problems.
`requirements.txt` lists direct dependencies; `requirements-lock.txt` pins the full
verified environment. Install the lock file to reproduce this run.

On macOS/Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`
(use `python3 -m venv .venv` to create it).

## 2. Download and inspect

```powershell
.\.venv\Scripts\python.exe -m spam_classifier.download
.\.venv\Scripts\python.exe -m spam_classifier.train --inspect-only
```

The downloader uses the original [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
Source citation, license, checksums, and duplicate/conflict rules are documented in
[data/SOURCE.md](data/SOURCE.md). The original archive README is preserved locally.
Downloading requires internet access; subsequent training and predictions run locally.
An existing dataset is reused without overwriting it.

Inspection writes `reports/data_audit.json` and `reports/conflicts.csv`.
Conflicts are messages assigned both labels; **all rows for those messages are excluded**.
The conflict file has a header even when there are no conflicts.

Observed in the downloaded file:

| Check | Result |
| --- | ---: |
| Raw messages | 5,574 |
| Raw ham / spam | 4,827 / 747 |
| Missing or blank rows | 0 |
| Repeated message rows beyond the first | 414 |
| Messages with conflicting labels | 0 |
| Unique messages after cleaning | 5,160 |
| Clean ham / spam | 4,518 / 642 |

These numbers come from reading the downloaded file, rather than copying counts
from a different CSV version of the corpus.

## 3. Train, evaluate, and save

```powershell
.\.venv\Scripts\python.exe -m spam_classifier.train
```

Training uses a reproducible **80/20 stratified split with seed 42**: 4,128 training
messages and 1,032 test messages. Duplicates are removed first. TF-IDF vocabulary/IDF
and classifier weights are fitted on training messages only. The majority baseline
learns its single label from training labels only.

The model is fixed in advance: default TF-IDF, logistic regression with `C=1.0`,
`solver="liblinear"`, `max_iter=1000`, and the standard 0.5 decision boundary.
There is no hyperparameter search, threshold tuning, or refitting on the test set.

Actual held-out results (spam is the positive class):

| Model | Spam precision | Spam recall | Spam F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Always predict ham | 0.00%* | 0.00% | 0.00% | 87.60% |
| TF-IDF + logistic regression | 95.83% | 71.88% | 82.14% | 96.12% |

*Precision is undefined when no spam is predicted; the report uses `zero_division=0`.

Model confusion matrix, with rows = actual and columns = predicted:

| | Predicted ham | Predicted spam |
| --- | ---: | ---: |
| Actual ham | 900 | 4 |
| Actual spam | 36 | 92 |

The baseline matrix is `[[904, 0], [128, 0]]` in the same order. The model caught
92 of 128 spam messages and incorrectly flagged 4 legitimate messages. This is
a useful first model with a substantial missed-spam rate; its probability estimates
have not been separately calibrated or validated as confidence guarantees.

Generated files:

- [reports/evaluation.md](reports/evaluation.md): readable results and both confusion matrices.
- [reports/metrics.json](reports/metrics.json): precision, recall, F1, and support for both classes,
  macro/weighted averages, split membership, versions, and the dataset checksum.
- `artifacts/spam_pipeline.joblib`: the complete fitted TF-IDF and classifier pipeline.

The saved model is the model evaluated above. Re-running training with the same
data/environment reproduces the split; it does not make the test set fresh again.

## 4. Start the API and interface

```powershell
.\.venv\Scripts\python.exe -m uvicorn spam_classifier.api:app --reload
```

Open **http://127.0.0.1:8000** for the interface or **http://127.0.0.1:8000/docs**
for interactive API documentation. Keep this terminal running; Ctrl+C stops it.

The interface includes quick-fill examples, a character counter, a clear button,
and a visual spam-probability result. Ctrl+Enter (Cmd+Enter on Mac) checks the message.
The layout adapts to mobile screens and provides loading and error feedback.

Test the endpoint from another PowerShell terminal:

```powershell
$body = @{ message = "Are we still meeting at 6?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $body
```

`POST /predict` accepts exactly one field:

```json
{"message": "Are we still meeting at 6?"}
```

It returns a `label` (`ham` or `spam`) and `spam_probability` (a number from 0 to 1).
The exact probability depends on the fitted model; no example score is fabricated here.
Empty/whitespace-only text, non-string values, missing/extra fields, and messages
longer than 5,000 characters return HTTP **422**. Surrounding whitespace is stripped.
The API accepts Unicode, although this mainly English corpus does not establish
accuracy on other languages. Unseen words are ignored by the learned vocabulary.

At startup FastAPI loads the pipeline once using `joblib.load`. It fails with an
instruction to train if the file is absent. `SMS_MODEL_PATH` can select another local
artifact. Only load trusted artifacts: joblib loading can execute Python code, and
Scikit-learn models should use the same dependency versions as training.

## 5. Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover missing/blank rows, conflicting labels, deduplication before splitting,
reproducible stratification, train-only vocabulary, pipeline persistence, metric
orientation, API loading/predictions, invalid requests, Unicode/length boundaries,
and serving the interface. Tests use small synthetic data, with no network download.

## Project map

```text
spam_classifier/
  download.py      UCI download and provenance
  data.py          Load, inspect, and deduplicate messages
  train.py         Split, fit, compare, report, save
  api.py           Validated HTTP endpoint and interface route
  paths.py         Shared project paths
static/index.html  Small HTML/CSS/JavaScript interface
tests/             Data, pipeline, and API tests
docs/              Learning guide and interview questions
data/SOURCE.md     Dataset attribution and cleaning policy
reports/           Actual audit and held-out results
artifacts/         Generated fitted pipeline
```

This first version is a local learning demo. Exact duplicates are handled; near
duplicates and shared spam campaigns can still cross the random split. The older
corpus is not evidence of performance on current SMS, phishing, or other languages.
If you later experiment with parameters, use validation splits/cross-validation
inside training data and reserve a new untouched test set for the final comparison.
