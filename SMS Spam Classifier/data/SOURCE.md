# Dataset source and cleaning policy

- Dataset: **SMS Spam Collection**, UCI Machine Learning Repository.
- Dataset page: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Original archive: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
- Citation: Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset].
  UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84
- License: **Creative Commons Attribution 4.0 (CC BY 4.0)**, as stated on UCI.
  https://creativecommons.org/licenses/by/4.0/
- Downloaded for this project: 15 October 2026.

The original dataset combines messages from several earlier collections, including
Grumbletext and the NUS SMS Corpus. See UCI and the archive's original README for
collection details. Attribution above applies to the dataset; the project does not
claim authorship of those messages.

`python -m spam_classifier.download` saves these original/local files:

- `raw/SMSSpamCollection`: UTF-8, each line is `label<TAB>message`.
- `raw/uci_readme.txt`: the archive's original README.
- `raw/source.json`: source URLs, citation, license, download time, and SHA-256 hashes.

Checksums observed in this run:

```text
Archive SHA-256:
1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3

SMSSpamCollection SHA-256:
7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d
```

Raw downloads are ignored by Git and are reproducible via the downloader. The
training report also records the hash of the exact dataset actually used.

## Policy, applied before train/test splitting

1. Count missing values and class balance in the input.
2. Strip surrounding whitespace from labels and messages, preserving internal
   text, punctuation, and case. Treat null/blank labels or messages as invalid
   rows and remove them. Record the number removed.
3. Reject unknown non-empty labels; only `ham` and `spam` are accepted.
4. Identify messages assigned both labels. Write their rows to
   `reports/conflicts.csv`; exclude **every row** belonging to those messages.
   Do not guess the correct label or decide by frequency.
5. For unambiguous messages, retain the first occurrence of each message.
6. Report post-cleaning class balance and then split.

This policy detects duplicates under the stated whitespace rule. It does not
merge paraphrases, case variants, or different numbers in otherwise similar spam.
The actual download had 414 repeated rows, no missing/blank rows, and no label
conflicts. The generated audit is authoritative for any later download.

