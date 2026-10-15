"""Download the original UCI archive; keep its README and provenance."""

import hashlib
import io
import json
from datetime import datetime, timezone
from urllib.request import urlopen
from zipfile import ZipFile

from spam_classifier.paths import DATA_PATH

SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def main():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        print(f"Already downloaded: {DATA_PATH}. Using the existing copy.")
        return
    with urlopen(DOWNLOAD_URL, timeout=60) as response:
        archive_bytes = response.read()
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        # Read named files rather than extracting arbitrary archive paths.
        dataset_bytes = archive.read("SMSSpamCollection")
        readme_bytes = archive.read("readme")
    DATA_PATH.write_bytes(dataset_bytes)
    (DATA_PATH.parent / "uci_readme.txt").write_bytes(readme_bytes)
    provenance = {
        "source_page": SOURCE_PAGE,
        "download_url": DOWNLOAD_URL,
        "citation": "Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI. DOI: 10.24432/C5CC84.",
        "license": "CC BY 4.0 (as stated on the UCI dataset page)",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
    }
    (DATA_PATH.parent / "source.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Downloaded: {DATA_PATH}")


if __name__ == "__main__":
    main()

