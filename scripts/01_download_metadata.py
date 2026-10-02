from pathlib import Path
import requests
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

RECORD_ID = "3887306"
ZIP_PATH = RAW / "bdg2_v1.zip"

print("=" * 80)
print("BDG2 DOWNLOAD + SELECTIVE EXTRACTION")
print("=" * 80)

# ------------------------------------------------------------
# 1. Get actual Zenodo download URL
# ------------------------------------------------------------

api_url = f"https://zenodo.org/api/records/{RECORD_ID}"

print("\nQuerying Zenodo...")

r = requests.get(api_url, timeout=60)
r.raise_for_status()

record = r.json()
files = record["files"]

if len(files) != 1:
    raise RuntimeError(
        f"Expected one release ZIP, found {len(files)} files."
    )

file_info = files[0]

filename = file_info["key"]
size_mb = file_info["size"] / 1024**2

download_url = (
    file_info.get("links", {}).get("self")
    or file_info.get("links", {}).get("download")
)

print(f"Release: {filename}")
print(f"Size:    {size_mb:.2f} MB")

# ------------------------------------------------------------
# 2. Download ZIP once
# ------------------------------------------------------------

if ZIP_PATH.exists():
    existing_mb = ZIP_PATH.stat().st_size / 1024**2

    print(f"\nZIP already exists: {ZIP_PATH}")
    print(f"Current size: {existing_mb:.2f} MB")

    # Basic completeness check
    expected_size = file_info["size"]

    if ZIP_PATH.stat().st_size != expected_size:
        print("\nExisting ZIP appears incomplete.")
        print("Deleting and downloading again...")
        ZIP_PATH.unlink()
    else:
        print("Size matches Zenodo record. Skipping download.")

if not ZIP_PATH.exists():

    print("\nDownloading BDG2 release...")
    print("This is approximately 568 MB and is needed only once.\n")

    with requests.get(
        download_url,
        stream=True,
        timeout=(30, 600)
    ) as response:

        response.raise_for_status()

        total = int(
            response.headers.get(
                "content-length",
                file_info["size"]
            )
        )

        downloaded = 0

        with open(ZIP_PATH, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=4 * 1024 * 1024
            ):
                if not chunk:
                    continue

                f.write(chunk)
                downloaded += len(chunk)

                pct = downloaded / total * 100

                print(
                    f"\rDownloaded: "
                    f"{downloaded / 1024**2:7.1f} MB "
                    f"({pct:5.1f}%)",
                    end=""
                )

    print("\n\nDownload complete.")

# ------------------------------------------------------------
# 3. Inspect ZIP
# ------------------------------------------------------------

print("\nInspecting archive...")

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    names = z.namelist()

    print(f"Files inside archive: {len(names):,}")

    # Find metadata.csv robustly
    candidates = [
        name for name in names
        if name.lower().endswith("/metadata/metadata.csv")
        or name.lower().endswith("metadata.csv")
    ]

    print("\nMetadata candidates:")

    for c in candidates:
        print("  ", c)

    if not candidates:
        print("\nCould not locate metadata.csv.")
        print("\nFirst 100 archive entries:")

        for name in names[:100]:
            print(name)

        raise SystemExit(1)

    # Prefer exact expected structure
    metadata_member = None

    for c in candidates:
        if "/data/metadata/metadata.csv" in c.replace("\\", "/"):
            metadata_member = c
            break

    if metadata_member is None:
        metadata_member = candidates[0]

    print("\nSelected:")
    print(metadata_member)

    # Read directly from ZIP without extracting whole archive
    with z.open(metadata_member) as f:
        df = pd.read_csv(f)

# ------------------------------------------------------------
# 4. Save standalone metadata
# ------------------------------------------------------------

OUTPUT = RAW / "bdg2_metadata.csv"

df.to_csv(OUTPUT, index=False)

print("\nMetadata saved:")
print(OUTPUT)

# ------------------------------------------------------------
# 5. Inspect metadata
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("METADATA SUMMARY")
print("=" * 80)

print("\nShape:")
print(df.shape)

print("\nColumns:")

for i, col in enumerate(df.columns, 1):
    print(f"{i:02d}. {col}")

print("\nPrimary space usage:")

possible_cols = [
    "primaryspaceusage",
    "primary_space_usage",
    "primaryuse",
    "primary_use"
]

use_col = next(
    (c for c in possible_cols if c in df.columns),
    None
)

if use_col:

    counts = df[use_col].value_counts(
        dropna=False
    )

    print(counts.to_string())

else:
    print("No primary-use column automatically detected.")

print("\n" + "=" * 80)
print("ELECTRICITY AVAILABILITY")
print("=" * 80)

electricity_cols = [
    c for c in df.columns
    if "electric" in c.lower()
]

if electricity_cols:

    for c in electricity_cols:
        print(f"\n{c}:")
        print(df[c].value_counts(dropna=False).head(20))

else:
    print("No electricity-related metadata column found.")

print("\n" + "=" * 80)
print("SITE DISTRIBUTION")
print("=" * 80)

site_candidates = [
    "site_id",
    "site",
    "siteid"
]

site_col = next(
    (c for c in site_candidates if c in df.columns),
    None
)

if site_col:

    print(
        df[site_col]
        .value_counts()
        .head(30)
        .to_string()
    )

else:
    print("No site column automatically detected.")

print("\n" + "=" * 80)
print("SUCCESS")
print("=" * 80)

print(f"Rows:      {len(df):,}")
print(f"Columns:   {len(df.columns):,}")
print(f"Saved:     {OUTPUT}")
print("\nThe full ZIP has NOT been extracted.")