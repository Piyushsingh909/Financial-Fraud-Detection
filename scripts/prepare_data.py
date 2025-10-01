import sys
from pathlib import Path

import polars as pl


def main() -> None:
	root_dir = Path(__file__).resolve().parents[1]
	raw_csv = root_dir / "data" / "raw" / "Fraud.csv"
	processed_dir = root_dir / "data" / "processed"
	processed_dir.mkdir(parents=True, exist_ok=True)
	out_parquet = processed_dir / "fraud.parquet"

	if not raw_csv.exists():
		print(f"Expected CSV at: {raw_csv}. Move 'Fraud.csv' into data/raw/ and retry.")
		sys.exit(1)

	# Read with a longer schema inference for robust dtypes
	df = pl.read_csv(raw_csv, infer_schema_length=10000)

	# Standardize column names
	df = df.rename({name: name.strip() for name in df.columns})

	# Optional: sanity checks for known columns
	required_cols = {
		"step",
		"type",
		"amount",
		"nameOrig",
		"oldbalanceOrg",
		"newbalanceOrig",
		"nameDest",
		"oldbalanceDest",
		"newbalanceDest",
		"isFraud",
		"isFlaggedFraud",
	}
	missing = [c for c in required_cols if c not in df.columns]
	if missing:
		print(f"Warning: missing expected columns: {missing}")

	# Persist to Parquet for fast re-use
	df.write_parquet(out_parquet, compression="zstd")
	print(f"Wrote {out_parquet} with {df.shape[0]} rows and {df.shape[1]} columns")


if __name__ == "__main__":
	main()


