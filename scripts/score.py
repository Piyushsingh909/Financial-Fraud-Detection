from pathlib import Path
import argparse
import polars as pl
import joblib


def main() -> None:
	parser = argparse.ArgumentParser(description="Score new data with trained model")
	parser.add_argument("--input", required=True, help="Path to parquet/csv file to score")
	parser.add_argument("--output", required=True, help="Path to write scored parquet")
	args = parser.parse_args()

	root = Path(__file__).resolve().parents[1]
	model_path = root / "models" / "xgb_baseline.joblib"
	if not model_path.exists():
		raise SystemExit(f"Model not found at {model_path}. Train the model first.")

	model = joblib.load(model_path)
	input_path = Path(args.input)

	if input_path.suffix.lower() == ".parquet":
		df = pl.read_parquet(input_path)
	else:
		df = pl.read_csv(input_path, infer_schema_length=10000)

	# Basic feature alignment as in training
	X = df.drop([c for c in ["isFraud", "nameOrig", "nameDest"] if c in df.columns])
	if "type" in X.columns:
		X = X.to_dummies(columns=["type"], drop_first=True)

	X_pd = X.to_pandas()
	proba = model.predict_proba(X_pd)[:, 1]
	out = df.with_columns(pl.Series("fraud_proba", proba))
	Path(args.output).parent.mkdir(parents=True, exist_ok=True)
	out.write_parquet(args.output)
	print(f"Wrote scores -> {args.output}")


if __name__ == "__main__":
	main()


