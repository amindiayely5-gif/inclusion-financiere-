import pandas as pd
from .config import RAW, TARGET

CAT_COLS = [
	"country",
	"location_type",
	"cellphone_access",
	"gender_of_respondent",
	"relationship_with_head",
	"marital_status",
	"education_level",
	"job_type",
]

NUM_COLS = ["household_size", "age_of_respondent"]


def load_raw(path=RAW) -> pd.DataFrame:
	return pd.read_csv(path)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
	"""Sépare X et y et retire les colonnes inutiles."""

	y = (df[TARGET] == "Yes").astype(int)
	X = df.drop(columns=[TARGET, "uniqueid", "year"])

	return X, y
