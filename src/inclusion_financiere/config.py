# src/inclusion_financiere/config.py
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "Data" / "raw" / "inclusion-financiere-en-afrique.csv"
TARGET = "bank_account"
