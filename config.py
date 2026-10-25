"""Settings for the California wildfire insurance-retreat study.

Put the downloaded raw files in data/raw/ under the names below (or pass
other paths on the command line). Every path is relative to the project root.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"

FILES = {
    # Zillow Home Value Index, ZIP level, all homes, smoothed & seasonally adjusted
    "zhvi": RAW / "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    # CAL FIRE Fire Hazard Severity Zones (SRA + LRA). Shapefile, GeoJSON or
    # GeoPackage all work; more than one file can be listed.
    "fhsz": [RAW / "fhsz_sra.geojson", RAW / "fhsz_lra.geojson"],
    # Census 2020 ZIP Code Tabulation Areas (cartographic boundary file)
    "zcta": RAW / "cb_2020_us_zcta520_500k.zip",
    # Optional: CA Dept. of Insurance FAIR Plan residential policies by ZIP
    "fair_plan": RAW / "fair_plan_by_zip.xlsx",
    # Optional, from WRDS: MSRB transactions + an issuer-to-ZIP/county map
    "msrb": RAW / "msrb_ca_trades.csv",
    "issuer_map": RAW / "issuer_geography.csv",
}

# Dated information shocks to California wildfire insurance pricing.
# Each is a candidate event for the difference-in-differences.
EVENTS = {
    "la_fires": "2025-01",            # Palisades/Eaton fires: salience shock
    "cat_models_approved": "2025-08",  # CDI allows forward-looking wildfire models
    "fair_plan_filing": "2025-10",     # FAIR Plan files 35.8% increase
    "fair_plan_approved": "2026-05",   # CDI approves 29.1%
    "fair_plan_effective": "2026-10",  # New rates apply from Oct 15, 2026
}

# Exposure score weights by hazard class
HAZARD_WEIGHTS = {"Very High": 1.0, "High": 0.5, "Moderate": 0.15}

# Event-study window, in months relative to the event
LEADS, LAGS = 12, 12
# Minimum post-event months before an event is estimated
MIN_POST_MONTHS = 3
# Equal-area projection for California (meters)
CA_CRS = "EPSG:3310"
