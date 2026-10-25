"""Build fake raw files with a known effect, to check the pipeline recovers it.

High-exposure ZIPs lose 3% of value, phased in over 6 months after 2025-08
(the catastrophe-model approval). Nothing else differs between groups.
"""
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)

# 30 x 20 grid of 5 km ZIPs in California Albers, across 6 counties
cells, rows = [], []
for i in range(30):
    for j in range(20):
        z = f"9{i:02d}{j:02d}"
        cells.append({"ZCTA5CE20": z, "geometry": box(i * 5000, j * 5000, (i + 1) * 5000, (j + 1) * 5000)})
        rows.append({"zip": z, "county": f"County{i // 5}"})
zcta = gpd.GeoDataFrame(cells, crs="EPSG:3310").to_crs("EPSG:4269")
shp = out / "shp"
shp.mkdir(exist_ok=True)
zcta.to_file(shp / "cb_2020_us_zcta520_500k.shp")
with zipfile.ZipFile(out / "cb_2020_us_zcta520_500k.zip", "w") as zf:
    for f in shp.iterdir():
        zf.write(f, f.name)

# Hazard zones: a Very High band and a High band in the north (j >= 12)
fhsz = gpd.GeoDataFrame(
    [{"HAZ_CLASS": "Very High", "geometry": box(0, 75000, 150000, 100000)},
     {"HAZ_CLASS": "High", "geometry": box(0, 60000, 150000, 75000)},
     {"HAZ_CLASS": "Moderate", "geometry": box(0, 50000, 150000, 60000)}],
    crs="EPSG:3310").to_crs("EPSG:4326")
fhsz.iloc[:2].to_file(out / "fhsz_sra.geojson", driver="GeoJSON")
fhsz.iloc[2:].to_file(out / "fhsz_lra.geojson", driver="GeoJSON")

# Zillow-style wide file
months = pd.period_range("2022-01", "2026-09", freq="M")
event = pd.Period("2025-08", "M")
recs = []
for r in rows:
    j = int(r["zip"][3:])
    high = j >= 15  # fully inside Very High
    base = rng.lognormal(13.3, 0.3)
    county_trend = 0.002 * (int(r["county"][-1]) - 2)
    vals = {}
    level = np.log(base)
    for m in months:
        level += 0.003 + county_trend + rng.normal(0, 0.002)
        k = (m - event).n
        hit = -0.03 * min(max(k + 1, 0), 6) / 6 if high else 0.0
        vals[m.strftime("%Y-%m-") + str(m.days_in_month)] = round(float(np.exp(level + hit)), 0)
    recs.append({"RegionID": r["zip"], "SizeRank": 0, "RegionName": r["zip"], "RegionType": "zip",
                 "StateName": "CA", "State": "CA", "City": "X", "Metro": "M",
                 "CountyName": r["county"], **vals})
pd.DataFrame(recs).to_csv(out / "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", index=False)
print("synthetic raw files written to", out)
