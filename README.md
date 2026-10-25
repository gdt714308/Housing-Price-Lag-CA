# California wildfire insurance-retreat study

Tests whether home values (and later, muni spreads) in high wildfire-hazard
California ZIPs fell relative to unexposed ZIPs in the same county after each
2025–26 insurance-pricing shock. If they did, prices had not been reflecting
the risk before the shock.

## Raw files to download (put them in `data/raw/`)

| File name expected | Source | Notes |
|---|---|---|
| `Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv` | Zillow Research → Data → Home Values → ZHVI, Geography: ZIP Code | "ZHVI All Homes (SFR, Condo/Co-op) Time Series, Smoothed, Seasonally Adjusted" |
| `fhsz_sra.geojson`, `fhsz_lra.geojson` | CAL FIRE / Office of the State Fire Marshal, Fire Hazard Severity Zones GIS data (California State Geoportal) | State (SRA) and local (LRA) layers. Shapefile, GeoJSON or GeoPackage all work; rename or edit `config.py` |
| `cb_2020_us_zcta520_500k.zip` | Census Bureau cartographic boundary files, 2020, ZIP Code Tabulation Areas, 1:500k | Leave it zipped |
| `fair_plan_by_zip.xlsx` (optional) | CA Dept. of Insurance residential market data by ZIP | Adds an insurance-based exposure measure |
| `msrb_ca_trades.csv` (later) | WRDS → MSRB transactions, CA issuers | See `muni.py` for columns |
| `issuer_geography.csv` (later) | Built by hand: `cusip6, zip` per issuer | Needed to link bonds to exposure |

## Run

```
pip install geopandas pyogrio statsmodels linearmodels openpyxl matplotlib
python run.py
```

Outputs land in `output/`: `zip_exposure.csv`, `did_results.csv`,
`did_continuous_results.csv`, and an event-study chart plus a raw-paths chart
for each event that has at least 3 post-event months of data.

## Design

- **Exposure:** share of each ZIP's land area in Very High (weight 1.0), High
  (0.5) and Moderate (0.15) hazard zones. Groups: `none` (zero exposure),
  `low` and `high` (exposed ZIPs split at the median).
- **Model:** log ZHVI on ZIP fixed effects and county-by-month fixed effects,
  so high- and no-exposure ZIPs are compared only within the same county and
  month. Errors clustered by ZIP. See `model.py`.
- **Events** (`config.py`): LA fires (Jan 2025), catastrophe models approved
  (Aug 2025), FAIR Plan filing (Oct 2025), approval (May 2026), new rates in
  effect (Oct 2026).

## Reading the results

- Check the **pre-trend p-value** on each chart first. Below about 0.10 means
  the groups were already diverging, so that event's estimate is not clean.
- **The events overlap.** They are 3–7 months apart, so a 12-month window
  around one catches the next. Treat the earliest event with flat pre-trends as
  the main estimate and the later ones as cumulative updates.
- **Zillow values are smoothed**, so a shock shows up over several months
  rather than at once.
- The muni benchmark is a same-day, same-maturity median of all CA trades.
  Because treated issuers are part of that median, effects are slightly
  understated (a planted 10 bp effect came back as 9.2 bp).

## Validation

`tests/make_synthetic.py` builds fake data where high-exposure ZIPs lose 3%
after Aug 2025. The pipeline recovers it (−2.3% average post-event, −3.0% at
full effect, pre-trend p = 0.78). Overlapping later events show failing
pre-trend tests, as they should.
