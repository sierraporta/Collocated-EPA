# Hourly precision of collocated duplicate regulatory monitors (U.S. AQS, 2023–2026)

Code, input data and derived tables for the article

> Arrieta-Guardo, I. S., Gonzalez-Ortiz, J. F., Negrin-Perez, E. A. and Sierra-Porta, D.
> *Hourly precision of collocated duplicate regulatory monitors for PM₂.₅ and ozone in the United States, 2023–2026.*
> Manuscript submitted to the *Journal of the Air & Waste Management Association*.

Repository: <https://github.com/sierraporta/Collocated-EPA>

Some sites in the U.S. regulatory network run two continuous monitors of the same
method designation side by side. The study uses these duplicates to estimate the
**hourly** precision of the record: how much two identical instruments disagree,
how that disagreement is structured (additive or proportional, independent
between hours or persistent), and how far it carries into annual statistics. The
analysis covers PM₂.₅, ozone and PM₁₀ from January 2023 to May 2026.

## Quick start

```bash
python3 -m pip install -r requirements.txt
cd code
python3 test_build_pocs.py      # ingestion tests on synthetic AQS files, no network
./rerun_all.sh                  # full analysis from the archived inputs (a few minutes)
```

`rerun_all.sh` regenerates every table in `Results_Collocated/` and every figure
in `Figures_Collocated/` from the files in `Data_Collocated/`. Running it on a
fresh clone reproduces the deposited tables exactly. You can redirect the output
with the environment variables `O` (results) and `F` (figures), for example
`O=/tmp/res F=/tmp/fig ./rerun_all.sh`.

## Pipeline and correspondence with the article

| Step | Script | What it does | Used in |
|---|---|---|---|
| 1 | `01_build_pocs.py` | Downloads the national hourly AQS files and keeps **every** parameter occurrence code (POC) at sites with more than one monitor, along with its method code. Not part of `rerun_all.sh` (see below). | Data |
| 0 | `00_pairs_inventory.py` | Inventory of monitor pairs. Classifies them by method designation into precision, reprocessing and comparability pairs. Site composition by climate region and land use. | Data, "Composition of the precision sites" |
| 2 | `02_concurrency.py` | Concurrent hours per pair. Replicate diagnostic (correlation, hours present in only one series), with the reprocessed T640 pairs as positive control. | Methods 3.1; Results "Pair validity" |
| 3 | `03_agreement.py` | Appendix A coefficient of variation (Eqs. 1–2), hourly and daily, by instrument family. Sensitivity to estimator and concentration floor. | Table 1; Methods 3.2 |
| 4 | `04_resolution_limit.py` | Paired difference in units of the reporting increment. Sheppard correction. | Table 3; Figure 2 |
| 5 | `05_derived.py` | Annual mean, 98th percentile, daily maximum hour, exceedance counts. Ozone MDA8 per 40 CFR 50 Appendix U. | Table 5; Figure 3 |
| 7 | `07_error_structure.py` | Allan deviation, two-component model σ(c) = √(a² + (bc)²), offset/slow/fast decomposition, σ_ref and R²_max, site bootstrap. | Tables 2 and 4; Figure 1 |
| 8 | `08_crossmethod_sarajevo.py` | T640 vs BAM comparison across methods in the network, and the same estimator applied to the Sarajevo data of Hagler et al. (2022). | Discussion, "Consistency with independent estimates" |
| 6 | `06_make_figures.py` | Figures 1–3 and supplementary Figure S1. | Figures |

The step numbers keep the order in which the scripts were written. `rerun_all.sh`
runs them in dependency order.

### Re-running the ingestion

`01_build_pocs.py` downloads 16 national hourly files from
<https://aqs.epa.gov/aqsweb/airdata/download_files.html> (several GB, hours of
transfer). You only need it to extend the period:

```bash
python3 01_build_pocs.py --years 2023 2024 2025 2026
python3 01_build_pocs.py --reindex      # rebuild the column index without network
```

It imports the download, streaming and unit-conversion routines from
`aqs_io.py`.

## Repository layout

```
code/                    analysis scripts (see table above) and tests
Data_Collocated/
  hourly_pocs/           hourly series, one parquet per pollutant and year;
                         columns are site#POC, values in ug/m3 or ppm
  poc_columns.parquet    index of those columns: site, POC, method code, GMT offset
  poc_methods.csv        method code per site, POC and year (from AQS daily summaries)
  method_designations.csv  AQS method code -> equivalent method designation and
                         instrument family (archived: AQS refreshes the code
                         table nightly, and the pair classification depends on it)
  hagler_sarajevo_daily.csv  daily T640/BAM data of Hagler et al. (2022)
  inputs/
    monitors_subset.csv  AQS monitor listing (monitor type, operating agency)
    stations.parquet     AQS site attributes (climate region, land use)
    poc_selection.parquet, stations_hourly.parquet
                         site and POC metadata from the hourly ingestion
Results_Collocated/      derived tables (parquet), one per analysis step
Figures_Collocated/      figures as they appear in the article
```

### Main result tables

| File | Content |
|---|---|
| `pairs_inventory.parquet` | All monitor pairs with ≥2000 h each, category, monitor types, operating agency |
| `concurrency.parquet` | Analysed pairs (≥500 concurrent hours): concurrency, correlation, category |
| `agreement_precision.parquet` | Hourly and daily Appendix A CV per precision pair-year |
| `agreement_precision_sensitivity.parquet` | PM₂.₅ hourly CV with floors of 0, 2 and 5 µg/m³, plain CV, MAD-based CV and median \|d\| (Methods 3.2) |
| `agreement_comparability[_alltypes].parquet` | Same for pairs of different designation (SLAMS only / all monitor types) |
| `resolution_limit.parquet` | Paired difference in reporting increments |
| `error_allan.parquet`, `error_bins.parquet`, `error_summary.parquet` | Error structure per pair-year |
| `error_family.parquet`, `error_model_fit.parquet` | Family medians and model fits with 95% site-bootstrap intervals |
| `derived.parquet` | Annual statistics for both monitors of each pair-year (2023–2025) |
| `crossmethod.csv` | T640–BAM network and Sarajevo comparison |

## Definitions used throughout

- **Pair-year:** two POCs of the same parameter at the same site in the same
  calendar year.
- **Precision pair:** identical method code, i.e. two units of the same instrument
  with the same data processing. Only these pairs enter the precision estimates.
- **Reprocessing pair:** the same T640/T640X instrument, once as originally
  reported and once as recalculated by EPA in 2024 with the Network Data
  Alignment (method codes 236/238 vs 736/738). Excluded from all agreement
  estimates and used as a positive control.
- **SLAMS only:** every estimate uses pairs in which both monitors are State and
  Local Air Monitoring Stations.
- The 2026 files end in May 2026. Annual statistics require ≥200 valid days and
  therefore cover 2023–2025.

## Instrument families

Results are reported by instrument family, assigned from the method code in
`Data_Collocated/method_designations.csv`. The families with precision pairs
are `T640 optical` (Teledyne T640/T640X), `Beta attenuation` (Met One
BAM-1020/1022), `SHARP` (Thermo 5030), `TEOM FDMS` and `TEOM/R&P` for
particulate matter, and `UV photometric` for ozone.

## Data sources and licence

All monitoring data come from the public files of the U.S. Environmental
Protection Agency Air Quality System (<https://aqs.epa.gov/aqsweb/airdata/>), a
work of the U.S. Government. The Sarajevo data are from Hagler et al. (2022),
doi:[10.23719/1522400](https://doi.org/10.23719/1522400). The code is released
under the MIT licence (see `LICENSE`).

## Requirements

Python ≥ 3.10 with numpy, pandas, pyarrow, scipy and matplotlib (see
`requirements.txt`). The published results were produced with Python 3.13.3,
numpy 2.1.3, pandas 2.2.3, scipy 1.18.1, matplotlib 3.10.0 and pyarrow 20.0.0.
