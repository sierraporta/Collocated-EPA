#!/usr/bin/env bash
# =============================================================================
# rerun_all.sh — Full analysis chain of the collocated-monitor study
# =============================================================================
#
# Default paths are relative to this script, so a fresh clone runs without
# editing anything:
#
#   cd code && ./rerun_all.sh
#
# **Ingestion is not part of the chain.** `01_build_pocs.py` downloads sixteen
# national hourly EPA files, takes hours and only needs repeating when years
# are added. aqs.epa.gov is also often blocked behind proxies, so it is run
# separately from a normal terminal:
#
#   python3 test_build_pocs.py                        # 14 checks, no network
#   python3 01_build_pocs.py --years 2023 2024 2025 2026
#   python3 01_build_pocs.py --reindex                # if a run was interrupted
#
# Also outside the chain: the per-POC, per-year method codes
# (`poc_methods.csv`), distilled from the AQS daily summaries, and the
# designation table (`method_designations.csv`), taken from the AQS method
# code list. Both are archived because AQS refreshes its tables nightly and the
# pair classification depends on them.
#
# Environment variables: O (results), F (figures).
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

O=${O:-../Results_Collocated}
F=${F:-../Figures_Collocated}
mkdir -p "$O" "$F" "$O/logs"
step () { printf '\n=== %s === %s\n' "$*" "$(date +%H:%M:%S)"; }

step "0. pair inventory, the three categories and representativeness"
python3 00_pairs_inventory.py --out-dir "$O" 2>&1 | tee "$O/logs/00_pairs.log"

step "2. concurrent hours and replicate diagnostic"
# The T640 reprocessing pairs serve as a positive control: they are known to
# be the same instrument twice, so their signature shows the degenerate case.
python3 02_concurrency.py --out "$O/concurrency.parquet" \
  2>&1 | tee "$O/logs/02_concurrency.log"

step "3. hourly against daily agreement, precision category"
python3 03_agreement.py --arm precision --concurrency "$O/concurrency.parquet" --out-dir "$O" \
  2>&1 | tee "$O/logs/03_agreement.log"

step "3b. comparability category (different instruments), for the Discussion"
python3 03_agreement.py --arm comparability --concurrency "$O/concurrency.parquet" --out-dir "$O" \
  2>&1 | tee "$O/logs/03_comparability.log"
python3 03_agreement.py --arm comparability --all-types \
  --concurrency "$O/concurrency.parquet" --out-dir "$O" \
  2>&1 | tee "$O/logs/03_comparability_alltypes.log"

step "4. resolution limit against instrument limit"
python3 04_resolution_limit.py --concurrency "$O/concurrency.parquet" \
  --out "$O/resolution_limit.parquet" \
  2>&1 | tee "$O/logs/04_resolution.log"

step "5. propagation to annual mean, 98th percentile and exceedances"
python3 05_derived.py --concurrency "$O/concurrency.parquet" \
  --out "$O/derived.parquet" \
  2>&1 | tee "$O/logs/05_derived.log"

step "7. error structure: Allan deviation, sigma(c), decomposition, sensor bound"
python3 07_error_structure.py --agreement "$O/agreement_precision.parquet" \
  --resolution "$O/resolution_limit.parquet" --out-dir "$O" \
  2>&1 | tee "$O/logs/07_error_structure.log"

step "8. cross-method check: network T640-BAM and Sarajevo"
python3 08_crossmethod_sarajevo.py --res-dir "$O" \
  2>&1 | tee "$O/logs/08_crossmethod.log"

step "6. figures"
python3 06_make_figures.py --res-dir "$O" --fig-dir "$F" \
  2>&1 | tee "$O/logs/06_figures.log"

step "DONE"
