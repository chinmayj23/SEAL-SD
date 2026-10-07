# SEAL-SD

**SEAL** (interpretable trajectory **S**egmentation for longitudinal data) groups
entities by the shape of their target trajectory and describes each group with a
compact rule over the entity descriptors. A single randomized search fits the
grouping and its rule together, scoring each candidate by

```
cost = rho + lambda * complexity
```

where `rho` is the within-group trajectory distance relative to the panel average
and `complexity` counts the propositions in the rules. The grouping and its
description are therefore chosen jointly, so every reported group arrives with a
short rule that identifies its members.

## Install

```
pip install -r requirements.txt
```

Requires numpy, pandas, scipy, scikit-learn and matplotlib.

## Data

The panels are built from public datasets. Place the preprocessed panel files under
`data/` (or point `SEAL_DATA` at their directory):

```
data/worldbank/worldbank_preprocessed.csv
data/co2/co2_preprocessed.csv        data/co2/owid_co2_data.csv
data/covid/covid_preprocessed.csv
data/epa/epa_preprocessed.csv
```

Each file is a long table with columns `entity`, `time`, the target column, and the
descriptor columns. The target and its transformations are excluded from the
descriptors in every panel.

## Usage

```
python run_panels.py        # fit SEAL on all four panels, write results/ and figures/
python baselines.py         # SEAL vs Cluster-then-Explain vs EMM on the shared objective
```

`run_panels.py worldbank` runs a single panel. A fixed random seed makes every run
reproducible.

## Layout

```
seal/            the method (pyramid distance, search, rule extraction, diversity ranking)
run_panels.py    panel descriptors, fitting, and figure rendering
baselines.py     the two baselines and the comparison table
results/         per-panel memberships and rules
figures/         the rule-card figure per panel
```

The `results/` and `figures/` committed here are the ones reported in the paper.
