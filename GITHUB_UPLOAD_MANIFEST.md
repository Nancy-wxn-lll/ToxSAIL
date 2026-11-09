# GitHub Upload Manifest

This package is the clean experiment-code release for ToxSAIL.

## Included

- `README.md`
- `.gitignore`
- `requirements.txt`
- `configs/toxsail.json`
- `hybrid_benchmark.py`
- `src/`
- `scripts/submission_supplementary_experiments.py`
- `scripts/summarize_feature_ablation.py`
- `scripts/summarize_submission_supplement.py`
- `tests/test_gps_no_leakage.py`
- `data/README.md`
- `data/*.tab`

## Excluded

- Generated figures and draw.io layouts: `figures/`, `drawio/`
- Generated result tables: `tables/`, `results/`
- Old exploratory attempts: `archive/`
- Local neural baseline and plotting workspaces: `baselines/`, `plotting/`
- Virtual environments and caches: `.venv*/`, `__pycache__/`, `.pytest_cache/`
- Manuscript drafts and local Word files: `*.docx`

The excluded files are not required to rerun the core experiments. Running the scripts will recreate result folders locally.
