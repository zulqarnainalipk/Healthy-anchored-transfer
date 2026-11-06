# Module 00: dataset audit

`data_audit_code/audit_dataset.py` reads a dataset folder (read only), walks the tree, reads image headers, hashes every file and writes CSV tables, two figures and a summary report. Nothing from the dataset is copied to the output folder.

**Inputs:** the folder with the source and target datasets. **Outputs:** `inventory.csv` and other tables, figures and a report (see `results/code_audit_result/` and `results/local_audit/`; `inventory.csv` has the same content in both, rows in a different order). Item-by-item results: `docs/audit_checklist.md`.

How to re-run:

```bash
python data_audit_code/audit_dataset.py --input path/to/plant-paper-all-sources --output path/to/out
```

The script logs each step with a time stamp.
