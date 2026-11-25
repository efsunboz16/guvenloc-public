import csv

from guvenloc.metrics import RUNS_FIELDS, log_run


def test_log_run_appends(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("seed: 0\n")
    out = tmp_path / "runs.csv"
    log_run("s.py", cfg, 0, "test", "success@25", 0.5, runs_csv=out)
    log_run("s.py", cfg, 0, "test", "success@25", 0.6, runs_csv=out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert list(rows[0].keys()) == RUNS_FIELDS
    assert len(rows) == 2 and rows[1]["value"] == "0.6"
