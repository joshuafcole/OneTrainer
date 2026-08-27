"""Tests for scripts/analyze_perf.py.

The analyzer is the only part of the OT_PERF slice with real arithmetic in it, and
the arithmetic is the part a reader trusts blindly: an exponent printed next to
"quadratic self-attention" will be believed. So the fit is checked against series
whose exponent is known exactly, and the report is driven through ``main()`` on a
synthetic log rather than by calling the helpers it happens to use.

The one judgement call the report makes on the user's behalf -- dropping the first
row of each resolution bucket, which carries torch.compile warmup -- is checked
end to end, because a warmup row left in the median is a silently wrong number
rather than a visibly missing one.

Run with ``python -m pytest tests/test_analyze_perf.py``.
"""

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from analyze_perf import load, main, median, powerlaw_slope, region_columns  # noqa: E402


def _write(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(path)


def _step(tokens, **kwargs):
    row = {"kind": "step", "latent_tokens": tokens}
    row.update(kwargs)
    return row


class TestMedian:
    def test_odd_length_takes_the_middle_of_the_sorted_values(self):
        assert median([5.0, 1.0, 3.0]) == 3.0

    def test_even_length_averages_the_two_middle_values(self):
        assert median([4.0, 1.0, 3.0, 2.0]) == 2.5

    def test_empty_is_nan_rather_than_an_exception(self):
        # A region that was never timed must not take the whole report down.
        assert math.isnan(median([]))


class TestPowerlawSlope:
    @pytest.mark.parametrize("exponent", [1.0, 1.5, 2.0])
    def test_recovers_the_exponent_of_a_clean_power_law(self, exponent):
        xs = [1024, 2048, 4096, 16384]
        ys = [3.7 * x ** exponent for x in xs]
        slope, intercept = powerlaw_slope(xs, ys)
        assert slope == pytest.approx(exponent, abs=1e-9)
        assert math.exp(intercept) == pytest.approx(3.7, rel=1e-9)

    def test_fewer_than_two_usable_points_is_nan(self):
        slope, intercept = powerlaw_slope([1024], [40.0])
        assert math.isnan(slope) and math.isnan(intercept)

    def test_non_positive_pairs_are_dropped_not_logged(self):
        # A zeroed region on one bucket must not poison the buckets that measured.
        xs = [1024, 2048, 4096]
        ys = [0.0, 2.0, 4.0]
        slope, _ = powerlaw_slope(xs, ys)
        assert slope == pytest.approx(1.0)

    def test_a_single_resolution_measured_many_times_is_nan_not_a_divide_by_zero(self):
        slope, _ = powerlaw_slope([4096, 4096, 4096], [10.0, 11.0, 12.0])
        assert math.isnan(slope)


class TestLoad:
    def test_splits_step_rows_from_cache_rows_and_skips_blank_lines(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text(
            json.dumps({"kind": "step", "step": 0}) + "\n"
            + "\n"
            + json.dumps({"kind": "cache", "label": "latents"}) + "\n"
            + "   \n"
            + json.dumps({"step": 1}) + "\n"  # no kind at all -> a step row
        )
        steps, caches = load(str(path))
        assert [s.get("step") for s in steps] == [0, 1]
        assert [c["label"] for c in caches] == ["latents"]


class TestRegionColumns:
    def test_discovers_regions_from_the_log_preferring_the_familiar_order(self):
        steps = [{"zzz_ms": 1.0, "backward_ms": 2.0, "predict_ms": 3.0, "aaa_ms": 4.0}]
        assert region_columns(steps) == ["predict_ms", "backward_ms", "aaa_ms", "zzz_ms"]

    def test_excludes_the_probes_own_derived_timings(self):
        # data_wait/step_total are not regions and get fixed places in the table.
        steps = [{"predict_ms": 1.0, "data_wait_ms": 2.0, "step_total_ms": 3.0}]
        assert region_columns(steps) == ["predict_ms"]

    def test_a_region_present_in_only_some_rows_still_gets_a_column(self):
        steps = [{"predict_ms": 1.0}, {"predict_ms": 1.0, "prior_predict_ms": 2.0}]
        assert region_columns(steps) == ["predict_ms", "prior_predict_ms"]


class TestReport:
    """Driven through main(), the production entry point, on a synthetic log."""

    def _log(self, tmp_path):
        rows = []
        for tokens, steady in ((1024, 40.0), (4096, 160.0), (16384, 640.0)):
            # The i==0 row is 10x the steady state: compile/warmup. If it reached the
            # median the printed number would be visibly wrong.
            rows.extend(_step(
                tokens,
                step=i,
                predict_ms=steady * (10 if i == 0 else 1),
                backward_ms=steady * 2,
                optimizer_ms=5.0,
                data_wait_ms=1.0,
                step_total_ms=steady * 3 + 6,
                vram_peak_reserved_gb=8.0,
                offload_xfers=40,
                recompiles=2 if i == 0 else 0,
            ) for i in range(4))
        rows.append({"kind": "cache", "label": "latents", "group": 0, "variation": 1,
                     "items": 100, "wall_s": 50.0, "items_per_s": 2.0, "mp_per_s_encode": 2.4})
        return _write(tmp_path / "ot_perf.jsonl", rows)

    def _row(self, out, tokens):
        for line in out.splitlines():
            if line.split()[:1] == [str(tokens)]:
                return line.split()
        raise AssertionError(f"no table row for {tokens} tokens in:\n{out}")

    def test_the_warmup_row_of_each_bucket_is_excluded_from_the_medians(self, tmp_path, capsys):
        main([self._log(tmp_path)])
        out = capsys.readouterr().out
        for tokens, steady in ((1024, 40.0), (4096, 160.0), (16384, 640.0)):
            cells = self._row(out, tokens)
            assert cells[1] == "3", "the first row of the bucket should be dropped"
            assert float(cells[2]) == pytest.approx(steady), "median must be the steady state, not warmup"

    def test_the_fit_reports_the_exponent_and_doubles_it_for_the_resolution_per_side(self, tmp_path, capsys):
        main([self._log(tmp_path)])
        out = capsys.readouterr().out
        # predict scales exactly linearly in tokens in this log.
        assert "predict              ~ tokens^1.00" in out
        assert "optimizer            ~ tokens^0.00" in out
        # step_total is dominated by the linear terms; tokens ~ resolution^2.
        [line] = [ln for ln in out.splitlines() if ln.strip().startswith("step_total")]
        exponent = float(line.split("tokens^")[1].split()[0])
        per_side = float(line.split("resolution^")[1].split()[0])
        # both halves are printed rounded to 2dp, so allow a rounding step on each
        assert per_side == pytest.approx(2 * exponent, abs=0.02)

    def test_recompiles_are_summed_across_every_row_including_the_dropped_warmup(self, tmp_path, capsys):
        # 2 per bucket x 3 buckets. The warmup row is dropped from the medians but its
        # recompiles are real and still count against the run.
        main([self._log(tmp_path)])
        assert "total dynamo recompiles across run: 6" in capsys.readouterr().out

    def test_cache_rows_are_totalled_per_label(self, tmp_path, capsys):
        main([self._log(tmp_path)])
        out = capsys.readouterr().out
        assert "grp0 var1:  100 items in    50.0s" in out
        assert "TOTAL latents: 100 encodes, 50s wall" in out

    def test_a_log_with_no_latent_tokens_says_so_instead_of_printing_an_empty_table(self, tmp_path, capsys):
        path = _write(tmp_path / "p.jsonl", [{"kind": "step", "step": 0, "predict_ms": 1.0}])
        main([path])
        assert "nothing to bucket by resolution" in capsys.readouterr().out

    def test_a_single_resolution_prints_the_table_but_no_fit(self, tmp_path, capsys):
        rows = [_step(4096, step=i, predict_ms=40.0, step_total_ms=50.0) for i in range(3)]
        main([_write(tmp_path / "p.jsonl", rows)])
        out = capsys.readouterr().out
        assert "4096" in out
        assert "power-law fit" not in out, "one point is not a trend"
