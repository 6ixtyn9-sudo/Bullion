# `bullion_daily.yml`: make a failed localdata push fail the run

**Status: not applied.** `.github/workflows/` is yours to edit, so this file is the
copy-paste text for the last residual from Sweep 2 (P2-N2). Nothing under
`.github/workflows/` in this repository was touched.

## What is wrong today

The final step of the `daily` job, **Commit localdata artifacts (never the SQLite
warehouse)**, retries the push three times and then simply stops:

```yaml
          git commit -m "localdata: daily paper run $(date -u +%Y-%m-%dT%H:%MZ)"
          for attempt in 1 2 3; do
            git pull --rebase origin main
            if git push; then
              break
            fi
            echo "push rejected; rebasing and retrying (attempt ${attempt})"
            sleep 10
          done
```

The loop is right; the exit is not. If all three pushes are rejected the step
still ends with the status of `sleep 10`, so the job is **green**, the tick looks
healthy, and the receipts, performance history and coverage files that were
committed locally are thrown away with the runner.

That matters more than it used to. Since PR #8 the warehouse heals itself from
the receipts on every run (`rehydrate_forecasts` in `scripts/daily.py`), so the
failure no longer corrupts calibration — but it does mean the durable copy of a
day's evidence never leaves the runner, and nothing tells you. Calibration
history that cannot be rebuilt is exactly the failure mode rehydration exists to
absorb; it should still be loud.

## The change

Four lines: track the outcome, and exit non-zero when no attempt succeeded.

```yaml
          git commit -m "localdata: daily paper run $(date -u +%Y-%m-%dT%H:%MZ)"
          pushed=0
          for attempt in 1 2 3; do
            git pull --rebase origin main
            if git push; then
              pushed=1
              break
            fi
            echo "push rejected; rebasing and retrying (attempt ${attempt})"
            sleep 10
          done
          if [ "${pushed}" -ne 1 ]; then
            echo "::error::localdata push failed after 3 attempts"
            exit 1
          fi
```

Notes on the shape:

* `git push` sits in an `if` condition, so the runner's `bash -e` does not abort
  the step mid-loop — the retries still happen.
* The `::error::` annotation surfaces on the run summary, not only in the log.
* `git pull --rebase` failing is already fatal under `-e`; that is the correct
  behaviour and needs no extra handling.
* The step keeps its earlier `exit 0` when there is nothing to commit, so a quiet
  day stays green.

## Verifying it after pasting

```bash
# the marker is in the step, and the file still parses
python3 - <<'PY'
import yaml
job = yaml.safe_load(open(".github/workflows/bullion_daily.yml"))["jobs"]["daily"]
step = next(s for s in job["steps"] if s["name"].startswith("Commit localdata"))
assert "push failed after 3 attempts" in step["run"]
print("exit check present; step body still parses")
PY
```

Both were run against this repository's current `bullion_daily.yml` with the
block substituted: `yaml.safe_load` succeeds (10 steps, unchanged) and
`bash -n` accepts the step body. A `workflow_dispatch` run then confirms it
end-to-end — the step log shows `push rejected; ... (attempt 3)` followed by the
error annotation and a red job, instead of a green one.

## What this is not

This is ops hygiene, not a data-integrity fix. It changes nothing about the
counted lane, the gate, settlement or calibration; it changes only what a failed
publish does to the run's colour.
