# `Show cards` glob — paste-ready

The dated card moved from `localdata/<lane>/cards/auto_tickets_<day>.txt` to
`localdata/<lane>/auto_tickets_<day>.txt`, and `settle_lane` now also publishes
`localdata/<lane>/auto_tickets_performance.txt`. The `Show cards` step still
globs the old directory, so it matches nothing and the run log stops printing the
cards.

The whole file with the change already applied is **`docs/workflows/bullion_daily.yml`** —
paste that over the live file and you are done; the snippet below is just the
diff, for review.

I cannot apply it directly: pushing `.github/workflows/` is refused for this GitHub App
(`refusing to allow a GitHub App to create or update workflow
.github/workflows/bullion_daily.yml without workflows permission`). Paste the
block below over the current step.

## Replace

```yaml
      - name: Show cards
        run: |
          shopt -s nullglob
          for card in localdata/*/cards/*.txt; do
            echo "=== ${card} ==="
            cat "${card}"
          done
```

## With

```yaml
      - name: Show cards
        run: |
          shopt -s nullglob
          for card in localdata/*/auto_tickets_????-??-??.txt localdata/*/auto_tickets_performance.txt; do
            echo "=== ${card} ==="
            cat "${card}"
          done
```

The `????-??-??` pattern keeps `auto_tickets_performance.txt` from matching the
dated-card glob, and `nullglob` still makes an empty tree print nothing instead
of a literal path.
