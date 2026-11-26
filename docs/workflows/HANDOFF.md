# Workflow handoff (Arena cannot push `.github/workflows/`)

Arena's GitHub App token cannot create or modify workflow files. The two YAML
files in this folder are complete and copy-paste ready.

## Step 1 — add and trigger the probe (required first)

1. In GitHub, open `docs/workflows/probe_espn.yml`.
2. Create `.github/workflows/probe_espn.yml` and paste the file's contents
   verbatim (no edits needed).
3. Commit on `main`.
4. **Actions → probe-espn → Run workflow** (leave `date` empty, `leagues: all`).
5. When it finishes, open `localdata/_probes/espn_probe_latest.json` and check
   the coverage lines printed in the job log:

   ```
   nhl              events=NNN pre=NNN offers=NNN moneyline=0.xx moneyline_draw=0.xx spread=0.xx total=0.xx
   ```

6. Report back with that file (or just the log block). Parsers will be corrected
   against real payloads before anything is enabled for tickets.

What the probe answers, per lane: event counts; named-book offer counts; which
books appear; whether a draw price path exists and under which key; whether
puckline/spread and total lines are present; predictor availability and its
`statistics` key names; summary field presence (`pickcenter`, `predictor`,
`injuries`, boxscore); and whether the core season event index resolves for
historical backfill.

Until that report is committed and inspected, every lane is `research_only`:
`scripts/daily.py` forecasts and grades but freezes **no** ticket, and writes
`research_only: espn_probe_evidence_missing` into
`localdata/_coverage/<lane>_<day>.json`.

## Step 2 — enable the daily lane (only after Step 1 is reviewed)

1. Create `.github/workflows/bullion_daily.yml` from
   `docs/workflows/bullion_daily.yml`.
2. Commit on `main`.
3. Trigger once manually (**Run workflow**, `league: all`, empty date/as_of) and
   confirm:
   * the run exits 0;
   * `localdata/<lane>/auto_tickets_YYYY-MM-DD_decision.json` appears where a
     qualifying ticket existed, or `…_no_bet.json` where none did;
   * `localdata/<lane>/auto_tickets_performance.json` exists;
   * `python3 scripts/replay_harness.py --parity` printed
     `"parity": true`.
4. Leave the cron schedule to take over.

Cron (UTC): `09:45` (11:45 SAST, before the soccer cutoff), `14:15` (16:15 SAST,
after the US cutoffs), `06:15` (08:15 SAST, grading).

## Notes

* The SQLite warehouse is cached, never committed (`.gitignore` covers
  `localdata/bullion.db`). Cached runs append evidence; a cold run does a bounded
  backfill of finals only.
* Receipts and no-bet locks are write-once. Re-running a day cannot rotate a
  frozen selection or price.
* Any lane below 50% pre-kickoff named-book coverage for a market stays out of
  tickets for that market automatically (see `lane_permission` in
  `src/bullion/espn.py`).
