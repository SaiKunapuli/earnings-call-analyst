# Convenience wrapper — runs the ECA pipeline with the project's venv Python
# so you don't have to type the full interpreter path.
#
# Usage (from the project root):
#   .\run.ps1                # full pipeline 01 -> 03 -> 04 -> 05
#   .\run.ps1 --dry-run      # preview the plan, run nothing
#   .\run.ps1 --from 03      # skip the slow price pull
#   .\run.ps1 --only 04 05   # just re-model + backtest
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\run_pipeline.py" @args
