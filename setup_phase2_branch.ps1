# ---------------------------------------------------------------------------
# Phase 2 — create the local dev branch and stage the Phase 1 changes.
#
# Workflow: develop in the scratch folder, then run this to bring the stable
# changes into your real git clone on a "phase2" branch. NOTHING is pushed,
# so your public repo and the live Streamlit app (which deploy from main) are
# untouched until you choose to push/merge.
#
# Run in PowerShell:  powershell -ExecutionPolicy Bypass -File .\setup_phase2_branch.ps1
# Adjust the two paths below if your folders differ.
# ---------------------------------------------------------------------------

$repo    = "C:\Users\borjs\Projects\PythonPortfolio"   # the git clone (deploys to Streamlit Cloud via main)
$scratch = "C:\PortfolioApp_Phase2"                    # this Phase 2 working copy

if (-not (Test-Path "$repo\.git")) {
    Write-Host "ERROR: $repo is not a git repository. Edit `$repo at the top of this script." -ForegroundColor Red
    exit 1
}

Set-Location $repo

# 1) Start from an up-to-date main
git checkout main
git pull --ff-only

# 2) Create (or switch to) the phase2 dev branch
git rev-parse --verify phase2 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { git checkout phase2 } else { git checkout -b phase2 }

# 3) Bring over ONLY the Phase 1 changes from the scratch copy
Copy-Item "$scratch\app.py"           "$repo\app.py"           -Force
Copy-Item "$scratch\core\scenario.py" "$repo\core\scenario.py" -Force

# 4) Stage and show exactly what changed — REVIEW THIS before committing
git add app.py core/scenario.py
Write-Host "`n=== git status ===" -ForegroundColor Cyan
git status
Write-Host "`n=== staged diff (summary) ===" -ForegroundColor Cyan
git --no-pager diff --staged --stat

Write-Host "`nReview the diff above. If it looks right, commit:" -ForegroundColor Yellow
Write-Host '  git commit -m "Phase 1: per-security min/max weight bounds in scalable CVaR engine"'
Write-Host "`nNothing has been pushed. When a phase is demo-ready you can back it up / open a PR:" -ForegroundColor Yellow
Write-Host "  git push -u origin phase2"
Write-Host "Or when fully stable, merge to main (this triggers the Streamlit Cloud deploy):" -ForegroundColor Yellow
Write-Host "  git checkout main; git merge phase2; git push"
