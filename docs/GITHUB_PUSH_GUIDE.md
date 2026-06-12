# GitHub Push Guide

Use these steps after you review the files locally.

## 1. Create a GitHub Repository

1. Go to GitHub.
2. Click **New repository**.
3. Name it something like `credit-risk-fraud-detection-platform`.
4. Keep it public or private based on your preference.
5. Do not initialize with a README, because this project already has one.

## 2. Check Local Files

From this project folder:

```powershell
git status
```

You should see new project files waiting to be staged.

## 3. Stage and Commit

```powershell
git add .
git commit -m "Initial credit risk and fraud detection platform"
```

## 4. Connect the Local Repo to GitHub

Replace `YOUR_USERNAME` and repository name:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/credit-risk-fraud-detection-platform.git
```

If a remote already exists, update it:

```powershell
git remote set-url origin https://github.com/YOUR_USERNAME/credit-risk-fraud-detection-platform.git
```

## 5. Push

```powershell
git branch -M main
git push -u origin main
```

## 6. Verify on GitHub

Open the repository page and confirm:

- README renders correctly.
- `docs/` contains business requirements and architecture.
- `src/` contains the project package.
- No `.env`, generated datasets, or local artifacts were uploaded.

## 7. Optional Demo Commands

After cloning the repo on another machine:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m credit_risk_platform.data.make_dataset --rows 5000 --out data/processed/risk_events.csv
python -m credit_risk_platform.pipelines.train_models --data data/processed/risk_events.csv --artifacts artifacts
python -m credit_risk_platform.rag.policy_assistant --question "What evidence is required for model validation?"
```
