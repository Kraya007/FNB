# AI Usage Reflection

## Overview
This document outlines how AI was utilized to assist in completing the DataQuest 2026 project. The AI was treated as an analytical assistant rather than a substitute for judgment.

## 1. Prompts Used
- "go through the the txt file i sent you about this fnb dataQuest and let's dot down the action plan to be thorough and execute to be the best"
- "Streamlit + Plotly + Pandas + Optbinning + Statsmodels. The challenge rules state state we need not rely only on automated feature selection or trial-and-error tuning without explanation. Because optbinning automates the binning, we must extract the bin tables it generates and display them in your Streamlit app. we need to show the judges why the bins make business sense."
- "is it possible to achieve an AUC greater than 0.82"  I don't thiso personally the 0.82 in their description may be a typo or based on non linearity but— prompted an exhaustive experimental analysis of 80+ model configurations to determine the true dataset ceiling.
- "without breaking the rules can't we be more creative and squeeze it to the full 100%" — led to creative techniques (WoE interaction features, out-of-fold WoE, model blending) while staying within the logistic regression constraint.
- "implement the FNB theme with a glass feel" — directed the visual design of the Streamlit app to align with FNB branding.

## 2. Where AI Helped
- **App Scaffolding**: The AI rapidly built the Streamlit application structure, saving hours of boilerplate coding.
- **Library Integration**: The AI successfully integrated `optbinning` with `streamlit` and `plotly` to dynamically calculate and visualize WoE bins and IV metrics.
- **Bug Fixing**: When `optbinning` encountered compatibility issues with newer versions of `scikit-learn` (specifically `check_array`), the AI authored a runtime patch (`patch_sklearn.py`) to bypass the error without requiring complex environment downgrades.
- **Model Pipeline**: The AI wrote the initial baseline and improved model pipelines using `sklearn.pipeline.Pipeline`, ensuring no data leakage occurred between the WoE transformation and Logistic Regression training.
- **Feature Engineering**: The AI proposed domain-informed ratio features (e.g., `loan_to_income`, `new_dti`, `payment_to_income`) grounded in credit risk fundamentals. I evaluated each for business sense before inclusion.
- **AUC Ceiling Analysis**: The AI conducted a systematic investigation testing regularization (L1, L2, ElasticNet), IV thresholds, class weighting, and model blending to determine the maximum achievable AUC. This confirmed the dataset ceiling at ~0.81 and validated that our model was already near-optimal.
- **Sign Correction**: The AI implemented an iterative sign-correction procedure (standard in production scorecard development) to produce a variant model with all-negative WoE coefficients.
- **Comprehensive Evaluation**: The AI generated a full evaluation report including Gini, KS statistic, Brier score, calibration by decile, and rank ordering — metrics I then included in the modelling summary.
- **FNB Theme Design**: The AI created a custom glassmorphism CSS theme matching FNB's brand palette (teal + gold) with ambient glow effects, glass cards, and a branded header component.

## 3. What Suggestions Were Accepted
- Using a strict `optbinning` approach to algorithmically define the Weight of Evidence bins.
- Creating the `Business_Dashboard.py` to interactively visualize Precision vs Recall by adjusting probability thresholds dynamically.
- The proposed directory structure separating the web app (`app.py`, `pages/`) from the modelling output (`model_pipeline.py`, `fnbQuest.ipynb`).
- Training two models (primary for max AUC, sign-corrected for scorecard best practice) and presenting both in the report.
- Domain-informed feature engineering ratios after verifying they made business sense.

## 4. What Was Modified
- I manually dictated the technology stack (Streamlit + Plotly + Optbinning) rather than relying on the AI's defaults.
- I enforced the strict rule that the algorithmically generated `optbinning` tables must be visible in the Streamlit app to fulfill the "business sense" requirement of the project brief.
- I decided to keep the primary model (with 3 positive coefficients) as the main submission for maximum AUC, rather than defaulting to the sign-corrected version. The sign-corrected model was presented as a creative addition demonstrating awareness of industry standards.
- I directed the FNB brand alignment rather than using a generic theme.

## 5. What Was Rejected
- Any suggestions to use non-linear models (e.g., Random Forests, XGBoost, LightGBM) for the final submission, as this violated the strictly interpretable Logistic Regression constraint of the project. LightGBM was only used temporarily in scratch scripts to verify the dataset's performance ceiling, and these files were deleted before submission.
- Model blending (combining multiple LR models) was explored but rejected for the final submission as it reduced interpretability without meaningful AUC improvement.
