# DataQuest 2026: Interpretable Credit Models - Modelling Summary

## 1. Research & Methodology
For this project, we were constrained to using a strictly interpretable Logistic Regression model. While non-linear models like LightGBM capture complex feature interactions inherently, Logistic Regression requires explicit feature engineering to model non-linear relationships. 

To bridge this gap, we relied on **Weight of Evidence (WoE)** and **Information Value (IV)**. WoE transforms raw continuous or categorical features into a monotonic, linear relationship with the log-odds of the default rate. This satisfies the linearity assumption of Logistic Regression while capturing non-linear behavior in the original data. `optbinning` was used to algorithmically find the optimal bins that maximize the IV while maintaining monotonicity.

## 2. Key EDA Findings
Using the interactive Univariate and Bivariate Explorer in the Streamlit app, we discovered the following:
- **Age**: Younger applicants show significantly higher default rates. However, due to fair lending regulations, utilizing age directly poses a severe compliance risk. 
- **Credit Utilization & Revolving Balance**: Extremely high credit utilization corresponds to a steep increase in defaults.
- **Home Ownership**: Renters display a higher risk profile compared to applicants holding a mortgage or owning a home outright.
- **Interest Rate**: Higher interest rates are strongly correlated with higher default flags, likely because the interest rate was previously assigned based on a legacy risk model.
- **DTI Ratio**: Applicants with high debt-to-income ratios default at significantly higher rates, but the relationship is non-linear — risk accelerates sharply above ~40%.
- **Loan Amount & Income**: The interaction between these two features is more predictive than either alone — a large loan is only risky if the borrower cannot afford it.

*Note: The Univariate Explorer tab in our application explicitly extracts the `optbinning` tables to justify these observations mathematically, satisfying the requirement to demonstrate business sense rather than relying purely on automated algorithms.*

## 2.5 Data Cleaning Approach

The dataset is intentionally messy, and our cleaning strategy was guided by two principles: **preserve information** (don't discard rows) and **avoid division-by-zero artefacts** in engineered ratios.

- **Zero-value guards in denominators**: Several engineered ratios divide by fields like `annual_income`, `employment_length_years`, or `num_open_accounts`, which can legitimately be zero. Rather than dropping these rows (which would discard valid applicants), we applied domain-appropriate floor substitutions — e.g., replacing zero income with 1 and zero employment length with 0.5 years — so the resulting ratios remain finite and interpretable without inflating to infinity.
- **Missing value imputation**: `months_since_last_delinquency` is null for applicants with no delinquency history. This missingness is *informative* — it signals a clean record. We imputed these with 999 (a large sentinel) so that `months_since_last_delinq_scaled` correctly assigns these applicants to the lowest-risk bin. For the baseline model, remaining numeric nulls were imputed with training-set medians.
- **No row or column drops**: We did not drop any observations. Categorical features with missing values (if any) are handled natively by `optbinning`, which assigns missing values to a dedicated bin during the WoE transformation. This preserves all available data for training and avoids the selection bias introduced by listwise deletion.

## 3. From EDA to Features — Traceability

A good model requires clear linkage between what was *discovered* and what was *engineered*. The table below traces each EDA finding to the feature it inspired and the rationale behind it:

| EDA Discovery | Engineered Feature | Rationale |
|---|---|---|
| Loan amount alone is weakly predictive, but large loans relative to income default far more often | `loan_to_income` | Captures affordability — a 50k loan is safe for a 200k earner but risky for a 40k earner |
| Credit utilisation is a top predictor, but raw revolving balance is not — the *ratio* matters | `revolving_to_income`, `credit_limit`, `available_credit` | Separates borrowers who carry high balances because they have high limits (low risk) from those who are maxed out (high risk) |
| DTI ratio is predictive, but doesn't account for the new loan being applied for | `monthly_payment`, `payment_to_income`, `new_dti` | The existing DTI ignores the new loan obligation. `new_dti` answers: "Can this borrower afford their current debt *plus* this new loan?" |
| Delinquency count is predictive, but doesn't account for borrower age | `delinq_to_age`, `accounts_per_year_age` | A 25-year-old with 2 delinquencies is riskier than a 55-year-old with the same count. Normalising by age captures behavioural maturity |
| Income and employment length are independently predictive | `income_per_age`, `loan_to_emp_length` | High income at a young age may signal instability; a large loan relative to short employment suggests over-leveraging |
| DTI and credit utilisation interact — high values in both are far worse than high in either alone | `utilization_dti_ratio` | Captures the compounding effect of being stretched on both existing debt payments and revolving credit |

### 3.1 What improvement did these adjustments induce?

| Stage | AUC | Change |
|-------|-----|--------|
| Baseline (raw numerics, no engineering) | 0.7525 | — |
| + WoE transformation (optbinning only) | ~0.79 | +0.04 |
| + Domain-engineered ratios (table above) | **0.8102** | +0.02 |
| **Total improvement** | | **+0.0577** |

The WoE transformation provided the largest single lift by capturing non-linear bin effects. The domain-engineered ratios added a further ~0.02 by encoding credit risk knowledge that cannot be discovered through binning alone (e.g., post-loan affordability, interaction effects).

## 4. Model Performance & Evaluation

### 4.1 Primary Model — Maximum AUC

The baseline model utilizing minimal preprocessing yielded an AUC of **0.7525**. By employing optimal binning with domain-informed feature engineering, the primary model achieved an AUC of **0.8102**, capturing approximately **99.7%** of the LightGBM non-linear ceiling (0.8127).

| Metric | Value | Rating |
|--------|-------|--------|
| AUC (test) | **0.8102** | Excellent |
| AUC (train) | 0.8157 | — |
| Train-Test gap | 0.0055 | No overfitting |
| Gini coefficient | **0.6205** | Excellent (>0.50) |
| KS Statistic | **0.4759** | Excellent (>0.40) |
| Brier Score | 0.1035 | Well-calibrated |
| Log Loss | 0.3397 | Good |

### 4.2 Rank Ordering (by probability decile)

The model achieves **perfect rank ordering** — default rates increase monotonically across all deciles:

| Decile | Probability Range | Actual Default Rate |
|--------|------------------|-------------------|
| 1 (Safest) | 0.0025 – 0.0184 | 1.6% |
| 2 | 0.0184 – 0.0304 | 2.7% |
| 3 | 0.0304 – 0.0455 | 3.8% |
| 4 | 0.0455 – 0.0653 | 5.7% |
| 5 | 0.0653 – 0.0902 | 7.2% |
| 6 | 0.0902 – 0.1248 | 10.8% |
| 7 | 0.1248 – 0.1747 | 14.7% |
| 8 | 0.1747 – 0.2504 | 22.0% |
| 9 | 0.2504 – 0.3858 | 30.3% |
| 10 (Riskiest) | 0.3858 – 0.9715 | 55.0% |

### 4.3 Calibration

Predicted probabilities closely match observed default rates across most deciles (predicted/actual ratios between 0.94 and 1.33). Calibration is strong in the lower-risk segments; the highest-risk decile exhibits moderate over-prediction (ratio of 1.33), which is common in logistic models at distribution tails where data density is lower. Overall, calibration is sufficient for policy use, though threshold decisions in the riskiest segment should account for this conservative bias.

## 5. Primary Model Equation

The final mathematical form of the linear predictor $\eta$ is:

$$
\begin{align*}
\eta &= -1.7556 \\
&\quad - 0.6012 \times \text{WoE}(\text{age}) \\
&\quad - 0.2432 \times \text{WoE}(\text{annual\_income}) \\
&\quad - 0.3559 \times \text{WoE}(\text{employment\_length\_years}) \\
&\quad - 1.2021 \times \text{WoE}(\text{num\_open\_accounts}) \\
&\quad - 0.3123 \times \text{WoE}(\text{num\_delinquencies\_2yr}) \\
&\quad - 0.0568 \times \text{WoE}(\text{total\_revolving\_balance}) \\
&\quad - 1.0509 \times \text{WoE}(\text{credit\_utilisation\_pct}) \\
&\quad - 0.2808 \times \text{WoE}(\text{months\_since\_oldest\_account}) \\
&\quad - 1.2272 \times \text{WoE}(\text{num\_hard\_inquiries\_6mo}) \\
&\quad - 0.4917 \times \text{WoE}(\text{loan\_amount}) \\
&\quad - 0.1178 \times \text{WoE}(\text{interest\_rate}) \\
&\quad - 0.8780 \times \text{WoE}(\text{dti\_ratio}) \\
&\quad - 0.2324 \times \text{WoE}(\text{months\_since\_last\_delinquency}) \\
&\quad - 0.2770 \times \text{WoE}(\text{pct\_accounts\_current}) \\
&\quad - 0.1283 \times \text{WoE}(\text{loan\_to\_income}) \\
&\quad + 0.1667 \times \text{WoE}(\text{credit\_limit}) \\
&\quad - 0.1458 \times \text{WoE}(\text{available\_credit}) \\
&\quad - 0.1308 \times \text{WoE}(\text{revolving\_to\_income}) \\
&\quad - 0.3987 \times \text{WoE}(\text{monthly\_payment}) \\
&\quad + 0.0160 \times \text{WoE}(\text{payment\_to\_income}) \\
&\quad - 0.0930 \times \text{WoE}(\text{accounts\_per\_year\_age}) \\
&\quad - 0.0523 \times \text{WoE}(\text{delinq\_to\_age}) \\
&\quad - 0.7326 \times \text{WoE}(\text{months\_since\_last\_delinq\_scaled}) \\
&\quad - 0.1831 \times \text{WoE}(\text{income\_per\_age}) \\
&\quad - 0.3762 \times \text{WoE}(\text{loan\_to\_emp\_length}) \\
&\quad - 0.7439 \times \text{WoE}(\text{total\_monthly\_debt}) \\
&\quad + 0.5256 \times \text{WoE}(\text{total\_monthly\_debt\_plus\_loan}) \\
&\quad - 0.1408 \times \text{WoE}(\text{new\_dti}) \\
&\quad - 0.0301 \times \text{WoE}(\text{income\_per\_account}) \\
&\quad - 0.0286 \times \text{WoE}(\text{revolving\_per\_account}) \\
&\quad - 0.1463 \times \text{WoE}(\text{utilization\_dti\_ratio})
\end{align*}
$$

**Note on positive coefficients:** Three features (`credit_limit`, `payment_to_income`, `total_monthly_debt_plus_loan`) exhibit positive coefficients due to multicollinearity with their parent features. This is an artefact of including both a ratio and its components — the model distributes the effect across correlated features. This does not violate the assessment rules and maximises predictive power.

**Note on age as a predictor:** As identified in our EDA (Section 2), age is flagged as a feature a regulator may disapprove of under fair lending principles. It is retained here for maximum predictive power in this analytical exercise. In a production deployment, age would require formal disparate impact testing, and if disparate impact were found, it should be substituted with a permitted behavioural proxy (e.g., `months_since_oldest_account`, which already captures credit history maturity without directly encoding a protected characteristic). The sign-corrected model in Section 6 demonstrates that near-identical performance is achievable under tighter constraints, and the same iterative removal approach could be applied to age if regulatory compliance demanded it.

## 6. Sign-Corrected Model — Scorecard Best Practice

As an additional demonstration of credit modelling expertise, we trained a **sign-corrected variant** using iterative coefficient correction — a standard practice in production scorecard development. Features with positive WoE coefficients were iteratively removed until all remaining coefficients were negative, ensuring full economic interpretability.

- **Sign-Corrected AUC: 0.8101** (28 features, all negative coefficients)
- Converged after 3 iterations, dropping: `credit_limit`, `total_monthly_debt_plus_loan`, `total_revolving_balance`
- AUC loss: only 0.0001 — negligible trade-off for full sign compliance

This demonstrates that the positive signs in the primary model are caused by redundancy, not model failure, and that correcting them has virtually no impact on performance.

## 7. Model Comparison Summary

| Model | AUC | Improvement | Features |
|-------|-----|-------------|----------|
| Baseline (raw numeric LR) | 0.7525 | — | 30 |
| **Primary Model (WoE + all features)** | **0.8102** | **+0.0577** | 31 |
| Sign-Corrected Model | 0.8101 | +0.0576 | 28 |
| LightGBM ceiling (reference only) | ~0.82 | — | — |

## 8. Business Recommendations
As visualized in the Business Value Dashboard:
- Setting a strict probability threshold increases the **precision** of our accepted portfolio but severely limits the **volume** (recall).
- A balanced threshold should be selected based on the actual dollar cost of a default versus the opportunity cost of denying a good loan. The dashboard allows stakeholders to dial in this threshold dynamically to visualize the portfolio default rate against total funded volume.
- At a threshold of **0.25**: approximately **80%** of applicants are approved, with **55.6%** of defaults caught and a portfolio default rate well below the base rate.
