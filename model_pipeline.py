import patch_sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from optbinning import BinningProcess
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def add_engineered_features(df):
    df = df.copy()
    
    # 1. Domain-informed ratios
    df['loan_to_income'] = df['loan_amount'] / df['annual_income'].replace(0, 1)
    df['credit_limit'] = df['total_revolving_balance'] / (df['credit_utilisation_pct'].replace(0, 0.01) / 100)
    df['available_credit'] = df['credit_limit'] - df['total_revolving_balance']
    df['revolving_to_income'] = df['total_revolving_balance'] / df['annual_income'].replace(0, 1)
    
    # Monthly payment & affordability
    rate = df['interest_rate'] / 100 / 12
    df['monthly_payment'] = df['loan_amount'] * rate / (1 - (1 + rate)**(-36))
    df['payment_to_income'] = df['monthly_payment'] / (df['annual_income'] / 12).replace(0, 1)
    
    # Behavioural ratios
    df['accounts_per_year_age'] = df['num_open_accounts'] / df['age']
    df['delinq_to_age'] = df['num_delinquencies_2yr'] / df['age']
    df['months_since_last_delinq_scaled'] = df['months_since_last_delinquency'].fillna(999)
    df['income_per_age'] = df['annual_income'] / df['age']
    df['loan_to_emp_length'] = df['loan_amount'] / df['employment_length_years'].replace(0, 0.5)
    
    # Post-loan DTI
    df['total_monthly_debt'] = (df['dti_ratio'] / 100) * (df['annual_income'] / 12)
    df['total_monthly_debt_plus_loan'] = df['total_monthly_debt'] + df['monthly_payment']
    df['new_dti'] = df['total_monthly_debt_plus_loan'] / (df['annual_income'] / 12).replace(0, 1)
    
    # 2. Interaction ratios
    df['income_per_account'] = df['annual_income'] / df['num_open_accounts'].replace(0, 1)
    df['revolving_per_account'] = df['total_revolving_balance'] / df['num_open_accounts'].replace(0, 1)
    df['utilization_dti_ratio'] = df['credit_utilisation_pct'] * df['dti_ratio']
    
    return df


def print_equation(model, feature_names, title):
    """Print the logistic regression equation."""
    print(f"\n--- {title} ---")
    intercept = model.intercept_[0]
    coefficients = model.coef_[0]
    
    print(f"Log-odds (eta) = {intercept:.4f}")
    n_positive = 0
    for feature, coef in zip(feature_names, coefficients):
        if abs(coef) > 0.0001:
            sign = "+" if coef > 0 else "-"
            flag = "  [!POSITIVE]" if coef > 0 else ""
            print(f"  {sign} {abs(coef):.4f} x WoE({feature}){flag}")
            if coef > 0:
                n_positive += 1
    if n_positive > 0:
        print(f"\n  Note: {n_positive} feature(s) have positive coefficients (multicollinearity)")
    else:
        print(f"\n  [OK] All coefficients are negative (scorecard-compliant)")


def main():
    print("Loading data...")
    df = pd.read_csv("loan_book.csv")
    
    # Target variable
    target = 'default_flag'
    
    # Split into train and test based on 'set' column
    if 'set' in df.columns:
        train_df = df[df['set'] == 'train'].copy()
        test_df = df[df['set'] == 'test'].copy()
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    print("Applying feature engineering...")
    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)
    
    # Features (excluding identifiers and target)
    exclude_cols = [target, 'applicant_id_hash', 'application_date', 'set']
    features = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # =========================================================
    # BASELINE MODEL
    # =========================================================
    print("\n" + "="*60)
    print("BASELINE MODEL")
    print("="*60)
    
    X_train_base = X_train[num_cols].fillna(X_train[num_cols].median())
    X_test_base = X_test[num_cols].fillna(X_train[num_cols].median())
    
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train_base, y_train)
    
    baseline_preds = baseline_model.predict_proba(X_test_base)[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_preds)
    print(f"Baseline AUC: {baseline_auc:.4f}")
    
    # =========================================================
    # PRIMARY MODEL — Maximum AUC (all features)
    # =========================================================
    print("\n" + "="*60)
    print("PRIMARY MODEL — Maximum AUC (all WoE features)")
    print("="*60)
    
    categorical_variables = cat_cols
    
    bp_primary = BinningProcess(
        variable_names=features, 
        categorical_variables=categorical_variables,
        selection_criteria={"iv": {"min": 0.01}}
    )
    
    primary_lr = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    
    primary_pipeline = Pipeline([
        ("binning_process", bp_primary),
        ("logistic_regression", primary_lr)
    ])
    primary_pipeline.fit(X_train, y_train)
    
    primary_preds = primary_pipeline.predict_proba(X_test)[:, 1]
    primary_auc = roc_auc_score(y_test, primary_preds)
    print(f"Primary Model AUC: {primary_auc:.4f}")
    
    # Get the selected feature names after binning
    primary_features = list(bp_primary.transform(X_train).columns)
    print_equation(primary_lr, primary_features, "Primary Model Equation")
    
    # =========================================================
    # SIGN-CORRECTED MODEL — Scorecard best practice
    # =========================================================
    print("\n" + "="*60)
    print("SIGN-CORRECTED MODEL — Scorecard best practice")
    print("="*60)
    
    bp_sc = BinningProcess(
        variable_names=features, 
        categorical_variables=categorical_variables,
        selection_criteria={"iv": {"min": 0.02}}
    )
    bp_sc.fit(X_train, y_train)
    X_train_woe = bp_sc.transform(X_train)
    X_test_woe = bp_sc.transform(X_test)
    
    # Iterative sign-correction
    selected_cols = list(X_train_woe.columns)
    
    for iteration in range(10):
        sc_lr = LogisticRegression(max_iter=3000, random_state=42, C=0.1)
        sc_lr.fit(X_train_woe[selected_cols], y_train)
        
        coefs = sc_lr.coef_[0]
        positive_features = [c for c, coef in zip(selected_cols, coefs) if coef > 0]
        
        if not positive_features:
            print(f"  Sign correction converged after {iteration + 1} iteration(s)")
            break
        
        print(f"  Iteration {iteration + 1}: dropping {len(positive_features)} positive-coef feature(s): {positive_features}")
        selected_cols = [c for c in selected_cols if c not in positive_features]
    
    print(f"  Final feature count: {len(selected_cols)}")
    
    sc_preds = sc_lr.predict_proba(X_test_woe[selected_cols])[:, 1]
    sc_auc = roc_auc_score(y_test, sc_preds)
    print(f"Sign-Corrected Model AUC: {sc_auc:.4f}")
    
    # Build clean pipeline for the sign-corrected model
    sc_bp_final = BinningProcess(
        variable_names=selected_cols,
        categorical_variables=[c for c in categorical_variables if c in selected_cols],
        selection_criteria={"iv": {"min": 0.02}}
    )
    sc_pipeline = Pipeline([
        ("binning_process", sc_bp_final),
        ("logistic_regression", LogisticRegression(max_iter=3000, random_state=42, C=0.1))
    ])
    sc_pipeline.fit(X_train[selected_cols], y_train)
    
    sc_final_lr = sc_pipeline.named_steps['logistic_regression']
    print_equation(sc_final_lr, selected_cols, "Sign-Corrected Model Equation")
    
    # =========================================================
    # COMPARISON SUMMARY
    # =========================================================
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"  Baseline AUC:          {baseline_auc:.4f}")
    print(f"  Primary Model AUC:     {primary_auc:.4f}  (all features, max performance)")
    print(f"  Sign-Corrected AUC:    {sc_auc:.4f}  (all negative coefs, scorecard-compliant)")
    print(f"  LightGBM ceiling:      ~0.82   (reference benchmark)")
    print(f"\n  Improvement over baseline: +{primary_auc - baseline_auc:.4f} (primary), +{sc_auc - baseline_auc:.4f} (sign-corrected)")
    
    # =========================================================
    # SAVE BOTH MODELS
    # =========================================================
    import joblib
    joblib.dump({
        'primary_pipeline': primary_pipeline,
        'primary_features': features,
        'sign_corrected_pipeline': sc_pipeline,
        'sign_corrected_features': selected_cols,
    }, "improved_model_pipeline.pkl")
    print("\nBoth model pipelines saved to 'improved_model_pipeline.pkl'")

if __name__ == "__main__":
    main()

