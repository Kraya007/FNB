# Video Walkthrough Script

**Total target: 5-7 minutes**  
**Setup:** Have both the PowerPoint and Streamlit app (localhost:8501) open. Screen record with OBS or PowerPoint recorder.

---

## PART 1: PowerPoint Intro (~1 min)

### Slide 1 — Title (10 sec)
> "Hi, I'm [your name]. For this DataQuest challenge, I built an interpretable credit model that captures 99.7% of the LightGBM performance ceiling — while keeping every single prediction fully explainable."

### Slide 2 — Our Approach (30 sec)
> "My approach followed four stages. First, I explored the data through a custom Streamlit app to find meaningful risk patterns. Second, I engineered 15 domain-informed features — each one traceable to an EDA discovery. Third, I trained a WoE logistic regression reaching AUC 0.8102. And fourth, I deployed a business dashboard for policy-level decisions."

> "Let me show you the app."

**[Switch to browser — localhost:8501]**

---

## PART 2: App Walkthrough (~4-5 min)

### Home Page (20 sec)
**Show:** Data quality report, missing values, target distribution chart

> "This is the home page. It gives a quick data quality overview — we can see the dataset has around 121,000 records, the target is default_flag at about 15% default rate, and there are some features with missing values like months_since_last_delinquency. Missing values were handled during the WoE binning process rather than being imputed."

---

### Research Page (30 sec)
**Click:** "Research" in the sidebar

**Show:** Scroll through the GLM vs Non-Linear plot, then scroll to the regulatory table

> "The Research page covers the theory. Here's a visual comparison of linear versus non-linear decision boundaries — logistic regression draws a straight line, while tree models can curve around the data. We chose the linear approach for full transparency."

> *[Scroll to bottom]*  
> "I also analysed which features a regulator might flag — age is high risk under the National Credit Act, and region can be a proxy for race. These are documented with their risk levels."

---

### Univariate Explorer (45 sec)
**Click:** "Univariate Explorer" in the sidebar

**Show:** Select `credit_utilisation_pct` from the dropdown. Wait for the binning table and chart to load.

> "This is where the real analysis happens. Let me select credit utilisation. The optbinning algorithm has found the optimal bins — and you can see in this chart how the WoE drops as utilisation increases. High utilisation means high risk. The IV of [read the value] confirms this is a strong predictor."

**Show:** Change dropdown to `dti_ratio`

> "If I switch to DTI ratio, we see a similar monotonic pattern — but notice how the risk accelerates sharply in the highest bins. This non-linear shape is exactly what WoE captures that raw numeric features miss."

---

### Bivariate Explorer (30 sec)
**Click:** "Bivariate Explorer" in the sidebar

**Show:** Select `loan_amount` as X and `annual_income` as Y. Point to the scatter plot.

> "The bivariate explorer shows interactions. With loan amount on X and annual income on Y, you can see the teal dots — good loans — cluster where income is high relative to the loan. The gold dots — defaults — are concentrated where borrowers took large loans relative to their income. This is exactly why I created the loan_to_income ratio feature."

---

### Model Comparison (30 sec)
**Click:** "Model Comparison" in the sidebar

**Show:** Wait for models to train. Point to the AUC bar chart, then scroll to ROC curves.

> "The model comparison page trains all four models side by side. You can see the baseline at 0.75, our primary model at 0.81, the sign-corrected version at essentially the same level, and LightGBM at the ceiling. The ROC curves show our teal line nearly overlapping the gold LightGBM line — that's the 99.7% capture rate."

---

### Business Dashboard (60 sec)
**Click:** "Business Dashboard" in the sidebar

**Show:** Adjust the PD threshold slider slowly from 0.10 to 0.30. Point to the metrics changing.

> "This is the business value dashboard. Watch what happens as I move the threshold from 10% to 30%. The approval rate climbs from about 50% to 85%, but the portfolio default rate also rises. You can see the exact trade-off in real time."

**Show:** Scroll to the "disproportionate risk" glass card

> "Below the chart, the app calculates where risk starts accelerating disproportionately. This is the elbow effect — the last 10 to 20 percent of approvals cause nearly half the defaults."

**Show:** Scroll to Precision vs Recall section. Point to the live metrics.

> "The precision and recall metrics update live. At this threshold, precision is [read value] and recall is [read value]. The key insight is that a single missed default costs 5 to 10 times more than a wrongly rejected good customer."

**Show:** Scroll to Scenario Analysis. Adjust LGD from 60 to 80.

> "The scenario analysis lets stakeholders model different economic conditions. If I increase the loss-given-default to 80% — simulating a recession — watch how the optimal threshold drops. The model automatically becomes more conservative to protect against rising losses."

---

### Scorecard & Explainer (45 sec)
**Click:** "Scorecard & Explainer" in the sidebar

**Show:** Select a feature in the Scorecard tab. Show the points chart.

> "This page converts the model into a production scorecard. Each feature bin adds or subtracts points from a base score of 600. Higher points mean lower risk."

**Show:** Click the "Applicant Explainability" tab. Select an applicant. Show the waterfall chart.

> "And this is the explainability view. For any individual applicant, the waterfall shows exactly which features pushed the prediction up or down. Teal bars are protective, gold bars are risk factors. This is what a bank needs for adverse action notices — telling a customer exactly why they were declined."

---

## PART 3: Back to PowerPoint Closing (~30 sec)

**[Switch back to PowerPoint — Slide 9]**

> "To summarise: we built a logistic regression that hits AUC 0.8102 — 99.7% of the non-linear ceiling — with a Gini of 0.62, no overfitting, and perfect rank ordering. Every prediction is fully explainable, every engineered feature traces back to a specific EDA finding, and the business dashboard supports real policy decisions under different economic scenarios."

> "Thank you."

---

## Recording Tips
- **Resolution:** 1920x1080 minimum
- **Font size:** Make sure the browser is zoomed to 100%
- **Speed:** Don't rush — let the charts load before talking about them
- **Cursor:** Use your mouse to point at things you're describing
- **Tone:** Confident but conversational — you're explaining your work, not reading a script
- **Length:** Aim for 5-7 minutes. Under 5 feels rushed, over 8 loses attention
