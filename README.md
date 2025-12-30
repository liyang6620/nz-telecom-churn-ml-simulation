# NZ Telecom Churn Simulation Project

## 1. Executive Summary
This project simulates how a New Zealand telecom provider could analyse and reduce customer churn using publicly available fictional CRM data. 
Because New Zealand telco CRM is confidential, a public dataset (JB-Link, Kaggle) is used solely as a proxy.  
The purpose is not only technical exploration but to demonstrate how data can support business insight, decision-making, and customer-focused retention thinking.

---

## 2. Business Context — Why Churn Matters in New Zealand

New Zealand’s telecommunications market is highly concentrated. According to the Commerce Commission Telecommunications Monitoring Report (June 2025), three mobile network operators — 2degrees, Spark and One NZ — collectively hold approximately 97.5% of the mobile market, while 11 MVNOs operate only at a relatively small scale. The broadband market is somewhat less concentrated, with the top three providers holding around 73% market share.

Although infrastructure access is strong — fibre is available to 100% of urban homes and over 90% of households have access to 4G fixed wireless — customer switching behaviour is limited. Only around 11% of broadband households changed providers in the six months to June 2024, and nearly one-third of those who attempted to switch experienced difficulty or dissatisfaction. Bundling also plays a role in customer inertia, with 11% of non-switchers noting they remained because their broadband was part of a broader household bundle (e.g., electricity + internet).

### Why This Problem Matters

In a mature market where service availability and network performance no longer serve as differentiators, **growth is increasingly dependent on retaining existing users rather than solely acquiring new ones**. Reducing churn even marginally (for example, by 5–10%) can have a material financial impact, given that acquisition costs are typically several times higher than the cost to retain an existing customer.

For telecom operators in New Zealand, churn is therefore a business-critical metric influencing:
- revenue forecasting and financial planning
- customer-lifetime-value (CLV) analysis
- resource allocation for outbound retention efforts
- marketing spend and incentive design
- product and pricing strategy (e.g., bundling, plan structure, discount thresholds)

### What This Project Simulates and Aims to Solve

Because customer-level churn datasets in New Zealand are not publicly available, this project uses a fictional proxy dataset to simulate how churn analysis could operate inside a telecom organisation. The project structure mirrors the typical analytical flow used internally by telcos when churn becomes a strategic priority:

- identifying customers most at risk of leaving before churn occurs  
- analysing behavioural and demographic factors influencing the likelihood of churn  
- grouping customers into segments to support prioritisation  
- translating analytical findings into targeted business actions, such as:
  - proactive retention outreach
  - bundle design to reduce churn likelihood among specific segments
  - targeted offers or price adjustments aligned with demonstrated price sensitivity
  - operational improvements to reduce switching friction (e.g., digital self-service pathways)

The intent is not only to produce analysis, but to simulate how such findings could feed into **commercial decisions** — for example, estimating potential revenue impact, identifying segments with the strongest upside, or informing investment into retention programmes.

### Practical Use of Insights

In a real-world context, the outputs of churn analysis would typically be consumed by multiple departments:

- Strategy and Finance teams — to understand the financial implications of churn and model retention-based revenue scenarios
- Marketing and CRM teams — to design and target personalised retention campaigns
- Customer Operations — to prioritise outbound contact lists and allocate resources efficiently
- Product and Pricing — to refine bundles, data limits, or plan design based on observed behaviours

This project reflects that operational flow, taking the perspective of **how data would support decision-making across the organisation**, rather than focusing solely on technical outputs.



---

## 3. Project Objective
This project aims to:
- explore patterns of churn using a proxy CRM dataset
- understand behavioural characteristics that may relate to churn
- develop a predictive or segmentation-based component
- most importantly, simulate how churn analysis could support real-world commercial thinking in Aotearoa New Zealand

---

## 4. Dataset and Ethical Disclaimer
Dataset: JB-Link fictional telecom CRM dataset (public Kaggle source)  
Reason for use: New Zealand telco CRM data is commercially confidential and cannot be shared publicly.  
Ethics: No real NZ customer data is used, and no dataset in this project contains personally identifiable information (PII). This project is a simulation only.
