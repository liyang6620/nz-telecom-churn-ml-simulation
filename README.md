# NZ Telecom Churn Simulation  
Commercial Analytics Case – Customer Retention & Revenue Protection

---

## Executive Summary

This project demonstrates how customer churn analytics can be operationalised within a New Zealand telecommunications provider to support retention prioritisation, revenue protection, and customer lifetime value optimisation in a mature and highly competitive market.

Due to the commercial sensitivity of real New Zealand telecom CRM data, a publicly available fictional telecom dataset is used solely as a proxy. The emphasis of this project is not model experimentation, but how churn insights are generated, interpreted, and applied in practice across strategy, CRM, customer operations, and product teams.

The deliverables include an end-to-end analytical workflow (exploratory analysis, feature design, risk scoring) and a lightweight operational scoring interface deployed via Streamlit to simulate real-world usage.

**Primary Stakeholders:** Strategy, CRM, Customer Operations, Product & Pricing  
**Decision Focus:** Churn risk identification, retention prioritisation, revenue-at-risk management

---

## Business Context — Why Churn Matters in New Zealand

**Source:** Commerce Commission New Zealand — Telecommunications Monitoring Report (30 June 2025)

The New Zealand telecommunications market is highly concentrated. Three mobile network operators — 2degrees, Spark, and One NZ — collectively account for approximately 97.5% of the mobile market, while a small number of MVNOs hold only a minor share. A similar concentration exists in broadband, where the top three providers account for approximately 73% of connections.

Network availability is no longer a primary source of differentiation. Fibre coverage is near-universal in urban areas, and more than 90% of households have access to 4G fixed wireless services. As a result, churn is driven less by infrastructure access and more by pricing, bundling, perceived value, service experience, and switching friction.

Despite this, switching remains limited. Only around 11% of broadband households switched providers in the six months to June 2024. Among those who attempted to switch, nearly one-third reported difficulty or dissatisfaction with the process. Product bundling — such as broadband combined with electricity — has further increased customer stickiness and was cited by 11% of non-switchers as a reason for remaining with their provider.

In this environment, churn analytics is not a niche modelling task, but a core commercial capability supporting retention strategy, financial planning, and customer experience design.

---

## Commercial Significance of Churn

In a mature and saturated market, sustainable growth depends more on retaining existing customers than acquiring new ones. Acquisition costs in telecommunications are typically several times higher than retention costs, meaning even modest reductions in churn can produce material improvements in revenue stability and customer lifetime value.

From a commercial perspective, churn directly influences revenue forecasting and financial planning, customer lifetime value modelling, prioritisation of CRM and customer operations resources, effectiveness of retention campaigns and incentives, and product design, pricing, and bundling strategy.

Churn analytics therefore functions as a cross-functional decision-support system, rather than a narrow predictive exercise.

---

## Project Scope

This project simulates how churn analytics would be used when churn is treated as a strategic business issue rather than a modelling challenge.

The scope includes identifying customers with elevated churn risk prior to departure, analysing behavioural, contractual, pricing, and service-related drivers of churn, segmenting customers to support prioritised retention approaches, and translating analytical outputs into commercially actionable insights.

Simulated applications include targeted retention outreach for high-risk customers, bundle or plan recommendations aligned with observed usage behaviour, pricing and incentive strategies informed by price sensitivity, and operational improvements aimed at reducing switching friction.

---

## Business Questions Addressed

This project is structured around decision-oriented questions commonly faced by telecom providers:

- Which customers represent the highest likelihood of churn?
- What behavioural or contractual characteristics are most strongly associated with churn?
- How should retention resources be prioritised under operational and budget constraints?
- Which churn outcomes are likely preventable versus structurally unavoidable?
- How can churn insights inform pricing, bundling, and customer experience strategy?

---

## Analytical Approach 

The analytical workflow mirrors common industry practice and is designed to be transferable to real telecom CRM environments.

Key components include structured data understanding and leakage control, exploratory analysis of churn patterns and customer segments, feature engineering focused on pre-churn observable signals, development of a probability-based churn risk score for ranking and segmentation, and evaluation using lift and decile concentration rather than accuracy alone.

Model outputs are treated as inputs into human decision-making rather than automated triggers.

---

## Key Analytical Insights

Several consistent patterns emerge from the analysis:

- Contract structure and tenure are the strongest structural drivers of churn. Month-to-month and short-tenure customers exhibit substantially higher churn, reflecting low switching friction and limited customer attachment.
- Pricing exposure and billing characteristics matter more than absolute price levels. Higher monthly charges, billing volatility, and certain payment methods are associated with elevated churn risk.
- Service configuration and optional add-ons influence customer stickiness. Customers with value-added services show lower churn, consistent with ecosystem dependency.
- Customer interaction signals act as near-term early-warning indicators of churn.
- Churn is not purely a low-usage problem. Higher-usage customers can be more likely to churn due to higher expectations and greater sensitivity to perceived value.

---

## Modelling and Risk Scoring

A gradient-boosted decision tree model is used to capture non-linear relationships and interactions across contract, pricing, service, usage, and customer experience variables.

The model outputs a churn risk score (probability) rather than a binary churn label, enabling customer ranking, prioritisation, and segmentation under capacity and budget constraints.

Evaluation focuses on risk concentration and lift, demonstrating that churn is heavily concentrated in the highest-risk segments, making targeted retention commercially efficient.

---

## Operationalisation: Streamlit Scoring Application

To simulate real-world usage, a lightweight Streamlit application is included and deployed.

**Live App:**  
https://nz-telecom-churn-ml-simulation-8murruu2syd4x2wry5ikcd.streamlit.app/

The app loads persisted model and preprocessing artefacts, accepts a customer snapshot CSV, applies consistent feature preparation and schema alignment, and outputs a churn risk score for each customer suitable for CRM or operational use.

---

## Intended Operational Use

In a production setting, insights from this analysis would typically support strategy and finance teams in estimating revenue at risk, CRM and marketing teams in designing targeted retention campaigns, customer operations teams in prioritising outbound contact, and product and pricing teams in reviewing plan structures, bundles, and perceived customer value.

Churn outputs are designed to complement human judgement rather than replace it.

---

## Dataset Source and Ethics

**Dataset:** JB-Link Telco Customer Churn Dataset  
**Source:** Kaggle – John Flag  
https://www.kaggle.com/datasets/johnflag/jb-link-telco-customer-churn

The dataset is a publicly available fictional telecom CRM dataset and is used solely as a proxy to demonstrate churn analytics workflows. No real New Zealand customer data is used, and the dataset contains no personally identifiable information. All analysis is conducted for demonstration and portfolio purposes only.

---



