# JMI Commerce — Executive Decision Support System

## 1. System Overview & Description
**JMI Commerce** is an executive-level retail business intelligence and automated decision support dashboard. Built exclusively for a single user persona—the **business store manager or retail decision-maker**—JMI Commerce eliminates manual guesswork by transforming over **341,000+ transactional sales and inventory records** (`Retail_sales.csv`) into predictive operational controls.

Unlike consumer e-commerce platforms that serve shopping carts, JMI Commerce is an internal portal focused on supply chain optimization, monthly sales forecasting, stock movement audit flags, product substitution matching, and automated inventory cataloging.

---

## 2. Business Vision, Mission, and Goals

### Company Vision
To empower JMI Commerce retail leadership with automated, data-driven operational intelligence that transforms raw retail sales, warehouse transfer logs, and supplier inventories into proactive, profitable stock strategies.

### Company Mission
To equip the store business manager with a consolidated executive decision portal utilizing core machine learning algorithms—ranging from monthly sales forecasting to transfer anomaly detection and product substitution—ensuring balanced stock holding, eliminated stockouts, and maximum gross revenue.

### Strategic & Operational Goals
1. **Goal 1: Operational Precision (0% Stockouts):** Eliminate retail stock gaps and dead inventory by grounding monthly stock purchases in linear regression sales projections.
2. **Goal 2: Revenue Protection (-95% Audit Flags):** Automatically flag anomalous transfers, inventory shrinkage, and misreported store sales using Support Vector Machine (SVM) decision boundaries.
3. **Goal 3: Catalog Velocity (+30% Substitution Rate):** Instantly match out-of-stock items with nearest in-stock catalog substitutes using $k$-Nearest Neighbors ($k$-NN) Euclidean distance matching.

---

## 3. Retail Pain Points & Machine Learning Solution Matrix

| Machine Learning Algorithm | Underlying Retail Issue | JMI Commerce Application | Contribution to Business Goals |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Unpredictable sales volume leading to costly holding fees or lost customer transactions. | Monthly Retail Sales Forecasting Module | **Operational Precision:** Projects continuous sales figures from warehouse and transfer inputs. |
| **Decision Tree Classifier** | Dead stock and slow-moving items taking up valuable retail and warehouse shelf space. | Slow-Moving & Inventory Risk Auditor | **Operational Precision:** Branching logic flags slow-moving stock for clearance. |
| **Support Vector Machine (SVM)** | Transfer discrepancies, unverified stock movements, and inventory shrinkage risks. | Transfer Anomaly & Audit Detection Queue | **Revenue Protection:** Isolates abnormal stock movements prior to month-end accounting. |
| **$k$-Nearest Neighbors ($k$-NN)** | Out-of-stock catalog items resulting in lost sales when exact brands are unavailable. | Catalog Product Substitute & Attribute Matcher | **Catalog Velocity:** Finds nearest feature-matched substitutes by Euclidean distance. |
| **Naive Bayes Classifier** | Unstructured product description cataloging causing misplaced supplier category tags. | Product Description & Type Categorizer | **Operational Precision:** Automates category tagging (WINE, LIQUOR, BEER, etc.) from invoice text. |
| **Logistic Regression** | Inability to predict which warehouse restocks will achieve top-tier retail sales conversion. | High-Velocity Retail Conversion Predictor | **Catalog Velocity:** Calculates Sigmoid conversion probability for restock batches. |
| **Artificial Neural Network (ANN)** | Multi-variable non-linear sales dynamics, complex promotional interactions, and multi-factor stockout risk. | Deep Demand & Dynamic Stockout Risk Predictor | **Operational Precision:** Combines sales, catalog, supplier, timing, and stock scenario inputs for demand and replenishment planning. |

---

## 4. Detailed Algorithm Explanations & Non-Technical Guides

### 1. Linear Regression — Sales Forecasting
* **Non-Technical Analogy:** Imagine a store ruler. If you know that transferring 10 cases from the warehouse consistently generates $9.95 in retail sales, Linear Regression draws that exact trend line so managers can project monthly income before placing stock orders.
* **Why it's necessary:** Eliminates speculative inventory purchasing and maintains cash flow balance.

### 2. Decision Tree — Inventory Risk Audit
* **Non-Technical Analogy:** Imagine a decision flowchart at the store manager's desk: *"Is retail sales under $1?"* → If Yes, *"Is warehouse stock over 5 units?"* → If Yes, tag as *"Slow-Moving / High Holding Risk"*.
* **Why it's necessary:** Frees up working capital by highlighting dead inventory early.

### 3. Support Vector Machine (SVM) — Anomaly Detection
* **Non-Technical Analogy:** Imagine a high-security perimeter fence separating normal store transactions from suspicious, anomalous stock transfers. SVM calculates an optimal separation boundary to isolate transfer anomalies.
* **Why it's necessary:** Protects profits from misallocated stock transfers and unrecorded inventory shrinkage.

### 4. $k$-Nearest Neighbors ($k$-NN) — Product Substitutes
* **Non-Technical Analogy:** When a customer requests an out-of-stock wine or beer, an experienced store clerk finds another bottle with similar price, sales volume, and supplier profile. KNN calculates Euclidean feature distance to locate exact substitutes in the catalog.
* **Why it's necessary:** Prevents lost transactions by keeping floor staff equipped with instant stock alternatives.

### 5. Naive Bayes — Item Categorization
* **Non-Technical Analogy:** If an incoming invoice text contains words like `'CABERNET'`, `'CHARDONNAY'`, or `'750ML'`, Bayes' Theorem calculates the highest conditional probability that the product belongs to the `'WINE'` category.
* **Why it's necessary:** Automates inventory catalog ingestion for thousands of raw supplier records.

### 6. Logistic Regression — High-Velocity Conversion
* **Non-Technical Analogy:** Think of a top-seller meter. Higher warehouse volume and transfers raise the probability curve toward 100%, signaling that the product will become a top-performing retail item.
* **Why it's necessary:** Directs promotional shelf space toward high-probability top-selling products.

---

## 5. System Execution Guide

### Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

### Train the ANN Demand Model
Run the training script from the project directory after installing dependencies:
```bash
python train_ann_model.py
```
The script reads `Retail_sales.csv`, creates an 80/20 train/test split, scales numeric inputs, one-hot encodes product/supplier categories, trains an `MLPRegressor`, reports RMSE/MAE/R2, and writes `ann_model.pkl`. The generated artifact is intentionally excluded from source control and must be regenerated for each environment. Start the Flask app with `python app.py`; the ANN endpoint is `POST /api/ann/predict`.

The CSV has no promotion history, on-hand stock, or stockout labels. The ANN therefore predicts retail sales from available historical fields; promotion uplift is a user-entered scenario, and the displayed stockout exposure is an uncovered-demand estimate rather than a learned or calibrated probability. The reported random holdout metrics do not establish future-period accuracy; temporal validation needs denser, sequential sales history.