# 🏠 House Price Analysis  
*A practical end-to-end data analysis project inspired by Kaggle’s “House Prices” dataset.*

---

## 📘 Project Overview
This project walks through a complete **data analysis and prediction workflow** — from data cleaning and exploration to feature engineering, model building, and evaluation.  

The goal is to predict the final sale price of residential homes based on various features such as living area, number of garages, and overall quality.  

Everything here is designed to be **transparent, reproducible, and easy to follow** — the kind of project a data analyst would build to communicate both technical skill and reasoning.

---

## 🎯 Objectives
1. **Understand** the structure and meaning of the housing dataset.  
2. **Clean and prepare** the data for modeling (handling missing values, encoding categorical variables).  
3. **Explore** feature relationships and identify what really drives house prices.  
4. **Train and compare** a few baseline regression models.  
5. **Evaluate performance** using RMSE (Root Mean Squared Error).  
6. **Generate predictions** for unseen data.

---

## 🧰 Tools & Libraries
- **Python** 3.11+  
- **pandas**, **numpy** – data wrangling and transformation  
- **matplotlib** – basic data visualization  
- **scikit-learn** – preprocessing and regression modeling  
- **Jupyter Notebooks** – analysis and narrative documentation  

Install all dependencies easily:
```bash
pip install -r requirements.txt
```

---

## 📂 Repository Structure
```
house-price-analysis/
├─ data/                  # train.csv and test.csv go here
├─ notebooks/             # Jupyter notebooks for each step
│   ├─ 01_load_and_clean.ipynb
│   ├─ 02_eda.ipynb
│   ├─ 03_modeling.ipynb
│   └─ House_Prices_End_to_End.ipynb
├─ results/               # generated plots and submission.csv
├─ src/                   # modular Python scripts
│   ├─ preprocessing.py
│   ├─ visualization.py
│   └─ modeling.py
├─ requirements.txt       # dependencies
└─ README.md
```

---

## 📊 Exploratory Data Analysis
A few simple plots reveal the main story:

| Visualization | Description |
|----------------|-------------|
| **Histogram of log-transformed prices** | Shows that taking log stabilizes the right-skewed distribution of prices. |
| **GrLivArea vs SalePrice** | Larger living area strongly correlates with higher price. |
| **OverallQual vs SalePrice** | The overall quality rating is one of the most powerful predictors. |

Example plot:
```python
from src.visualization import plot_scatter
plot_scatter(train, "GrLivArea", "results/scatter_GrLivArea_SalePrice.png")
```

---

## 🤖 Modeling Approach
Three baseline regression models were compared:

| Model | Description | Validation RMSE |
|--------|--------------|----------------|
| **Linear Regression** | Simple and interpretable baseline | ~45,000 |
| **Random Forest** | Non-linear, ensemble approach | ~32,000 |
| **Gradient Boosting** | Sequential ensemble improving residuals | ~30,000 |

*(Values are approximate — your run may differ slightly depending on dataset and parameters.)*

---

## 📈 Results
The best performing model on validation data was **Gradient Boosting**, producing the lowest RMSE.  

Predicted prices were exported to:
```
results/submission.csv
```

If you’re using the included synthetic dataset, you’ll get a full working demo even without downloading Kaggle data.

---

## 🧮 Dataset
If you want to reproduce the full analysis with the **real Kaggle data**, download it from:  
👉 [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

Place these two files in the `data/` folder:
```
train.csv
test.csv
```

If not available, the project automatically generates a small **synthetic dataset** (`train_synthetic.csv`, `test_synthetic.csv`) that mimics the original structure — so all notebooks still run perfectly.

---

## 🧠 Key Takeaways
- **Data cleaning** matters as much as modeling — consistent encoding and scaling improve model stability.  
- **Feature importance** analysis helps understand real business drivers (like quality, area, year built).  
- **Baseline models** are valuable before jumping into complex architectures — simple methods often perform surprisingly well.  

---

## 🚀 How to Run Everything
1. Clone or download the repository  
2. Place the datasets inside the `data/` folder  
3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```
4. Run the end-to-end notebook  
   ```bash
   jupyter notebook notebooks/House_Prices_End_to_End.ipynb
   ```
5. View the output plots and generated `results/submission.csv`

---

## 💬 About the Author
Hi, I’m **Keivan Derafshian** — a Mathematics M.Sc. student and data analyst passionate about combining statistical reasoning with practical data science.  
This project demonstrates my workflow for building structured, reproducible analyses that are easy to understand, extend, and present to stakeholders.

---

## 🏁 Final Thoughts
This is not about leaderboard performance — it’s about **clarity, methodology, and communication**.  
If you’re reviewing this repository for hiring or collaboration, feel free to reach out or check my other analytics and data-science projects.
