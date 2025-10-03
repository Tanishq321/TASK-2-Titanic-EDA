# 🚢 Titanic Dataset - Data Cleaning & Exploratory Data Analysis (EDA)

This project performs **data cleaning** and **exploratory data analysis (EDA)** on the famous **Titanic Dataset** from Kaggle.  
The goal is to explore survival patterns and trends based on **gender, passenger class, age, and fare**.

---

## 📊 Steps Performed

### 🔹 Data Cleaning
- Handled missing values:
  - Filled missing **Age** with median.
  - Filled missing **Embarked** with mode.
- Extracted **Titles** from passenger names.
- Encoded categorical variables (e.g., **Sex → Male/Female**).
- Dropped unnecessary columns (`PassengerId, Name, Ticket, Cabin`).

### 🔹 Exploratory Data Analysis (EDA)
1. **Survival Count** → Overall survival numbers.  
2. **Survival by Gender** → Compared male vs female survival rates.  
   - 🟦 **Blue = Male**  
   - 🟪 **Pink = Female**  
3. **Survival by Passenger Class** → Impact of socio-economic class.  
4. **Age Distribution with Survival** → Younger passengers had better survival chances.  
5. **Fare Distribution with Survival** → Higher fare correlated with higher survival.  
6. **Correlation Heatmap** → Numeric variable relationships.

---

## 📈 Key Insights
- **Females had a much higher survival rate (74%) than males (18%).**  
- **First-class passengers survived more (63%) compared to third class (24%).**  
- **Children and younger passengers had better chances of survival.**  
- **Higher fare tickets increased the likelihood of survival.**

---

## 🛠️ Tech Stack
- Python 🐍
- Pandas
- Seaborn
- Matplotlib

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python titanic.py
