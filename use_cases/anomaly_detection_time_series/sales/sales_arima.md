# Arima for multivariate time series

The ARIMA model is traditionally used for **univariate** time series. However, for **multivariate time series**, we typically use an extension of ARIMA called **VAR (Vector AutoRegression)** or **ARIMAX/SARIMAX**, which allows for exogenous variables.

Let me give you a practical example using **ARIMAX**, which is essentially ARIMA with external (exogenous) regressors.

---

## 📘 Example: Forecasting Sales Using ARIMAX

### **Goal:**

Predict future **sales** using a multivariate time series dataset that includes:

* `Sales` (target variable)
* `Advertising` (exogenous variable)
* `Price` (exogenous variable)

---

### **1. Import Libraries**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
```

---

### **2. Load Example Data**

Assume we have the following time series data:

```python
# Create a toy dataset
np.random.seed(42)
date_range = pd.date_range(start='2020-01-01', periods=100, freq='W')
sales = np.random.normal(200, 10, size=100) + np.linspace(0, 50, 100)  # trend
advertising = np.random.normal(30, 5, size=100)
price = np.random.normal(10, 1, size=100)

df = pd.DataFrame({
    'Date': date_range,
    'Sales': sales,
    'Advertising': advertising,
    'Price': price
})
df.set_index('Date', inplace=True)
df.head()
```

---

### **3. Train-Test Split**

```python
train = df.iloc[:-12]
test = df.iloc[-12:]

y_train = train['Sales']
X_train = train[['Advertising', 'Price']]
y_test = test['Sales']
X_test = test[['Advertising', 'Price']]
```

---

### **4. Fit ARIMAX (SARIMAX with no seasonality)**

```python
model = SARIMAX(y_train,
                exog=X_train,
                order=(1, 1, 1),  # ARIMA(p,d,q)
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)
```

---

### **5. Forecast**

```python
forecast = results.predict(start=len(train), end=len(train) + len(test) - 1, exog=X_test)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Sales'], label='Actual Sales')
plt.plot(test.index, forecast, label='Forecasted Sales', linestyle='--')
plt.legend()
plt.title('ARIMAX Forecast')
plt.show()
```

---

### ✅ **Result:**

You now have a forecast of `Sales` that incorporates external factors like `Advertising` and `Price`, which may help improve accuracy over a univariate ARIMA model.

---

### 📌 Summary

* Use **ARIMA** for univariate forecasting.
* Use **ARIMAX** or **SARIMAX** when you want to include additional (exogenous) variables.
* Use **VAR** if you're modeling several **interdependent** time series together (e.g. forecasting multiple variables simultaneously).

---

Would you like an example using **VAR** instead (true multivariate modeling)?
