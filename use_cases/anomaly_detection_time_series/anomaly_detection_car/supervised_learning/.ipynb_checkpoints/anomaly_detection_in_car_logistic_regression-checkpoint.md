
# Supervised learning - Logistic Regression

The **Isolation Forest** is an **unsupervised** machine learning algorithm, meaning it doesn't require labeled data to detect anomalies—it only learns from the data distribution. In unsupervised anomaly detection, the algorithm is trying to identify patterns that deviate significantly from the majority of the data, without prior knowledge of what constitutes "normal" or "anomalous."

In contrast, **supervised anomaly detection** requires labeled data where the anomalies are already known. In a supervised setting, the algorithm learns a model based on both **normal** and **anomalous** labeled data, and it attempts to predict the label for unseen data.

### Supervised Anomaly Detection:

In supervised anomaly detection, we would use **classification algorithms** to detect anomalies. The idea is to train a model on a labeled dataset (with known anomalies) and then predict whether new data points are normal or anomalous based on the learned model.

The basic steps for using a supervised algorithm for anomaly detection are:

1. **Data Labeling**: You must label the dataset, marking the instances of anomalies (usually with `1` for anomalies and `0` for normal data).

2. **Feature Engineering**: Choose relevant features that might help in distinguishing anomalies from normal data (e.g., temperature, pressure, time series features).

3. **Model Training**: Train a classifier like **Logistic Regression**, **Support Vector Machines (SVM)**, **Random Forests**, or **Gradient Boosting** using the labeled data.

4. **Prediction**: After training, use the model to classify new data as either normal or anomalous.

### Example of Supervised Anomaly Detection with a Classifier

Let's implement **Logistic Regression** as a simple classifier for anomaly detection. We will use the same dataset but now explicitly label the anomalies.

### 1. Simulating Supervised Labels:

First, let's manually label anomalies (i.e., use `1` for anomalies and `0` for normal data) and then train a supervised model.

#### Step 1: Labeling the Data

We’ll assume we have labeled anomalies (same anomalies as before) based on time periods.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Simulate true anomalies (1 = anomaly, 0 = normal)
true_anomalies = np.zeros(len(df_imputed))
true_anomalies[30:35] = 1  # Anomaly in temperature1
true_anomalies[50:55] = 1  # Anomaly in temperature2
true_anomalies[70:75] = 1  # Anomaly in temperature3

# Features: temperature sensors and pressure
features = df_imputed[['temperature1', 'temperature2', 'temperature3', 'temperature4', 'pressure']]

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Labels: true anomalies
labels = true_anomalies

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
```

#### Step 2: Training a Logistic Regression Classifier

Now, we'll train a **Logistic Regression** model to detect anomalies.

```python
# Create and train the Logistic Regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Print classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

#### Step 3: Understanding the Output

The **classification report** will include:

* **Precision**: The proportion of positive predictions that are actually correct (how many predicted anomalies are truly anomalies).
* **Recall**: The proportion of actual anomalies that are correctly identified (how many true anomalies were detected).
* **F1-Score**: The harmonic mean of precision and recall.
* **Support**: The number of occurrences of each class (normal vs. anomaly).

The **confusion matrix** will show:

* **True Positives (TP)**: Correctly classified anomalies.
* **False Positives (FP)**: Normal data incorrectly classified as anomalies.
* **True Negatives (TN)**: Correctly classified normal data.
* **False Negatives (FN)**: Anomalies incorrectly classified as normal.

### Example Output:

Here’s what the output might look like for the confusion matrix and classification report:

```
Classification Report:
               precision    recall  f1-score   support

           0       0.95      0.98      0.97        18
           1       0.85      0.75      0.80         6

    accuracy                           0.93        24
   macro avg       0.90      0.87      0.88        24
weighted avg       0.93      0.93      0.93        24

Confusion Matrix:
 [[17  1]
  [ 2  4]]
```

### Key Metrics:

* **Precision** for anomalies (`1`) tells us how many of the detected anomalies are actually anomalies.
* **Recall** for anomalies (`1`) tells us how many of the actual anomalies were correctly identified.
* **Accuracy** and **F1-score** give us an overall idea of the model’s performance.

### Visualizing the Results:

You could also visualize how well your model is performing by plotting the data and highlighting the predicted anomalies.

```python
# Plot predicted anomalies
plt.figure(figsize=(12, 6))

plt.plot(df_imputed['time'], df_imputed['temperature1'], label='Temperature1')
plt.plot(df_imputed['time'], df_imputed['temperature2'], label='Temperature2')
plt.plot(df_imputed['time'], df_imputed['temperature3'], label='Temperature3')
plt.plot(df_imputed['time'], df_imputed['temperature4'], label='Temperature4')

# Highlight anomalies in red (predicted anomalies)
predicted_anomalies = pd.Series(y_pred, index=X_test.index)
plt.scatter(df_imputed['time'][predicted_anomalies == 1], df_imputed['temperature1'][predicted_anomalies == 1], color='red', label='Predicted Anomalies')

plt.title('Temperature Over Time with Predicted Anomalies')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()
```

### Key Differences Between Supervised and Unsupervised Anomaly Detection:

1. **Supervised Anomaly Detection**:

   * Requires labeled data.
   * Models learn the characteristics of both **normal** and **anomalous** data.
   * After training, the model can predict whether a new data point is normal or anomalous based on the learned patterns.
2. **Unsupervised Anomaly Detection**:

   * Does not require labeled data.
   * Models try to find outliers or unusual data points based on the overall distribution of the data.
   * Anomalies are detected when a data point significantly deviates from the majority of the data.

### Conclusion:

* **Supervised learning** requires labeled data and is typically more reliable when you have clear definitions of anomalies.
* **Unsupervised learning** (like Isolation Forest) is ideal when you don't have labeled anomalies but still want to detect unusual patterns.

Supervised methods generally work better when you have clear, labeled data and anomalies are known beforehand. However, unsupervised methods can be useful when you don't have labeled data or if anomalies are not easily labeled.

Let me know if you'd like more details on how to tweak this approach or if you need further clarification!



# Improve the precision of the algorithm

To improve **precision** in anomaly detection, particularly with a **supervised learning** approach like **Logistic Regression**, you should consider several key adjustments in the process. Precision is defined as:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Where:

* **TP (True Positives)**: Correctly predicted anomalies.
* **FP (False Positives)**: Normal data incorrectly predicted as anomalies.

A high precision means that when the model predicts an anomaly, it is more likely to be correct (i.e., fewer false positives).

Here are some strategies to improve precision in supervised anomaly detection:

---

### 1. **Adjust the Decision Threshold**:

* **Logistic Regression** and other classifiers typically use a threshold of 0.5 to decide whether a prediction should be classified as `1` (anomaly) or `0` (normal). By default, it predicts the label `1` when the predicted probability is greater than or equal to 0.5.

* To **improve precision**, you can **increase the threshold**. This will make the model more conservative in predicting anomalies, potentially reducing false positives (FP) at the cost of increasing false negatives (FN).

For example, if you set a higher threshold, say `0.7`, the model will only classify a data point as an anomaly when its predicted probability of being an anomaly is above `70%`. This can help ensure that only very strong outliers are flagged as anomalies, which will likely improve precision.

Here’s how to adjust the decision threshold in **Logistic Regression**:

```python
# Make probability predictions instead of class predictions
y_prob = clf.predict_proba(X_test)[:, 1]  # Probabilities for class '1' (anomaly)

# Set a higher threshold (e.g., 0.7 instead of 0.5)
threshold = 0.7
y_pred_high_threshold = (y_prob >= threshold).astype(int)

# Evaluate precision with the new threshold
from sklearn.metrics import precision_score
print("Precision with threshold 0.7:", precision_score(y_test, y_pred_high_threshold))
```

### 2. **Balance the Classes (Address Class Imbalance)**:

* **Anomaly detection** often suffers from class imbalance (i.e., far fewer anomalies than normal data). If your dataset has very few anomalies, the model might be biased towards predicting normal data, which can hurt precision (because anomalies are rare and often overlooked).

**Ways to address class imbalance**:

* **Oversampling**: Increase the number of anomalies in the training set by generating synthetic anomalies (e.g., using **SMOTE** or random oversampling).
* **Undersampling**: Reduce the number of normal instances in the training set, though this might lead to loss of useful data.
* **Class weights**: Assign a higher weight to the anomaly class during model training so that the model pays more attention to anomalies.

#### Example using class weights in Logistic Regression:

```python
# Train Logistic Regression with class weights (weight anomalies more heavily)
clf = LogisticRegression(class_weight={0: 1, 1: 10})  # Give more weight to anomalies (class 1)
clf.fit(X_train, y_train)
```

This approach will help the model focus more on predicting anomalies (which are underrepresented) without affecting the detection of normal instances.

### 3. **Feature Engineering**:

Improving the **feature set** might help the classifier to more effectively distinguish between normal and anomalous data.

* **Time-Series Features**: If you're working with time-series data, consider creating additional features that capture trends and seasonality (e.g., moving averages, rolling windows, lag features, or Fourier transforms).
* **Interaction Features**: Sometimes interactions between features (e.g., the combination of temperature and pressure) may give more useful signals for detecting anomalies.

You can use **domain knowledge** to extract additional features, such as:

* **Temperature differences** between cylinders (if you're monitoring multiple sensors).
* **Rate of change** in temperature or pressure.
* **Rolling statistics** like moving averages and standard deviations.

### 4. **Change the Model or Algorithm**:

Logistic Regression might not always be the best for anomaly detection. Here are a few options to improve performance:

#### a. **Random Forest Classifier**:

Random Forest is an ensemble method that can better capture complex patterns in the data. It's robust to outliers and can naturally handle class imbalances.

```python
from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest model
clf_rf = RandomForestClassifier(class_weight={0: 1, 1: 10})  # Balance class weights
clf_rf.fit(X_train, y_train)

# Predict and evaluate precision
y_pred_rf = clf_rf.predict(X_test)
print("Random Forest Precision:", precision_score(y_test, y_pred_rf))
```

#### b. **Support Vector Machine (SVM)**:

Support Vector Machines with a **non-linear kernel** (e.g., radial basis function kernel) can also be effective for anomaly detection. SVM can be particularly good at creating a decision boundary that effectively separates normal data from anomalies.

```python
from sklearn.svm import SVC

# Train a Support Vector Classifier with class weights
clf_svm = SVC(class_weight={0: 1, 1: 10})
clf_svm.fit(X_train, y_train)

# Predict and evaluate precision
y_pred_svm = clf_svm.predict(X_test)
print("SVM Precision:", precision_score(y_test, y_pred_svm))
```

#### c. **Gradient Boosting Machines (GBM)**:

Gradient boosting algorithms like **XGBoost** or **LightGBM** can provide a good balance between speed and predictive power, and they are capable of modeling complex non-linear relationships.

### 5. **Cross-Validation and Hyperparameter Tuning**:

Fine-tuning the hyperparameters of your classifier can also help improve performance, including precision.

* **Grid Search** or **Randomized Search** can be used to tune hyperparameters like regularization (`C` for Logistic Regression), maximum depth of trees (for Random Forest), or kernel functions (for SVM).

Here’s an example of tuning the **C parameter** for Logistic Regression using grid search:

```python
from sklearn.model_selection import GridSearchCV

# Hyperparameter grid for Logistic Regression
param_grid = {'C': [0.1, 1, 10, 100]}

# Grid Search for hyperparameter tuning
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='precision')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predict and evaluate precision
y_pred_best = best_model.predict(X_test)
print("Tuned Logistic Regression Precision:", precision_score(y_test, y_pred_best))
```

### 6. **Increase the Training Data**:

More training data can help the model learn better patterns, especially when anomalies are rare. If you can get more labeled data or generate synthetic anomalies, you can help the model learn to better distinguish between normal and anomalous instances.

---

### Summary of Improvements for Precision:

1. **Increase the decision threshold** to reduce false positives.
2. **Address class imbalance** by using class weights, oversampling, or undersampling.
3. **Improve feature engineering**, especially if your data is time-series.
4. **Try different models** like Random Forest, SVM, or Gradient Boosting for better performance.
5. **Use hyperparameter tuning** to find the best model settings.
6. **Increase the size of the training data** to improve generalization.

By focusing on these aspects, you should be able to improve precision without sacrificing recall too much. Let me know if you'd like further details on any of these strategies!
