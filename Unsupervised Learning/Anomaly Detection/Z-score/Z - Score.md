# Z score

Performing Z-score anomaly detection on a time series data is a common approach to identify unusual data points (anomalies). Here's how you can apply Z-score anomaly detection to time series data:

### Steps to Perform Z-Score Anomaly Detection:

1. **Compute the rolling mean and rolling standard deviation**: Since time series data can have trends and seasonality, it's better to calculate the mean and standard deviation over a moving window instead of using a global mean and standard deviation. This allows the Z-score to adapt to local patterns in the data.

2. **Calculate the Z-score**: For each point in the time series, calculate the Z-score as:

   Zt=Xt−μtσtZ_t = \frac{X_t - \mu_t}{\sigma_t}

   Where:

   - $X_t$ is the value of the time series at time $t$.
   - $\mu_t$ is the rolling mean at time $t$.
   - $\sigma_t$ is the rolling standard deviation at time $t$.

3. **Determine the threshold for anomaly detection**: Typically, a threshold of $\pm 3$ Z-scores is used to detect anomalies, meaning any point with a Z-score greater than 3 or less than -3 is considered an anomaly.

4. **Mark anomalies**: If the absolute value of the Z-score is above a predefined threshold, you classify that point as an anomaly.

### Example: Z-Score Anomaly Detection in Python

Here is a simple example of how to perform Z-score anomaly detection in a time series using Python and `pandas`.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example time series data (could be any time-based data)
np.random.seed(42)
data = np.random.randn(100) * 20 + 50  # Simulated data (normal distribution)
data[30] = 150  # Injecting an anomaly at index 30
data[60] = -100  # Injecting an anomaly at index 60

# Create a DataFrame
df = pd.DataFrame(data, columns=['Value'])

# Calculate rolling mean and rolling standard deviation
window_size = 10  # You can adjust this based on your data's seasonality
df['Rolling_Mean'] = df['Value'].rolling(window=window_size).mean()
df['Rolling_Std'] = df['Value'].rolling(window=window_size).std()

# Calculate Z-score
df['Z-Score'] = (df['Value'] - df['Rolling_Mean']) / df['Rolling_Std']

# Set a Z-score threshold for anomaly detection (typically, threshold > 3 or < -3)
threshold = 3
df['Anomaly'] = df['Z-Score'].apply(lambda x: 1 if abs(x) > threshold else 0)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df['Value'], label='Time Series Data')
plt.plot(df['Rolling_Mean'], label='Rolling Mean', linestyle='--', color='orange')
plt.scatter(df.index[df['Anomaly'] == 1], df['Value'][df['Anomaly'] == 1], color='red', label='Anomalies', zorder=5)
plt.title('Z-Score Anomaly Detection in Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Output the detected anomalies
anomalies = df[df['Anomaly'] == 1]
print("Anomalies detected at indices:", anomalies.index.tolist())
```

### Explanation of the Code:

1. **Data Simulation**:
   - We generate random data with a mean of 50 and a standard deviation of 20.
   - We then inject two anomalies into the time series at indices 30 and 60.
2. **Rolling Mean and Std**:
   - We compute the rolling mean and rolling standard deviation using a window size of 10. You can adjust the window size depending on your time series' seasonality or periodicity.
3. **Z-Score Calculation**:
   - The Z-score is calculated as the difference between the value and the rolling mean, divided by the rolling standard deviation.
4. **Anomaly Detection**:
   - Points with an absolute Z-score greater than 3 are considered anomalies. We use this threshold to flag the anomalies.
5. **Plotting**:
   - The plot visualizes the time series, the rolling mean, and the detected anomalies. Anomalies are marked with red dots.

### Output:

The output includes a plot with anomalies clearly marked, and the indices where anomalies are detected. The threshold for the Z-score can be adjusted based on how sensitive you want the anomaly detection to be.

### Notes:

- **Window Size**: The window size for the rolling mean and standard deviation should be chosen based on the characteristics of your time series (e.g., if the data has strong seasonality, the window size should reflect that periodicity).
- **Threshold Tuning**: The threshold of 3 is a common default, but you may want to tune it based on your data and the level of anomaly detection sensitivity you desire.

Let me know if you'd like to dive deeper into any of these steps or need further clarification!

Performing Z-score anomaly detection on a time series data is a common approach to identify unusual data points (anomalies). Here's how you can apply Z-score anomaly detection to time series data:

### Steps to Perform Z-Score Anomaly Detection:

1. **Compute the rolling mean and rolling standard deviation**: Since time series data can have trends and seasonality, it's better to calculate the mean and standard deviation over a moving window instead of using a global mean and standard deviation. This allows the Z-score to adapt to local patterns in the data.

2. **Calculate the Z-score**: For each point in the time series, calculate the Z-score as:

   Zt=Xt−μtσtZ_t = \frac{X_t - \mu_t}{\sigma_t}

   Where:

   - $X_t$ is the value of the time series at time $t$.
   - $\mu_t$ is the rolling mean at time $t$.
   - $\sigma_t$ is the rolling standard deviation at time $t$.

3. **Determine the threshold for anomaly detection**: Typically, a threshold of $\pm 3$ Z-scores is used to detect anomalies, meaning any point with a Z-score greater than 3 or less than -3 is considered an anomaly.

4. **Mark anomalies**: If the absolute value of the Z-score is above a predefined threshold, you classify that point as an anomaly.

### Example: Z-Score Anomaly Detection in Python

Here is a simple example of how to perform Z-score anomaly detection in a time series using Python and `pandas`.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example time series data (could be any time-based data)
np.random.seed(42)
data = np.random.randn(100) * 20 + 50  # Simulated data (normal distribution)
data[30] = 150  # Injecting an anomaly at index 30
data[60] = -100  # Injecting an anomaly at index 60

# Create a DataFrame
df = pd.DataFrame(data, columns=['Value'])

# Calculate rolling mean and rolling standard deviation
window_size = 10  # You can adjust this based on your data's seasonality
df['Rolling_Mean'] = df['Value'].rolling(window=window_size).mean()
df['Rolling_Std'] = df['Value'].rolling(window=window_size).std()

# Calculate Z-score
df['Z-Score'] = (df['Value'] - df['Rolling_Mean']) / df['Rolling_Std']

# Set a Z-score threshold for anomaly detection (typically, threshold > 3 or < -3)
threshold = 3
df['Anomaly'] = df['Z-Score'].apply(lambda x: 1 if abs(x) > threshold else 0)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df['Value'], label='Time Series Data')
plt.plot(df['Rolling_Mean'], label='Rolling Mean', linestyle='--', color='orange')
plt.scatter(df.index[df['Anomaly'] == 1], df['Value'][df['Anomaly'] == 1], color='red', label='Anomalies', zorder=5)
plt.title('Z-Score Anomaly Detection in Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Output the detected anomalies
anomalies = df[df['Anomaly'] == 1]
print("Anomalies detected at indices:", anomalies.index.tolist())
```

### Explanation of the Code:

1. **Data Simulation**:
   - We generate random data with a mean of 50 and a standard deviation of 20.
   - We then inject two anomalies into the time series at indices 30 and 60.
2. **Rolling Mean and Std**:
   - We compute the rolling mean and rolling standard deviation using a window size of 10. You can adjust the window size depending on your time series' seasonality or periodicity.
3. **Z-Score Calculation**:
   - The Z-score is calculated as the difference between the value and the rolling mean, divided by the rolling standard deviation.
4. **Anomaly Detection**:
   - Points with an absolute Z-score greater than 3 are considered anomalies. We use this threshold to flag the anomalies.
5. **Plotting**:
   - The plot visualizes the time series, the rolling mean, and the detected anomalies. Anomalies are marked with red dots.

### Output:

The output includes a plot with anomalies clearly marked, and the indices where anomalies are detected. The threshold for the Z-score can be adjusted based on how sensitive you want the anomaly detection to be.

### Notes:

- **Window Size**: The window size for the rolling mean and standard deviation should be chosen based on the characteristics of your time series (e.g., if the data has strong seasonality, the window size should reflect that periodicity).
- **Threshold Tuning**: The threshold of 3 is a common default, but you may want to tune it based on your data and the level of anomaly detection sensitivity you desire.

Let me know if you'd like to dive deeper into any of these steps or need further clarification!



## instead of mean, could we use the median?

Yes, absolutely! You can replace the rolling **mean** with the rolling **median** for anomaly detection. In fact, using the **median** can sometimes be more robust in the presence of **outliers** or **skewed data**, as the median is less sensitive to extreme values compared to the mean.

Here's how you can modify the Z-score anomaly detection approach to use the rolling median instead of the rolling mean:

### Modified Approach Using Median

1. **Compute the rolling median and rolling interquartile range (IQR)** instead of the mean and standard deviation. This will make the anomaly detection more robust to outliers.

2. **Calculate the Z-score** using the rolling median and rolling IQR.

   - You can scale the IQR to approximate a standard deviation (since it represents the spread of the data), and then compute the "Z-score" as:

   Zt=Xt−μtIQRtZ_t = \frac{X_t - \mu_t}{\text{IQR}_t}

   Where:

   - $X_t$ is the value at time $t$.
   - $\mu_t$ is the rolling median at time $t$.
   - $\text{IQR}_t$ is the rolling interquartile range (IQR) at time $t$.

3. **Set an anomaly detection threshold**: Typically, you can still use a threshold of 3 for the Z-score, but this is more flexible and can be adjusted depending on your specific data characteristics.

### Example Code Using Median for Z-Score Calculation

Here’s how you can implement this approach in Python:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example time series data (with injected anomalies)
np.random.seed(42)
data = np.random.randn(100) * 20 + 50  # Simulated data (normal distribution)
data[30] = 150  # Injecting an anomaly at index 30
data[60] = -100  # Injecting an anomaly at index 60

# Create a DataFrame
df = pd.DataFrame(data, columns=['Value'])

# Calculate rolling median and rolling interquartile range (IQR)
window_size = 10  # You can adjust this based on your data's seasonality
df['Rolling_Median'] = df['Value'].rolling(window=window_size).median()
df['Rolling_IQR'] = df['Value'].rolling(window=window_size).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

# Calculate Z-score using the rolling median and IQR
df['Z-Score'] = (df['Value'] - df['Rolling_Median']) / df['Rolling_IQR']

# Set a Z-score threshold for anomaly detection (typically, threshold > 3 or < -3)
threshold = 3
df['Anomaly'] = df['Z-Score'].apply(lambda x: 1 if abs(x) > threshold else 0)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df['Value'], label='Time Series Data')
plt.plot(df['Rolling_Median'], label='Rolling Median', linestyle='--', color='orange')
plt.scatter(df.index[df['Anomaly'] == 1], df['Value'][df['Anomaly'] == 1], color='red', label='Anomalies', zorder=5)
plt.title('Z-Score Anomaly Detection with Median in Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Output the detected anomalies
anomalies = df[df['Anomaly'] == 1]
print("Anomalies detected at indices:", anomalies.index.tolist())
```

### Key Differences in This Approach:

1. **Rolling Median**: Instead of the mean, we use the rolling median (`df['Rolling_Median']`).
2. **Rolling IQR**: Instead of the rolling standard deviation, we use the rolling interquartile range (`df['Rolling_IQR']`), which is calculated as the difference between the 75th and 25th percentiles of the rolling window.
3. **Z-Score Calculation**: The Z-score is then calculated using the median and IQR, which makes the detection more robust against extreme values or outliers.

### Plot and Results

The plot will show the time series with anomalies marked in red. The rolling median will be represented by a dashed orange line, and the Z-score-based anomaly detection will highlight the points that significantly deviate from the median, based on the scaled IQR.

### Why Use Median and IQR?

- **Median** is more robust than the mean to extreme values (outliers).
- **IQR** captures the spread of the middle 50% of data, so it is also more resistant to outliers than standard deviation.
- This approach is particularly helpful in cases where your data is **skewed** or has **heavy tails** (non-normal distributions).

### Considerations:

- **Sensitivity to Window Size**: The rolling window size (`window_size`) is important. If it's too small, the model might be too sensitive to short-term fluctuations, and if it's too large, it might miss anomalies that are more localized.
- **Threshold Adjustments**: You might want to experiment with different Z-score thresholds, depending on how aggressively you want to detect anomalies.

Let me know if you'd like more details or further adjustments!





## could I use Z-score if I have a non gaussian distribution?

Using the **Z-score** for anomaly detection is based on the assumption that your data follows a **normal distribution** (Gaussian distribution). This assumption might not hold if your data is **non-Gaussian** (e.g., skewed, bimodal, or has heavy tails), which can affect the accuracy and reliability of the Z-score.

However, the **Z-score** can still be applied in some cases with **non-Gaussian distributions**, but there are some important considerations and alternative approaches you might want to explore.

### When You Can Use Z-Score with Non-Gaussian Distributions:

1. **Data is approximately symmetric**: If the distribution is roughly symmetric (even if it's not perfectly normal), the Z-score might still work reasonably well. For example, **log-normal** or **beta distributions** may work, as long as they are not heavily skewed.
2. **Data with a single mode**: If the distribution has a single peak (unimodal) and is not highly skewed or has outliers that dominate the mean, then Z-scores can still be useful.
3. **Central Limit Theorem**: In some cases, even if the underlying data isn't normal, if you're aggregating large enough samples (or working with large datasets), the **Central Limit Theorem** can make the sample mean approximately normal, which means Z-scores may still be somewhat effective.

### Issues with Z-Score for Non-Gaussian Data:

1. **Skewed Distributions**: If the distribution is heavily skewed (like exponential or power-law distributions), the mean and standard deviation won't accurately represent the data’s spread, and Z-scores may give misleading results.
2. **Heavy Tails**: If the distribution has **heavy tails** (e.g., Cauchy or Pareto distributions), the Z-score will not capture outliers properly because it assumes data points further from the mean are "rare" or outliers, but in heavy-tailed distributions, extreme values might occur more often than expected.
3. **Multimodal Distributions**: If the data is **multimodal** (has multiple peaks), using a single mean and standard deviation will fail to capture the true structure of the data, making the Z-score ineffective.

### Alternatives to Z-Score for Non-Gaussian Data:

If your data does not follow a Gaussian distribution, there are several alternative anomaly detection methods that may be more suitable:

1. **IQR (Interquartile Range)**:
   - The IQR method is robust to skewed data and outliers. It uses the 25th and 75th percentiles to define the spread of the central 50% of the data.
   - **Formula**: $\text{IQR} = Q_3 - Q_1$, where $Q_1$ and $Q_3$ are the first and third quartiles, respectively. Outliers are typically points that are less than $Q_1 - 1.5 \times \text{IQR}$ or greater than $Q_3 + 1.5 \times \text{IQR}$.
2. **Isolation Forest**:
   - This is a tree-based algorithm that works well with high-dimensional and non-normal data. It isolates observations by randomly partitioning the data, and anomalies are easier to isolate.
   - **Good for high-dimensional data** and non-Gaussian distributions.
3. **Local Outlier Factor (LOF)**:
   - LOF is an unsupervised method that measures the local density deviation of a data point with respect to its neighbors. It’s effective when the data has clusters and varying densities.
   - This method is useful for **non-Gaussian** data with different density distributions in different regions.
4. **One-Class SVM**:
   - This algorithm is good for anomaly detection in cases where you only have one class of data (i.e., no labeled anomalies). It learns a boundary around the "normal" data points and flags points outside that boundary as anomalies.
   - Works well with data that is not normally distributed.
5. **Autoencoders** (Deep Learning):
   - Autoencoders are neural networks trained to compress and then reconstruct input data. Anomalies are detected by comparing the reconstruction error. This method is very flexible and can capture complex patterns in data.
   - It works well for data with complex structures, including non-Gaussian data.
6. **Quantile-based Methods**:
   - Quantile-based anomaly detection methods (like using percentiles) can be more robust to non-normal data, especially when your data is skewed or has long tails.
7. **K-Means Clustering (with distance to centroids)**:
   - If your data has distinct clusters, the distance from a point to the cluster centroid can be used to detect anomalies. Points that are far from the centroid can be flagged as anomalies.

### Example: Using IQR for Anomaly Detection in Non-Gaussian Data

Here’s a Python example that demonstrates using the **IQR** method for anomaly detection in non-Gaussian data:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example non-Gaussian data (skewed distribution)
np.random.seed(42)
data = np.random.exponential(scale=20, size=100)  # Exponentially distributed data
data[30] = 150  # Injecting an anomaly at index 30
data[60] = -100  # Injecting an anomaly at index 60

# Create a DataFrame
df = pd.DataFrame(data, columns=['Value'])

# Calculate IQR (Interquartile Range)
Q1 = df['Value'].quantile(0.25)
Q3 = df['Value'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Mark anomalies
df['Anomaly'] = df['Value'].apply(lambda x: 1 if x < lower_bound or x > upper_bound else 0)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df['Value'], label='Time Series Data')
plt.scatter(df.index[df['Anomaly'] == 1], df['Value'][df['Anomaly'] == 1], color='red', label='Anomalies', zorder=5)
plt.title('IQR Anomaly Detection in Non-Gaussian Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Output the detected anomalies
anomalies = df[df['Anomaly'] == 1]
print("Anomalies detected at indices:", anomalies.index.tolist())
```

### Explanation:

- **Exponential Distribution**: In this example, we use an exponential distribution, which is non-Gaussian and skewed.
- **IQR Method**: We calculate the **Interquartile Range (IQR)** and then use it to identify anomalies. Points outside the bounds defined by $Q_1 - 1.5 \times \text{IQR}$ or $Q_3 + 1.5 \times \text{IQR}$ are flagged as anomalies.
- **Plot**: The anomalies are shown as red dots on the plot.

### Conclusion:

If your data is non-Gaussian, it's better to consider alternative methods like **IQR**, **Isolation Forest**, or **LOF**. While the Z-score can be applied in certain cases (e.g., roughly symmetric, unimodal distributions), its performance will degrade significantly for highly skewed, multimodal, or heavy-tailed distributions.
