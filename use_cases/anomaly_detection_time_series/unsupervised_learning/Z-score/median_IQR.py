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