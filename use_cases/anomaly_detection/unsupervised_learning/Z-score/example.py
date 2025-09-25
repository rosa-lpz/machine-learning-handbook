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
