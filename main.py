import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import ruptures as rpt

def generate_data():
    first_interest = rd.randint(10, 34)
    third_interest = rd.randint(60, 100)
    data = []
    
    for i in range(40):
        for j in range(rd.randint(3, 5)):
            y = rd.randint(-8, 8) - 30 if first_interest <= i else rd.randint(-8, 8)
            data.append([i, y])

    for i in range(41, 42):
        shift = int((i - 41) / 11 * 80) + 30
        for j in range(12):  
            data.append([i, rd.randint(-8, 8) - shift])

    for i in range(42, 61):
        shift = int((61 - i) / 9 * 80)
        for j in range(12):  
            data.append([i, rd.randint(-8, 8) - shift])

    for i in range(61, 101):
        for j in range(rd.randint(3, 5)):
            y = rd.randint(-8, 8) + 30 if third_interest <= i else rd.randint(-8, 8)
            data.append([i, y])
    
    return np.array(data)

data = generate_data()
df = pd.DataFrame(data, columns=['x', 'y'])

grouped = df.groupby('x')['y'].apply(list)

def hampel_filter(series, window_size=7, threshold=3.0):
    clean_series = series.copy()
    rolling_med = series.rolling(window_size, center=True).median()
    rolling_mad = series.rolling(window_size, center=True).apply(lambda x: np.median(np.abs(x - np.median(x))))
    diff = np.abs(series - rolling_med)
    outliers = diff > threshold * (1.4826 * rolling_mad)
    clean_series[outliers.fillna(False)] = rolling_med[outliers.fillna(False)]
    return clean_series.interpolate().bfill().ffill()

# Apply Hampel filter to each group
df['clean_y'] = df.groupby('x')['y'].transform(hampel_filter)

df_mean = df.groupby('x')['clean_y'].mean()

x_vals = df['x'].values
y_vals = df['clean_y'].values

# PELT algorithm
model = "l2"
algo = rpt.Pelt(model=model).fit(y_vals)
penalty = max(5, np.std(y_vals) * 2)
breakpoints = algo.predict(pen=penalty)
if breakpoints and breakpoints[-1] == len(y_vals):
    breakpoints = breakpoints[:-1]

# Compute confidence scores
scores = []
last_cp = 0
filtered_bkps = []
filtered_scores = []
for cp in breakpoints:
    prev_segment = y_vals[last_cp:cp]
    next_cp = breakpoints[breakpoints.index(cp) + 1] if breakpoints.index(cp) + 1 < len(breakpoints) else len(y_vals)
    next_segment = y_vals[cp:next_cp]

    mean_before = np.mean(prev_segment) if len(prev_segment) > 0 else y_vals[0]
    mean_after = np.mean(next_segment) if len(next_segment) > 0 else y_vals[-1]

    score = abs(mean_after - mean_before) / (np.std(y_vals) + 1e-8)
    scores.append(score)

    CONFIDENCE_THRESHOLD = 0.32
    CONFIDENCE_CEILING = 0.55
    MIN_DIST = 5

    if score >= CONFIDENCE_THRESHOLD and score <= CONFIDENCE_CEILING:
        if not filtered_bkps or (cp - filtered_bkps[-1]) > MIN_DIST:
            filtered_bkps.append(cp)
            filtered_scores.append(score)

    last_cp = cp

plt.figure(figsize=(12, 6))
plt.scatter(df['x'], df['y'], label="Original Data", alpha=0.4)
plt.scatter(df['x'], df['clean_y'], label="Cleaned Data", color="blue", s=10)
plt.plot(df_mean.index, df_mean.values, label="Mean Line", color="green", linewidth=2)

for i, cp in enumerate(filtered_bkps):
    plt.axvline(x=df['x'].iloc[cp], color='red', linestyle='--', linewidth=1, label="Detected Shift" if i == 0 else "")
    plt.text(df['x'].iloc[cp], df['clean_y'].max() * 0.9, f"{filtered_scores[i]:.2f}", color='red', rotation=90, va='bottom')

plt.legend()
plt.title("Detected Shifts within Boundaries")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.tight_layout()
plt.show()
