import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from river import drift

def generate_data():
    first_interest = rd.randint(10, 25)
    third_interest = rd.randint(67, 100)
    data = []
    
    for i in range(40):
        for j in range(rd.randint(3, 5)):
            y = rd.randint(-8, 8) - 30 if (first_interest <= i and 32 >= i) else rd.randint(-8, 8)
            data.append([i, y])

    for i in range(41, 42):
        shift = int((i - 41) / 11 * 80)
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

ph = drift.PageHinkley(
    delta=0.5,
    threshold=100,
    alpha=0.9999
)

significant_values = {}
detection_order = []

for i in range(len(df)):
    y = df.loc[i, 'y']
    x = df.loc[i, 'x']
    ph.update(y)
    if ph.drift_detected and x not in significant_values:
        print(f"Change detected at index {i}, x={x}, y={y}")
        significant_values[x] = y
        detection_order.append(x)

all_x = list(significant_values.keys())
to_remove = set()

for i in range(len(all_x)):
    for j in range(i + 1, len(all_x)):
        if abs(all_x[i] - all_x[j]) <= 6:
            to_remove.add(all_x[i])
            to_remove.add(all_x[j])

filtered_significant = {
    x: y for x, y in significant_values.items() if x not in to_remove
}

plt.scatter(df['x'], df['y'], label='Data')

for x in filtered_significant.keys():
    plt.axvline(x=x, color='red', linestyle='--')

corrected_df = df.copy()
change_points = sorted(filtered_significant.keys())

for change in change_points:
    before_window = corrected_df[(corrected_df['x'] >= change - 6) & (corrected_df['x'] < change)]
    after_window = corrected_df[(corrected_df['x'] > change) & (corrected_df['x'] <= change + 6)]

    if not before_window.empty and not after_window.empty:
        before_mean = before_window['y'].mean()
        after_mean = after_window['y'].mean()
        offset = after_mean - before_mean
        mask = corrected_df['x'] >= change
        pre_point = corrected_df[corrected_df['x'] == (change - 1)]
        if not pre_point.empty:
            pre_y = pre_point['y'].values[0]
            if abs(pre_y - after_mean) < abs(pre_y - before_mean):
                mask |= (corrected_df['x'] == (change - 1))

        corrected_df.loc[mask, 'y'] -= offset


plt.figure(figsize=(10, 6))
plt.scatter(corrected_df['x'], corrected_df['y'], label='Corrected Data')
plt.title("Corrected Data After Simplified Jump Adjustment")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
