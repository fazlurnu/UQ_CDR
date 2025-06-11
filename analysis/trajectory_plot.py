import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# --- Load Data ---
df = pd.read_csv('../results/BlueSky/trajectories/20/dpsi_2/trajectory_p_oi_m_20_15_0_1.csv')

# --- Utility functions ---
def parse_array_string(arr_str):
    cleaned = arr_str.strip('[]')
    return np.array([float(x) for x in cleaned.split()])

def parse_list_string(list_str):
    return ast.literal_eval(list_str)

# --- Build long-format DataFrame ---
all_points = []

for idx, row in df.iterrows():
    ids = parse_list_string(row['id'])
    lats = parse_array_string(row['lat'])
    lons = parse_array_string(row['lon'])

    temp_df = pd.DataFrame({
        'id': ids,
        'lat': lats,
        'lon': lons
    })
    all_points.append(temp_df)

points_df = pd.concat(all_points, ignore_index=True)
points_df['id'] = points_df['id'].astype(str)

# --- Plotting ---
plt.figure(figsize=(10, 8))
dro_ids = sorted(set(i for i in points_df['id'] if i.startswith('DRO')))

for dro_id in dro_ids:
    dri_id = dro_id.replace('DRO', 'DRI')
    dro_data = points_df[points_df['id'] == dro_id]
    dri_data = points_df[points_df['id'] == dri_id]

    if dro_data.empty or dri_data.empty:
        continue

    # Normalize to DRO origin
    origin_lat, origin_lon = dro_data.iloc[0]['lat'], dro_data.iloc[0]['lon']
    lat_factor = 111_320  # meters per degree latitude
    lon_factor = 111_320 * np.cos(np.radians(origin_lat))  # meters per degree longitude

    dro_x = (dro_data['lon'] - origin_lon) * lon_factor
    dro_y = (dro_data['lat'] - origin_lat) * lat_factor
    dri_x = (dri_data['lon'] - origin_lon) * lon_factor
    dri_y = (dri_data['lat'] - origin_lat) * lat_factor

    plt.plot(dro_x, dro_y, color='tab:blue', alpha=0.2)
    plt.plot(dri_x, dri_y, color='tab:red', alpha=0.2)

plt.xlabel('East Offset (meters)')
plt.ylabel('North Offset (meters)')
plt.axis('equal')
plt.tight_layout()
plt.show()
