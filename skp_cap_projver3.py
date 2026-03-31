#Capstone, ML implementation
#File function: read data from DNR, read true data -> train data -> output results to new validated file
#Do a ML based on random forest regression algorithm
#done:Convert all time from UTC to UTC-8 Alaska Time

import matplotlib.pyplot as plt
import os
import re
import glob
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

###All definitions section
###
###
#read folder full of CSVs
SENSOR_FOLDER = "sensor_csvs"
SNOW_FILE = "Snow Monitor Depths Actual vs Recorded(1).csv"
REFERENCE_2024_FILE = "2024.csv"

ACTUAL_COLS = [4, 6, 8, 10, 12, 14, 16]
UTC_ZONE = "UTC" #TIMEZONE def
ALASKA_ZONE = "America/Anchorage"

#done:Load actual snow depth values, each one is done in rows for matching ID in actual_cols

def extract_site_id_from_filename(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None

#This is for matching each site with true snow data

def load_truth_table(snow_file):
    snow_df = pd.read_csv(snow_file, header=None)

    measurement_dates = []
    for c in ACTUAL_COLS:
        dt = pd.to_datetime(snow_df.iloc[0, c], errors="coerce")
        measurement_dates.append(dt.date() if pd.notna(dt) else None)

    return snow_df, measurement_dates


def get_truth_for_site(snow_df, site_id, measurement_dates):
    site_rows = snow_df[snow_df[2].astype(str).str.strip() == str(site_id)]
    if site_rows.empty:
        return None

    site_row = site_rows.iloc[0]

    true_depths = []

    valid_dates = []
    # Build from all the snow data we get per day, read timestamp, date, create new columns into a feature from CSV

    for c, dt in zip(ACTUAL_COLS, measurement_dates):
        val = pd.to_numeric(site_row[c], errors="coerce")
        if dt is not None and pd.notna(val):
            valid_dates.append(dt)
            true_depths.append(float(val))

    if len(valid_dates) == 0:
        return None
    return pd.DataFrame({
        "date": valid_dates,
        "true_snow_depth_cm": true_depths
    })


# ----------- NEW FEATURE ENGINEERING -----------
def build_daily_features(sensor_csv_path):
    df = pd.read_csv(sensor_csv_path)

    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' col not in {sensor_csv_path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(ALASKA_ZONE)
    df["date"] = df["timestamp"].dt.date

    sensor_cols = [c for c in df.columns if c.endswith(" m")]
    if len(sensor_cols) == 0:
        raise ValueError(f"no sensor cols in {sensor_csv_path}")
#Uitlize mean,min,max temp variance for training
    daily_mean = df.groupby("date")[sensor_cols].mean()
    daily_mean.columns = [f"{c}_mean" for c in sensor_cols]

    daily_min = df.groupby("date")[sensor_cols].min()
    daily_min.columns = [f"{c}_min" for c in sensor_cols]

    daily_max = df.groupby("date")[sensor_cols].max()
    daily_max.columns = [f"{c}_max" for c in sensor_cols]

    daily_var = df.groupby("date")[sensor_cols].var()
    daily_var.columns = [f"{c}_var" for c in sensor_cols]

#Gradient features for ML train
    sorted_cols = sorted(sensor_cols, key=lambda x: float(x.split()[0]))

    gradient_features = {}
    for i in range(len(sorted_cols) - 1):
        lower = sorted_cols[i]
        upper = sorted_cols[i + 1]
        gradient_features[f"grad_{lower}_to_{upper}"] = df[upper] - df[lower]

    gradient_df = pd.DataFrame(gradient_features)
    gradient_df["date"] = df["date"]

    daily_grad = gradient_df.groupby("date").mean()

#Combine all daily feautres
    daily = pd.concat(
        [daily_mean, daily_min, daily_max, daily_var, daily_grad],
        axis=1
    ).reset_index()

    return daily

#Hardcode: 2024 ref data real dates
def load_2024_reference_data():
    data = [
        ("2024-10-16", 0),
        ("2024-10-20", 0),
        ("2024-10-26", 0),
        ("2024-10-29", 7.5),
        ("2024-10-31", 0),
        ("2024-11-01", 8),
        ("2024-11-02", 7),
        ("2024-11-04", 5),
        ("2024-11-05", 4),
        ("2024-11-06", 3.5),
    ]

    df = pd.DataFrame(data, columns=["date", "depth_in"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["true_snow_depth_cm"] = df["depth_in"] * 2.54 #Convert data to cm

    return df[["date", "true_snow_depth_cm"]]


#Load in all rows for datasheet
snow_df, measurement_dates = load_truth_table(SNOW_FILE)
sensor_files = sorted(glob.glob(os.path.join(SENSOR_FOLDER, "*.csv")))

if len(sensor_files) == 0:
    raise ValueError(f"No files in {SENSOR_FOLDER}")

all_training_rows = []
all_daily_predictions_input = []

print("Importing:\n")

for sensor_path in sensor_files:
    site_id = extract_site_id_from_filename(sensor_path)

    if site_id is None:
        continue

    print(f"Loading {site_id}")

    truth_df = get_truth_for_site(snow_df, site_id, measurement_dates)
    if truth_df is None:
        continue

    try:
        daily_features = build_daily_features(sensor_path)
    except Exception as e:
        print(f"Skipping {sensor_path}: {e}")
        continue

    train_df = pd.merge(truth_df, daily_features, on="date", how="inner")

    if len(train_df) == 0:
        continue

    train_df["site_id"] = int(site_id)
    train_df["source_file"] = os.path.basename(sensor_path)

    all_training_rows.append(train_df)

    daily_pred_df = daily_features.copy()
    daily_pred_df["site_id"] = int(site_id)
    daily_pred_df["source_file"] = os.path.basename(sensor_path)

    all_daily_predictions_input.append(daily_pred_df)

#Add the 2024 dataset
print("2024 data;")

try:
    ref_features = build_daily_features(REFERENCE_2024_FILE)
    ref_truth = load_2024_reference_data()

    ref_train_df = pd.merge(ref_truth, ref_features, on="date", how="inner")

    if len(ref_train_df) > 0:
        ref_train_df["site_id"] = 9999
        ref_train_df["source_file"] = "2024_reference"
        all_training_rows.append(ref_train_df)
        print(f"Added {len(ref_train_df)} rows from 2024 data")
except Exception as e:
    print(f"Failed to load 2024 data: {e}")

#Train model on random forest using daTa
full_train_df = pd.concat(all_training_rows, ignore_index=True)
full_daily_df = pd.concat(all_daily_predictions_input, ignore_index=True)

drop_cols = ["date", "true_snow_depth_cm", "source_file", "site_id"]

X = full_train_df.drop(columns=drop_cols, errors="ignore")
y = full_train_df["true_snow_depth_cm"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42
    ))
])
#xVal if there is more than 5 rows to train with data given, output mae and r² values
if len(full_train_df) >= 5:
    loo = LeaveOneOut()
    preds = cross_val_predict(model, X, y, cv=loo)
    # Save and quickly show all validation results
    print("\nResults from cross validation")
    print(f"MAE: {mean_absolute_error(y, preds):.2f} cm")
    print(f"R²: {r2_score(y, preds):.3f}")

#Plot
    plt.figure()

    plt.scatter(y, preds)

    # Perfect prediction line
    min_val = min(y.min(), preds.min())
    max_val = max(y.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("Actual Snow Depth (cm)")
    plt.ylabel("Predicted Snow Depth (cm)")
    plt.title("Predicted vs Actual Snow Depth")

    plt.savefig("prediction_vs_actual.png")
    plt.close()

    print("Saved: prediction_vs_actual.png")

#Fit train and save output
model.fit(X, y)

pred_X = full_daily_df.drop(columns=["date", "source_file", "site_id"], errors="ignore")

full_daily_df["predicted_snow_depth_cm"] = model.predict(pred_X)
full_daily_df["predicted_snow_depth_cm"] = full_daily_df["predicted_snow_depth_cm"].clip(lower=0)


output = full_daily_df[["site_id", "date", "source_file", "predicted_snow_depth_cm"]]
output.to_csv("predictedsnowdepthdata.csv", index=False)

print("\nSaved: predictedsnowdepthdata.csv")