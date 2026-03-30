#Capstone, ML implementation
#File function: read data from DNR, read true data -> train data -> output results to new validated file
#Do a ML based on random forest regression algorithm
#done:Convert all time from UTC to UTC-8 Alaska Time

##Important fixes:
#todo:improve algorithm to reduce noticable errors in many spots
#todo:improve r² val
#todo:improve MSE if possible
#todo:dataset has good average but some erronious extremes SEE OUTPUT, determine a way to address this
##Ideas to implement:
#todo: output these as graphs too maybe? would be nice for visualizing if needed

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
#Note: this is the exact columns true measurements were taken in the file provided,
#I think it's structured very poorly but I'm not changing it
ACTUAL_COLS = [4, 6, 8, 10, 12, 14, 16]
def extract_site_id_from_filename(filepath):
#Ignore all letters when importing file but keep the number code when importing files (easy to cross reference if needed)
    filename = os.path.basename(filepath)
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    return None

#done:Load actual snow depth values, each one is done in rows for matching ID in actual_cols
def load_truth_table(snow_file):
    snow_df = pd.read_csv(snow_file, header=None)
    measurement_dates = []
    for c in ACTUAL_COLS:
        dt = pd.to_datetime(snow_df.iloc[0, c], errors="coerce")
        measurement_dates.append(dt.date() if pd.notna(dt) else None)
    return snow_df, measurement_dates
#This is for matching each site with true snow data
def get_truth_for_site(snow_df, site_id, measurement_dates):
    site_rows = snow_df[snow_df[2].astype(str).str.strip() == str(site_id)]
    if site_rows.empty: #Handle empty cases
        return None
    site_row = site_rows.iloc[0]
    true_depths = []
    valid_dates = []
    for c, dt in zip(ACTUAL_COLS, measurement_dates):
        val = pd.to_numeric(site_row[c], errors="coerce")
        if dt is not None and pd.notna(val):
            valid_dates.append(dt)
            true_depths.append(float(val))
    if len(valid_dates) == 0: #Handle nil cases
        return None
    return pd.DataFrame({
        "date": valid_dates,
        "true_snow_depth_cm": true_depths
    })
#Build from all the snow data we get per day, read timestamp, date, create new columns into a feature from CSV
def build_daily_features(sensor_csv_path):
    df = pd.read_csv(sensor_csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' col not in {sensor_csv_path}") #Throw if not found in file

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date
#Sensor temp cols
    sensor_cols = [c for c in df.columns if c.endswith(" m")]
    feature_cols = sensor_cols
    if len(feature_cols) == 0:
        raise ValueError(f"no cols in {sensor_csv_path}") #Throw if not found for whatever reason

#Train temperature: Mean, minimum, and maximum. Merge all for data implementation.
    daily_mean = df.groupby("date")[feature_cols].mean(numeric_only=True)
    daily_mean.columns = [f"{c}_mean" for c in daily_mean.columns]
    daily_min = df.groupby("date")[feature_cols].min(numeric_only=True)
    daily_min.columns = [f"{c}_min" for c in daily_min.columns]
    daily_max = df.groupby("date")[feature_cols].max(numeric_only=True)
    daily_max.columns = [f"{c}_max" for c in daily_max.columns]
    daily = pd.concat([daily_mean, daily_min, daily_max], axis=1).reset_index()
    return daily


#Load snow depth (real)
snow_df, measurement_dates = load_truth_table(SNOW_FILE)
#All the csvs in the folder
sensor_files = sorted(glob.glob(os.path.join(SENSOR_FOLDER, "*.csv")))

if len(sensor_files) == 0:
    raise ValueError(f"why didn't you put anything in {SENSOR_FOLDER}?")


print("Importing DNR files:\n")
for f in sensor_files:
    print("*", os.path.basename(f))

###Next step: train ts or something
###
###
all_training_rows = []
all_daily_predictions_input = []
for sensor_path in sensor_files:
    site_id = extract_site_id_from_filename(sensor_path)

    if site_id is None: #Throw if unable to determine id for whatever reason
        print(f"{sensor_path} Skipped! Reason: Cannot determine ID from filename.")
        continue
    print(f"* Loading ID: {site_id} from file {os.path.basename(sensor_path)}.")
#Truth sensor rows
    truth_df = get_truth_for_site(snow_df, site_id, measurement_dates)
    if truth_df is None: #Throw if unable to determine row with corresponding id
        print(f"{site_id}Skipped! Reason: cannot determine real snow depth row for matching ID.")
        continue
    try: #Build daily temp range function from sensor data
        daily_features = build_daily_features(sensor_path)
    except Exception as e: #Throw if unable for whatever reason
        print(f"{sensor_path} Skipped! Cannot process. {e}")
        continue
#Let feature function know site id, merge dates that have non-nil dates, proceed to train
    daily_features["site_id"] = int(site_id)
    train_df = pd.merge(truth_df, daily_features, on="date", how="inner")
    if len(train_df) == 0:
        print(f"{site_id} Skipped! Reason: Nil values found for matching dates between file and actual depth recorded csv.")
        continue
    train_df["site_id"] = int(site_id)
    train_df["source_file"] = os.path.basename(sensor_path)
    all_training_rows.append(train_df)
#SAVE ts
    daily_pred_df = daily_features.copy()
    daily_pred_df["source_file"] = os.path.basename(sensor_path)
    all_daily_predictions_input.append(daily_pred_df)

#Combine all
if len(all_training_rows) == 0: #throw if nothing can be trained
    raise ValueError("you trained nothing")
full_train_df = pd.concat(all_training_rows, ignore_index=True)
full_daily_df = pd.concat(all_daily_predictions_input, ignore_index=True)
print("\nData imported from DNR files")
print(f"Rows found: {len(full_train_df)}")
print(f"Sensor sites imported: {full_train_df['site_id'].nunique()}")
print(full_train_df[["site_id", "date", "true_snow_depth_cm", "source_file"]].head(20))

drop_cols = ["date", "true_snow_depth_cm", "source_file"] #prepare all columns for training, x,y,train,target with random forest
X = full_train_df.drop(columns=drop_cols, errors="ignore")
y = full_train_df["true_snow_depth_cm"]
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", RandomForestRegressor(
        n_estimators=300, #play with if desired
        max_depth=8,
        random_state=42
    ))
])

#xVal if there is more than 5 rows to train with data given, output mae and r² values
if len(full_train_df) >= 5:
    loo = LeaveOneOut()
    cv_preds = cross_val_predict(model, X, y, cv=loo)
    mae = mean_absolute_error(y, cv_preds)
    r2 = r2_score(y, cv_preds)
#Save and quickly show all validation results
    print("Results from cross validation")
    print(f"MAE: {mae:.2f}cm")
    print(f"R²: {r2:.3f}")
    validation_results = full_train_df[["site_id", "date", "true_snow_depth_cm", "source_file"]].copy()
    validation_results["cv_predicted_cm"] = cv_preds
    validation_results.to_csv("results_from_site_training.csv", index=False)
    print("\nSave successful! Located in root folder: results_from_site_training.csv")
else:
    print("\nSave failed! Reason: not enough input data for rows")

#Fit model to all data, predict snow depth for every day in included dataset, save all data, output back to CSV
model.fit(X, y)
pred_X = full_daily_df.drop(columns=["date", "source_file"], errors="ignore")
full_daily_df["predicted_snow_depth_cm"] = model.predict(pred_X)
full_daily_df["predicted_snow_depth_cm"] = full_daily_df["predicted_snow_depth_cm"].clip(lower=0)
pred_output = full_daily_df[["site_id", "date", "source_file", "predicted_snow_depth_cm"]].copy()
pred_output.to_csv("predictedsnowdepthdata.csv", index=False)

print("\nSave successful! Located in root folder: predictedsnowdepthdata.csv")
