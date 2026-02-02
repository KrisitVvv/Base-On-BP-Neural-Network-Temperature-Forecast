import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import argparse
import os
import warnings
import time
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

warnings.filterwarnings('ignore', category=ConvergenceWarning)

def BP(predict_data, df, columns, daily, target_type, pbar_global):
    day = columns[daily]
    avg = df[day].mean()
    std = df[day].std()
    start_year = 2011
    end_year = 2024
    target_date = datetime.strptime(day, '%m-%d').date()
    target_year = 2025
    data = df.loc[df['年份'].between(start_year, end_year)]

    X_raw = data[['年份']].values
    y_raw = data[target_date.strftime('%m-%d')].values

    mask = ~np.isnan(y_raw)
    X = X_raw[mask]
    y = y_raw[mask]
    
    if len(X) == 0 or len(y) == 0:
        print(f'No valid data for {day}, using historical average.')
        prediction = avg if not np.isnan(avg) else 0.0
        r2 = 0.0 
    elif len(X) < 2:
        print(f'Insufficient data for {day} ({len(X)} data point(s)), using last known value.')
        prediction = y[0] if not np.isnan(y[0]) else avg
        r2 = 0.0
    else:
        min_valid_temp = avg - 3 * std if not np.isnan(std) else avg - 30
        max_valid_temp = avg + 3 * std if not np.isnan(std) else avg + 30
        min_r2_threshold = 0.5
        def predict_and_evaluate(random_seed=None):
            split_idx = max(1, int(len(X) * 0.8))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            if len(X_train) == 0 or len(y_train) == 0:
                X_train, X_val = X[split_idx:], X[:split_idx]
                y_train, y_val = y[split_idx:], y[:split_idx]
            if len(X_train) == 0 or len(y_train) == 0:
                X_train, y_train = X, y
                X_val, y_val = X, y
            
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_val_scaled = scaler_X.transform(X_val)
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=10000,
                random_state=random_seed if random_seed is not None else np.random.randint(0, 10000),
                early_stopping=True,
                validation_fraction=0.2,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08,
                n_iter_no_change=100,
                tol=1e-4
            )
            
            model.fit(X_train_scaled, y_train)
            X_val_scaled_for_pred = scaler_X.transform(X_val)
            y_pred = model.predict(X_val_scaled_for_pred)
            if len(y_val) > 0 and len(y_pred) > 0:
                r2 = r2_score(y_val, y_pred)
            else:
                r2 = 0.0
            prediction_year = [[target_year]]
            prediction_year_scaled = scaler_X.transform(prediction_year)
            prediction = model.predict(prediction_year_scaled)[0]
            
            return prediction, r2

        best_prediction = None
        best_r2 = float('-inf')
        max_attempts = 50 
        retry_count = 0 
        total_attempts = 0 
        while best_r2 < 0 and retry_count < 2: 
            for attempt in range(max_attempts):
                total_attempts += 1
                prediction, r2 = predict_and_evaluate()
                if r2 > best_r2:
                    best_r2 = r2
                    best_prediction = prediction
                if r2 >= min_r2_threshold:
                    break
            if best_r2 < 0:
                retry_count += 1
                best_prediction = None
                best_r2 = float('-inf')
                pbar_global.set_postfix({
                    'Date': day,
                    'Type': target_type,
                    'R²': f'{best_r2:.3f}',
                    'Status': f'Retrying {retry_count}/2 (Best R² < 0)'
                })
            else:
                break

        if best_r2 < 0:
            print(f'Warning: After {retry_count} retries, best R² is still {best_r2:.3f} for {day}')
        prediction = best_prediction
        r2 = best_r2
        
        pbar_global.set_postfix({
            'Date': day,
            'Type': target_type,
            'R²': f'{r2:.3f}',
        })
        if prediction is not None and min_valid_temp is not None and max_valid_temp is not None:
            if prediction < min_valid_temp or prediction > max_valid_temp:
                print(f'Prediction for {day} is out of reasonable range: {prediction:.3f}, Range: [{min_valid_temp:.3f}, {max_valid_temp:.3f}]')
                prediction = avg

    predict = {
        '日期': day,
        f'预测{target_type}温度': prediction,
    }
    predict_data.append(predict)
    
    return prediction

parser = argparse.ArgumentParser(description='BP Network Forecast Temperature')
parser.add_argument('-r', '--read-path', type=str, default='./reptile/output/pivot', 
                    help='read file path，default:./reptile/output/pivot')
args = parser.parse_args()

forecast_dir = './forecast'
os.makedirs(forecast_dir, exist_ok=True)

read_path = args.read_path
max_temp_path = os.path.join(read_path, 'MaxTemperature.xlsx')
min_temp_path = os.path.join(read_path, 'MinTemperature.xlsx')

print("Begin forecasting temperature...")
df_max = pd.read_excel(max_temp_path)
columns = df_max.columns
predict_data_max = []

df_min = pd.read_excel(min_temp_path)
columns = df_min.columns
predict_data_min = []
total_days = 365 + 365
with tqdm(total=total_days, desc="Overall Progress") as pbar_global:
    for daily in range(1, 335):
        BP(predict_data_max, df_max, columns, daily, "max", pbar_global)
        pbar_global.update(1)
        
    for daily in range(1, 335):
        BP(predict_data_min, df_min, columns, daily, "min", pbar_global)
        pbar_global.update(1)

new_df_max = pd.DataFrame(predict_data_max)
forecast_max_path = os.path.join(forecast_dir, 'forecast_max.xlsx')
new_df_max.to_excel(forecast_max_path, index=False)
print(f'Forecast Max_Temperature save as {forecast_max_path}')

new_df_min = pd.DataFrame(predict_data_min)
forecast_min_path = os.path.join(forecast_dir, 'forecast_min.xlsx')
new_df_min.to_excel(forecast_min_path, index=False)
print(f'Forecast Min_Temperature save as {forecast_min_path}')

df1 = pd.read_excel(forecast_max_path)
df2 = pd.read_excel(forecast_min_path)
result = pd.concat([df1, df2])
forecast_path = os.path.join(forecast_dir, 'forecast.xlsx')
result.to_excel(forecast_path, index=False)
print(f'Final Forecast Temperature save as {forecast_path}')