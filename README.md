# -NBA-
本專題透過分析 2000 年至 2023 年進入 NBA 的球員數據，旨在建立一個能精準預測球員長期價值的決策系統。技術核心首先利用多種ˋ一氣學習模型，以前兩年的基礎表現數據為輸入特徵，預測球員生涯的 Win Share 總值，並以 $R^2$ 作為評核模型擬合優度的關鍵指標。在量化數據之外，本研究更進一步結合爬蟲技術抓取球員相關新聞，運用自然語言處理與深度學習對標題進行情感分析，將球員的場外輿論與心理特質轉化為關鍵變數，與 Win Share 預測值共同構成球隊執行選擇權的判斷基準。未來，專題計畫導入大型語言模型（LLM），針對球員的數據表現與新聞動態產出深度的自動化分析報告，提供更具解釋性與前瞻性的建隊策略建議。

## 資料集
自現有kaggle資料庫(https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats) 提取球員數據，並針對2000-2023球員進入聯盟前兩年之平均數據與第三年之win share進行篩選
(https://www.kaggle.com/datasets/yhsarawang/nba-players-first-two-year-average-data)

## 模型訓練與預測
經過多方評比後，選定六大模型進行測試與調參，分別為
1. Random Forest
<details>
<summary>點擊展開查看核心模型程式碼</summary>
  
```python

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================
# 1) 讀取 & 篩選
# =========================
df = pd.read_csv('NBA_Normalized_Final.csv')
bool_cols = [
    'Is_Undrafted',
    'Pos_PG',
    'Pos_C',
    'Pos_SG',
    'Pos_SF',
    'Pos_PF'
]   
for col in bool_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0})
    )

df_g = df[
    (df['season'] > 1)&(df['year_start']>1999)
]

feature_cols=['G','2P%','2PA','3P','3P%','3PA','3PAr','AST','AST%','BLK','BLK%','BPM','DBPM','DRB','DRB%','DWS','FG','FG%','FGA','FT','FT%','FTA','FTr','OBPM','ORB','ORB%','OWS','PER','PF','PTS','Pos_C','Pos_PF','Pos_PG','Pos_SF','Pos_SG','STL','STL%','TOV','TOV%','TRB','TRB%','TS%','USG%','VORP','WS','WS/48','eFG%']
target_col = 'Year4_WS'


data = df_g.dropna(subset=feature_cols + [target_col])

X = data[feature_cols]
y = data[target_col]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 2) 定義評估函數
# =========================
def eval_regression(y_true, y_pred, name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n[{name}]")
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R²  :", r2)
    return rmse, mae, r2

# =========================
# 3) Random Forest + 隨機搜尋調參
# =========================
rf = RandomForestRegressor(random_state=42, n_jobs=-1)



param_dist = {
    
    "max_depth": [3, 4, 5, 6], 
    "min_samples_leaf": [10, 15, 20, 30], 
    "min_samples_split": [20, 30, 50],
    "max_features": ["sqrt", "log2", 0.3], 
    "n_estimators": [300, 500],
    "bootstrap": [True] 
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,                      
    scoring="r2",  
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

search.fit(X_train, y_train)

# =========================
# 4) 輸出最佳參數與 CV 表現
# =========================
print("\nBest Params:")
print(search.best_params_)

best_r2_cv = search.best_score_
print("\nBest CV R^2:", best_r2_cv)

# =========================
# 5) 用最佳模型做 train/test 評估
# =========================
best_model = search.best_estimator_

y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

eval_regression(y_train, y_pred_train, "Best RF - Train")
eval_regression(y_test, y_pred_test, "Best RF - Test")

train_r2 = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_r2 = np.sqrt(mean_squared_error(y_test, y_pred_test))


# =========================
# 6) 特徵重要度
# =========================
importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances.head(10))
```
</details>
  
2. Ridge Regression 
<details>
<summary>點擊展開查看核心模型程式碼</summary>
  
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =========================
# 1. 讀取與清理資料
# =========================
df = pd.read_csv('NBA_Normalized_Final.csv')

bool_cols = ['Is_Undrafted', 'Pos_PG', 'Pos_C', 'Pos_SG', 'Pos_SF', 'Pos_PF']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})

# =========================
# 2. 篩選資料 
# =========================
df_g = df[(df['season'] > 1) & (df['year_start'] > 1999)].copy()

feature_cols = [
    '2P%','2PA','3P','3P%','3PA','3PAr','AST','AST%','BLK','BLK%','BPM','DBPM',
    'DRB','DRB%','DWS','FG','FG%','FGA','FT','FT%','FTA','FTr',
    'OBPM','ORB','ORB%','OWS','PER','PF','PTS','Pos_C','Pos_PF','Pos_PG',
    'Pos_SF','Pos_SG','STL','STL%','TOV','TOV%','TRB','TRB%','TS%','USG%',
    'VORP','WS','WS/48','eFG%'
]
target_col = 'Year3_WS'

# 清洗目標值與特徵值
data = df_g.dropna(subset=[target_col]).copy()
for c in feature_cols:
    data[c] = pd.to_numeric(data[c], errors='coerce')
data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
data = data.dropna(subset=[target_col]).copy()

X = data[feature_cols].copy()
y = data[target_col].copy()

# =========================
# 3. 切分訓練 / 測試
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. RidgeCV
# =========================
alphas = np.logspace(-4, 4, 60)
cv_splits = min(5, len(X_train))

ridge_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RidgeCV(alphas=alphas, cv=cv_splits, scoring='r2')) 
])

ridge_pipe.fit(X_train, y_train)

# 計算各項 R2
train_r2 = r2_score(y_train, ridge_pipe.predict(X_train))
best_cv_r2 = ridge_pipe.named_steps["model"].best_score_
test_r2 = r2_score(y_test, ridge_pipe.predict(X_test))

# 計算其他指標 
y_pred_test = ridge_pipe.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

# =========================
# 5. 輸出結果
# =========================
print("\n" + "="*40)
print("       RIDGE REGRESSION FINAL REPORT")
print("="*40)
print(f"1. Train R²:   {train_r2:.4f}")
print(f"2. Best CV R² :  {best_cv_r2:.4f}")
print(f"3. Test R²:   {test_r2:.4f}")
print("-"*40)
print(f"Chosen Alpha :     {ridge_pipe.named_steps['model'].alpha_:.4f}")
print(f"Test RMSE:                {rmse_test:.4f}")
print(f"Test MAE:                 {mae_test:.4f}")
print("="*40)

# =========================
# 6. 特徵係數排名 
# =========================
coefs = ridge_pipe.named_steps["model"].coef_
coef_df = pd.DataFrame({"feature": feature_cols, "coef": coefs})
coef_df["abs_coef"] = coef_df["coef"].abs()
print(coef_df.sort_values("abs_coef", ascending=False).head(10)[["feature", "coef"]])
```
</details>

3. Lasso Regression 
<details>
<summary>點擊展開查看核心模型程式碼</summary>
  
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =========================
# 1. 讀取與清理資料
# =========================
df = pd.read_csv('NBA_Normalized_Final.csv')

bool_cols = ['Is_Undrafted', 'Pos_PG', 'Pos_C', 'Pos_SG', 'Pos_SF', 'Pos_PF']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
target_year=2023

# =========================
# 2. 篩選資料 (三年級 WS 預測)
# =========================
df_g = df[(df['season'] > 2)&(df['year_start']>1999)&(df['year_start']!=target_year) ].copy()

feature_cols = [
    '2P%','2PA','3P','3P%','3PA','3PAr','AST','AST%','BLK','BLK%','BPM','DBPM',
    'DRB','DRB%','DWS','FG','FG%','FGA','FT','FT%','FTA','FTr',
    'OBPM','ORB','ORB%','OWS','PER','PF','PTS','Pos_C','Pos_PF','Pos_PG',
    'Pos_SF','Pos_SG','STL','STL%','TOV','TOV%','TRB','TRB%','TS%','USG%',
    'VORP','WS','WS/48','eFG%'
]
target_col = 'Year4_WS'

# 清洗目標值與特徵值
data = df_g.dropna(subset=[target_col]).copy()
for c in feature_cols:
    data[c] = pd.to_numeric(data[c], errors='coerce')
data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
data = data.dropna(subset=[target_col]).copy()

X = data[feature_cols].copy()
y = data[target_col].copy()

# =========================
# 3. 切分訓練 / 測試 
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. 建立 Pipeline 與 GridSearchCV
# =========================

alphas = np.logspace(-4, 1, 40)
cv_splits = min(5, len(X_train))

# 建立管線
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("lasso", Lasso(max_iter=50000, random_state=42, tol=1e-4))
])

# 使用 GridSearchCV 來自動尋找最佳 alpha
param_grid = {'lasso__alpha': alphas}
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=cv_splits, 
    scoring='r2', 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# =========================
# 5. 計算各項指標
# =========================
# 取出最佳模型
best_model = grid_search.best_estimator_
best_alpha = grid_search.best_params_['lasso__alpha']
best_cv_r2 = grid_search.best_score_

# 預測並計算 R2
train_r2 = r2_score(y_train, best_model.predict(X_train))
test_r2 = r2_score(y_test, best_model.predict(X_test))

# 計算 RMSE 與 MAE
y_pred_test = best_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

# 提取 Lasso 獨有的「特徵選擇」結果
lasso_coefs = best_model.named_steps["lasso"].coef_
n_total_features = len(feature_cols)
n_retained_features = np.sum(lasso_coefs != 0)

# =========================
# 6. 正式輸出實戰報告
# =========================
print("\n" + "="*45)
print("         LASSO REGRESSION FINAL REPORT")
print("="*45)
print(f"1. Train R² (訓練集得分):   {train_r2:.4f}")
print(f"2. Best CV R² (交叉驗證):  {best_cv_r2:.4f}")
print(f"3. Test R² (實測測試集):   {test_r2:.4f}")
print("-" * 45)
print(f"Chosen Alpha (最佳 λ):     {best_alpha:.4f}")
print(f"Test RMSE:                {rmse_test:.4f}")
print(f"Test MAE:                 {mae_test:.4f}")
print(f"Feature Selection:        保留 {n_retained_features} / {n_total_features} 個特徵")
print("="*45)

# =========================
# 7. 顯示被 Lasso 保留的關鍵特徵
# =========================
coef_df = pd.DataFrame({"feature": feature_cols, "coef": lasso_coefs})
coef_df["abs_coef"] = coef_df["coef"].abs()


active_features = coef_df[coef_df["coef"] != 0].sort_values("abs_coef", ascending=False)

print(active_features[["feature", "coef"]].head(15).to_string(index=False))
```
</details>

4. Elastic Net
<details>
<summary>點擊展開查看核心模型程式碼</summary>
  
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =========================
# 1. 讀取與清理資料
# =========================
df = pd.read_csv('NBA_Normalized_Final_3.csv')

bool_cols = ['Is_Undrafted', 'Pos_PG', 'Pos_C', 'Pos_SG', 'Pos_SF', 'Pos_PF']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})

# =========================
# 2. 篩選資料 (四年級 WS 預測)
# =========================
df_g = df[(df['season'] > 2) & (df['year_start'] > 1999) ].copy()

feature_cols = [
    '2P%','2PA','3P','3P%','3PA','3PAr','AST','AST%','BLK','BLK%','BPM','DBPM',
    'DRB','DRB%','DWS','FG','FG%','FGA','FT','FT%','FTA','FTr',
    'OBPM','ORB','ORB%','OWS','PER','PF','PTS','Pos_C','Pos_PF','Pos_PG',
    'Pos_SF','Pos_SG','STL','STL%','TOV','TOV%','TRB','TRB%','TS%','USG%',
    'VORP','WS','WS/48','eFG%'
]
target_col = 'Year4_WS'

data = df_g.dropna(subset=[target_col]).copy()
for c in feature_cols:
    data[c] = pd.to_numeric(data[c], errors='coerce')
data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
data = data.dropna(subset=[target_col]).copy()

X = data[feature_cols].copy()
y = data[target_col].copy()

# =========================
# 3. 切分訓練 / 測試 
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. 建立 Pipeline 與 GridSearchCV (自動尋找最佳 alpha & l1_ratio)
# =========================
alphas = np.logspace(-4, 1, 30)
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
cv_splits = min(5, len(X_train))

# 建立管線
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("enet", ElasticNet(max_iter=200000, random_state=42, tol=1e-4))
])

# 使用 GridSearchCV
param_grid = {
    'enet__alpha': alphas,
    'enet__l1_ratio': l1_ratios
}

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=cv_splits, 
    scoring='r2', 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# =========================
# 5. 計算各項指標
# =========================
# 取出最佳模型與參數
best_model = grid_search.best_estimator_
best_alpha = grid_search.best_params_['enet__alpha']
best_l1_ratio = grid_search.best_params_['enet__l1_ratio']
best_cv_r2 = grid_search.best_score_

# 預測並計算 R2
train_r2 = r2_score(y_train, best_model.predict(X_train))
test_r2 = r2_score(y_test, best_model.predict(X_test))

# 計算 RMSE 與 MAE
y_pred_test = best_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

# 提取 Elastic Net 獨有的「特徵選擇」結果
enet_coefs = best_model.named_steps["enet"].coef_
n_total_features = len(feature_cols)
n_retained_features = np.sum(enet_coefs != 0)

# =========================
# 6. 正式輸出實戰報告
# =========================
print("\n" + "="*48)
print("       ELASTIC NET REGRESSION FINAL REPORT")
print("="*48)
print(f"1. Train R² (訓練集得分):   {train_r2:.4f}")
print(f"2. Best CV R² (交叉驗證):  {best_cv_r2:.4f}")
print(f"3. Test R² (實測測試集):   {test_r2:.4f}")
print("-" * 48)
print(f"Chosen Alpha (最佳懲罰):   {best_alpha:.4f}")
print(f"Chosen L1 Ratio (L1比例): {best_l1_ratio:.4f}")
print(f"Test RMSE:                {rmse_test:.4f}")
print(f"Test MAE:                 {mae_test:.4f}")
print(f"Feature Selection:        保留 {n_retained_features} / {n_total_features} 個特徵")
print("="*48)

# =========================
# 7. 顯示被 Elastic Net 保留的關鍵特徵
# =========================
coef_df = pd.DataFrame({"feature": feature_cols, "coef": enet_coefs})
coef_df["abs_coef"] = coef_df["coef"].abs()

# 只列出係數不為 0 的特徵
active_features = coef_df[coef_df["coef"] != 0].sort_values("abs_coef", ascending=False)

print(active_features[["feature", "coef"]].head(15).to_string(index=False))
```
</details>

5. Gradient Boost 
<details>
<summary>點擊展開查看核心模型程式碼</summary>
  
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold # 💡 匯入 RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------
# 1) 讀資料 & 篩選
# -----------------------
df = pd.read_csv('NBA_Normalized_Final.csv')
bool_cols = ['Is_Undrafted', 'Pos_PG', 'Pos_C', 'Pos_SG', 'Pos_SF', 'Pos_PF']

for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})

df_g = df[(df['season'] > 2) & (df['year_start'] > 1999)].copy()

feature_cols=['2P%','2PA','3P','3P%','3PA','3PAr','AST','AST%','BLK','BLK%','BPM','DBPM','DRB','DRB%','DWS','FG','FG%','FGA','FT','FT%','FTA','FTr','OBPM','ORB','ORB%','OWS','PER','PF','PTS','Pos_C','Pos_PF','Pos_PG','Pos_SF','Pos_SG','STL','STL%','TOV','TOV%','TRB','TRB%','TS%','USG%','VORP','WS','WS/48','eFG%']
target_col = 'Year4_WS'

data = df_g.dropna(subset=feature_cols + [target_col])
X = data[feature_cols]
y = data[target_col]

# -----------------------
# 2) 切 train/test 
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# 3) 建立模型 + RandomizedSearch
# -----------------------
gbr = GradientBoostingRegressor(random_state=42)


param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4],              
    "min_samples_leaf": [1, 2, 4, 8],
    "subsample": [0.6, 0.8, 1.0]         
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)


grid = RandomizedSearchCV(
    estimator=gbr,
    param_distributions=param_dist, 
    n_iter=60,                     
    scoring="r2",                  
    cv=cv,
    n_jobs=-1,                     
    verbose=1,
    random_state=42,               
    return_train_score=True
)

grid.fit(X_train, y_train)

print("\nBest Params:", grid.best_params_)
print("Best CV R²:", grid.best_score_)

best_model = grid.best_estimator_

# -----------------------
# 4) 評估（Train/Test）
# -----------------------
y_train_pred = best_model.predict(X_train)
y_test_pred  = best_model.predict(X_test)

def report(y_true, y_pred, name=""):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n[{name}]")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    return r2

r2_train = report(y_train, y_train_pred, "GBR - Train")
r2_test  = report(y_test,  y_test_pred,  "GBR - Test")
print(f"\nOverfitting Gap (Train - Test R²): {r2_train - r2_test:.4f}")
```
</details>

6. PLS Regression 
<details>
<summary>點擊展開查看核心模型程式碼</summary>
  
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

# =========================
# 1.讀取與準備資料
# =========================
df = pd.read_csv('NBA_Normalized_Final.csv')

# 布林欄位處理
bool_cols = ['Is_Undrafted', 'Pos_PG', 'Pos_C', 'Pos_SG', 'Pos_SF', 'Pos_PF']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})

# 篩選資料 (三年級 WS 預測)
df_g = df[(df['season'] > 1) & (df['year_start'] > 1999)].copy()

feature_cols = [
    '2P%','2PA','3P','3P%','3PA','3PAr','AST','AST%','BLK','BLK%','BPM','DBPM',
    'DRB','DRB%','DWS','FG','FG%','FGA','FT','FT%','FTA','FTr',
    'OBPM','ORB','ORB%','OWS','PER','PF','PTS','Pos_C','Pos_PF','Pos_PG',
    'Pos_SF','Pos_SG','STL','STL%','TOV','TOV%','TRB','TRB%','TS%','USG%',
    'VORP','WS','WS/48','eFG%'
]
target_col = 'Year3_WS'

# 清洗目標值 
data = df_g.dropna(subset=[target_col]).copy()
for c in feature_cols:
    data[c] = pd.to_numeric(data[c], errors='coerce')
data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
data = data.dropna(subset=[target_col]).copy()

X = data[feature_cols].copy()
y = data[target_col].copy()

# =========================
# 2.切分 train/test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3.建立 Pipeline 與 GridSearchCV 
# =========================
param_grid = {'pls__n_components': range(1, 21)}
cv_splits = min(5, len(X_train))

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", StandardScaler()),                  
    ("pls", PLSRegression())
])

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=cv_splits, 
    scoring='r2', 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 取出最佳模型
best_model = grid_search.best_estimator_
best_n_components = grid_search.best_params_['pls__n_components']
best_cv_r2 = grid_search.best_score_

# =========================
# 4.預測 
# =========================
y_train_pred = best_model.predict(X_train).ravel()
y_test_pred = best_model.predict(X_test).ravel()

# =========================
# 5.計算基本誤差與 R2
# =========================
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test  = np.sqrt(mean_squared_error(y_test, y_test_pred))

mae_test = mean_absolute_error(y_test, y_test_pred)
medae_test = median_absolute_error(y_test, y_test_pred)

# =========================
# 6.CV RMSE 
# =========================
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    best_model, X_train, y_train,
    scoring="neg_root_mean_squared_error",
    cv=cv, n_jobs=-1
)
cv_mean = -cv_scores.mean()
cv_std  = cv_scores.std(ddof=1)

# =========================
# 7.Train–Test Gap
# =========================
gap = rmse_test - rmse_train

# =========================
# 8.Spearman rank correlation
# =========================
def spearman_corr(y_true, y_pred):
    rt = np.argsort(np.argsort(y_true))
    rp = np.argsort(np.argsort(y_pred))
    return np.corrcoef(rt, rp)[0, 1]

spearman = spearman_corr(y_test, y_test_pred)

# =========================
# 9.輸出報告
# =========================
print("\n" + "="*45)
print("          PLS REGRESSION EVALUATION")
print("="*45)
print(f"Chosen n_components:      {best_n_components}")
print(f"Train R²:                 {train_r2:.6f}")
print(f"Best CV R² (train_5fold): {best_cv_r2:.6f}")
print(f"Test R²:                  {test_r2:.6f}")
print("-" * 45)
print(f"RMSE (test):              {rmse_test:.6f}")
print(f"MAE  (test):              {mae_test:.6f}")
print(f"MedianAE (test):          {medae_test:.6f}")
print(f"CV RMSE (train, 5-fold):  {cv_mean:.6f} ± {cv_std:.6f}")
print(f"Train–Test Gap:           {gap:.6f}")
print(f"Spearman rank corr (test):{spearman:.6f}")
print("="*45)
``` 
</details>


### 模型成效對比 (Model Performance Comparison)

| 模型 (Model) | BEST CV $R^2$ | TEST $R^2$ | RMSE | MAE | MedianAE | CV RMSE | Train-Test Gap | Spearman |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boost** | 0.5718 | 0.5626 | 1.6916 | 1.2571 | 0.9955 | 1.7642 ± 0.1842 | 0.1775 | 0.6526 |
| **Random Forest** | 0.5570 | 0.5280 | 1.7572 | 1.2734 | 0.9104 | 1.8013 ± 0.1442 | 0.2411 | 0.6620 |
| **Ridge** | 0.5530 | 0.5910 | 1.5696 | 1.2149 | 0.9149 | 1.7825 ± 0.1282 | -0.1357 | 0.6859 |
| **Elastic Net** | 0.5562 | 0.5788 | 1.5487 | 1.2079 | 0.9094 | 1.7816 ± 0.1342 | -0.2098 | 0.7167 |
| **Lasso** | 0.5562 | 0.5756 | 1.5411 | 1.2012 | 0.0000 | 1.7783 ± 0.1349 | -0.2112 | 0.7209 |
| **PLS** | 0.5486 | 0.5816 | 1.5436 | 1.1928 | 0.8728 | 1.7875 ± 0.1217 | -0.1852 | 0.7079 |

**1. 預測精準度與誤差分析 (Precision & Error Metrics)**:
雖然各模型在交叉驗證中的 $R^2$ 表現相近（皆落在 0.54 - 0.57 之間），但在代表盲測精準度的測試集誤差指標上，Lasso Regression 取得了 1.5411 的最低 RMSE（均方根誤差）。較低的 RMSE 與 MAE（1.2012）意味著 Lasso 在預測真實生涯 Win Share 數值時，整體的平均偏離幅度最小。在需要精確量化球員合約價值的商業場景中，誤差最小的模型具備最高的實用價值。

**2. 泛化能力與過擬合控制 (Generalization & Stability)**:
從長期預測的穩定度來看，Lasso 與 Elastic Net 在交叉驗證中的表現極為穩健（CV RMSE 為 1.7783 ± 0.1349），標準差控制在極低範圍，證明模型在面對不同世代的球員樣本時，具備優異的泛化能力。此外，雖然非線性的樹模型（如 Random Forest）看似強大，但其 Train-Test Gap 偏高（0.2411），存在較高的過擬合風險；反之，Lasso 的泛化表現更為穩定。

$\Rightarrow$ 本專案最終選定 Lasso Regression 作為球隊合約決策矩陣的核心預測引擎

## Win Share預測及球員報酬計算
為使Win Share能真實反映球員帶來之報酬，並結合實質選擇權概念計算球員現價，同時消弭不同年代貨幣價值波動帶來之預測誤差，本研究以以下方法進行球員價值預測：

#### 一、 球員實質價值與理論薪資估算 (Valuation Phase)
系統首先量化球員在聯盟第三年的預期邊際貢獻，將其場上表現轉化為理論薪資佔比：

1. **生涯表現預測**：採用最優之機器學習模型（Lasso），預測球員關鍵第三年的預期勝場貢獻度 $\widehat{WS}$。
2. **邊際價值計算**：結合該年度的薪資上限（Salary Cap）與籃球相關收入因子（BRI Factor），計算每單位 Win Share 的市場邊際價值量化公式如下：
   
   $$Marginal\ Value\ per\ WS = \frac{Salary\ Cap \times Team\ Count \times BRI\ Factor}{Total\ Games}$$

3. **理論薪資佔比估算**：依據上述邊際價值，推導出該球員在健康市場機制下應得的「理論薪資佔比（*Pred_Salary_Share_Percent*）」：
   
  $$Pred\ Salary\ Share\ Percent = \left[ \frac{\widehat{WS} \times Marginal\ Value\ per\ WS}{Salary\ Cap} \right] \times 100\%$$

#### 二、 定額合約成本基準 (Cost Benchmark Phase)

依據 NBA 勞資協議（CBA）之規定，首輪新秀合約前幾年的薪資與選秀順位高度相關且受到嚴格規範。系統會根據該球員的選秀順位，自動導入對應之新秀標準薪資，並計算出其佔當年度薪資上限的「合約成本佔比（*Est_Cost_Percent*）」，作為決策對比的硬性基準線。

#### 三、 動態決策矩陣與誤差優化 (Decision & Optimization Phase)

在實務決策中，單純依據數值高低進行二分法判定（硬閾值決策）容易因統計噪聲而失真。因此，本系統設計了動態決策緩衝機制：

1. **基礎決策邏輯**：
   * 當*Pred_Salary_Share_Percent* $$\ge$$ *Est_Cost_Percent* 時 $\rightarrow$ **執行選擇權（Exercise Option）**。
   * 當 *Pred_Salary_Share_Percent* $$<$$ *Est_Cost_Percent*時 $\rightarrow$ **判定不執行（Decline Option）**。

2. **基於統計誤差（RMSE）的決策優化**：
   在實測中，單純的硬閾值判定會導致部分邊緣球員的預測結果出現不合理偏差。為了提升模型的容錯率與實務應用價值，本系統引入了**模型均方根誤差（RMSE）作為動態緩衝範圍**。
   * **優化機制**：若系統初步判定為「不執行」，但兩者間的價值差距小於最優模型的統計誤差範圍時，即：
     
     |*Est_Cost_Percent - Pred_Salary_Share_Percent*| $$\le RMSE$$
     
   * **決策修正**：系統會自動將該球員從「不執行」修正移入「球團重點觀察名單（Watchlist）」，並結合次階段的新聞情感分析（NLP Sentiment Score）進行質化調校，避免錯估具備潛在爆發力或場外正面效應的邊緣球員。
  



