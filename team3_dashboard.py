import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN
import warnings
from pandas.api.types import CategoricalDtype
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.exceptions import UndefinedMetricWarning
from collections import Counter
import shap
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from scipy.stats import f, norm
import time
import koreanize_matplotlib
from streamlit_option_menu import option_menu
import matplotlib.patches as patches
import pickle 
import json

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

st.set_page_config(page_title="반도체 공정 모니터링 대시보드", layout="wide")
plt.switch_backend("Agg")
np.random.seed(42)

def create_summarized_beeswarm_payload(shap_explanation, target_class_name, max_display=10, num_bins=4):
    """
    SHAP 데이터를 '구간별 통계'로 요약하여 LLM을 위한 경량화된 payload를 생성합니다.

    Args:
        shap_explanation (shap.Explanation): SHAP 설명 객체 (sv_c).
        target_class_name (str): 설명 대상 클래스 이름 (fault_choice).
        max_display (int): 포함할 상위 특성의 수.
        num_bins (int): 특성 값을 나눌 구간(bin)의 수. (예: 4 -> 4분위수)
    
    Returns:
        dict: LLM에 전달할 작고 효율적인 JSON payload.
    """
    shap_values = shap_explanation.values
    feature_values = shap_explanation.data
    feature_names = shap_explanation.feature_names
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:max_display]

    feature_details_summary = []
    for i in top_indices:
        # 1. 해당 특성의 값과 SHAP 값을 DataFrame으로 만듭니다.
        df = pd.DataFrame({
            'feature_value': feature_values[:, i],
            'shap_value': shap_values[:, i]
        })

        # 2. feature_value를 기준으로 데이터를 num_bins개의 분위수(quantile)로 나눕니다.
        # qcut은 각 구간에 동일한 수의 데이터가 들어가도록 나눕니다.
        try:
            df['bin'] = pd.qcut(df['feature_value'], q=num_bins, labels=False, duplicates='drop')
            bin_labels = [f"Quantile {q+1}/{num_bins}" for q in range(df['bin'].nunique())]
            df['bin'] = pd.qcut(df['feature_value'], q=num_bins, labels=bin_labels, duplicates='drop')

        except ValueError: # 모든 값이 동일하여 구간을 나눌 수 없는 경우
            df['bin'] = 'Single Value'

        # 3. 각 bin별로 통계치를 계산합니다.
        # shap_value의 평균/중간값, feature_value의 범위, 데이터 개수(count)
        summary = df.groupby('bin').agg(
            shap_mean=('shap_value', 'mean'),
            shap_median=('shap_value', 'median'),
            feature_min=('feature_value', 'min'),
            feature_max=('feature_value', 'max'),
            sample_count=('shap_value', 'size')
        ).reset_index()

        # 4. JSON으로 변환하기 좋게 레코드 형태로 변환합니다.
        bin_summaries = summary.to_dict(orient='records')
        
        # 소수점 정리
        for record in bin_summaries:
            for key, val in record.items():
                if isinstance(val, float):
                    record[key] = round(val, 4)

        feature_info = {
            "feature_name": feature_names[i],
            "mean_abs_shap_value": round(float(mean_abs_shap[i]), 4),
            "value_shap_summary": bin_summaries # Raw data 대신 요약 정보 전달
        }
        feature_details_summary.append(feature_info)
        
    payload = {
        "plot_type": "SHAP Beeswarm Summary (Quantized)",
        "explanation_scope": "Global",
        "target_class_name": target_class_name,
        "top_features_summary": feature_details_summary
    }
    
    return payload





# --- 폰트 설정: 한글 표시 (MSPC 그래프용) ---
# Streamlit 환경에 따라 폰트 설정이 다를 수 있습니다.
# 해당 환경에 맞는 한글 폰트를 설정하거나 기본 폰트를 사용하세요.
#plt.rcParams['font.family'] = 'AppleGothic'
#plt.rcParams['axes.unicode_minus'] = False
# ------------------------------------------


# ===============================
# 공통 유틸: CSV 로더 & 컬럼 정규화
# ===============================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = pd.Index([str(c) for c in df.columns])
    cols = cols.str.replace('\ufeff', '', regex=False).str.strip()
    lower = cols.str.lower()
    mapping = {}
    if 'time' not in cols:
        aliases = ['time','time(s)','timestamp','timesec','sec','seconds']
        for a in aliases:
            if a in lower:
                idx = list(lower).index(a); mapping[cols[idx]] = 'Time'; break
    if 'Step Number' not in cols:
        aliases = ['step number','step_number','step','process step','stepnum']
        for a in aliases:
            if a in lower:
                idx = list(lower).index(a); mapping[cols[idx]] = 'Step Number'; break
    df = df.rename(columns=mapping)
    df.columns = [str(c).replace('\ufeff','').strip() for c in df.columns]
    return df

def robust_read_csv(path: Path) -> pd.DataFrame:
    for enc in ['utf-8-sig','utf-8','cp949']:
        try:
            df = pd.read_csv(path, encoding=enc)
            df = normalize_columns(df)
            if df.shape[1] == 1:
                for sep in [';','\t','|']:
                    try:
                        df2 = pd.read_csv(path, encoding=enc, sep=sep, engine='python')
                        df2 = normalize_columns(df2)
                        if df2.shape[1] > 1:
                            return df2
                    except Exception:
                        pass
            return df
        except Exception:
            continue
    df = pd.read_csv(path)
    return normalize_columns(df)

def find_existing_path(primary: str) -> Path:
    candidates = [Path(primary), Path.cwd()/primary, Path.cwd().parent/primary]
    for p in candidates:
        if p.exists():
            return p
    return Path(primary)

# ===============================
# (엔드포인트용) 데이터 로드
# ===============================
@st.cache_data(show_spinner=False)
def load_data(file_path=None):
    try:
        if file_path:
            p = find_existing_path(file_path)
            df = robust_read_csv(p)
        msg = ""
    except FileNotFoundError:
        msg = "🚨 데이터 파일을 찾을 수 없습니다. 예시 데이터를 생성합니다."
        np.random.seed(42)
        num_time_windows = 35
        # 예시 데이터 생성 로직...
        wafers = [f'29{i:02d}' for i in range(1, 30)]
        df = pd.DataFrame({
            '시간': pd.date_range("2023-01-01", periods=num_time_windows, freq="T"),
            '온도(℃)': np.random.normal(200, 5, num_time_windows),
            '압력(Pa)': np.random.normal(50, 2, num_time_windows),
            'Time': np.arange(num_time_windows),
            'Step Number': np.concatenate([np.repeat(4, 15), np.repeat(5, num_time_windows - 15)])
        })
    return df, msg

# ===============================
# (엔드포인트)) 데이터/피처/학습
# ===============================
@st.cache_data(show_spinner=False)
def load_endpoint_data():
    df_ep, ep_msg = load_data(file_path='final_merged_data4.csv')
    return df_ep, ep_msg

# ===============================
# (결함분류) 데이터/피처/학습
# ===============================
@st.cache_data(show_spinner=False)
def load_fault_data():
    try:
        p = find_existing_path("final_data_with_rfm3.csv")
        df_merged = robust_read_csv(p)
        msg = ""
    except FileNotFoundError:
        msg = "🚨 'final_data_with_rfm3.csv' 파일을 찾을 수 없습니다. 예시 데이터를 생성합니다."
        np.random.seed(42)
        num_time_windows = 35
        wafers_29 = [f'29{i:02d}' for i in range(1, 30)]
        wafers_31 = [f'31{i:02d}' for i in range(1, 44)]
        wafers_33 = [f'33{i:02d}' for i in range(1, 44)]
        wafers = wafers_29 + wafers_31 + wafers_33
        fault_mapping = {1:'calibration',15:'TCP +50',16:'RF -12',17:'BCl3 +10',18:'BCl3 +5',19:'BCl3 -5',
                         20:'Cl2 +5',21:'Cl2 -10',22:'Cl2 -5',23:'He Chuck',24:'Pr +1',25:'Pr +2',
                         26:'Pr +3',27:'Pr -2',28:'RF +10',29:'RF +8',30:'TCP +10',31:'TCP +20',
                         32:'TCP +30',33:'TCP -15',34:'TCP -20'}
        wafer_fault_map_33 = {f'33{i:02d}': fault_mapping.get(i,'calibration') for i in range(1,44)}
        df_merged = pd.DataFrame({
            'wafer_names': np.repeat(wafers, num_time_windows),
            'time_window': np.tile(range(num_time_windows), len(wafers)),
            'OES_value_1': np.random.rand(len(wafers)*num_time_windows)*100,
            'OES_value_2': np.random.rand(len(wafers)*num_time_windows)*50,
            'RFM_value_1': np.random.rand(len(wafers)*num_time_windows)*10,
            'RFM_value_2': np.random.rand(len(wafers)*num_time_windows)*20,
            'Pressure': np.random.rand(len(wafers)*num_time_windows)*5,
        })
        df_merged['wafer_names'] = df_merged['wafer_names'].astype(str)
        df_merged['fault_name'] = df_merged['wafer_names'].map(lambda x: wafer_fault_map_33.get(x,'calibration') if x.startswith("33") else 'calibration')
    df_merged['wafer_group'] = df_merged['wafer_names'].astype(str).str[:2]
    return df_merged, msg

@st.cache_data(show_spinner=False)
def make_features_by_group(df_merged: pd.DataFrame, group_key: str):
    df_grp = df_merged.copy() if group_key=="전체" else df_merged[df_merged['wafer_group']==group_key].copy()
    numeric_cols = [col for col in df_grp.select_dtypes(include=np.number).columns.tolist() if col not in ['wafer_names', 'time_window', 'target']]
    for col in numeric_cols:
        df_grp[f'{col}_diff1'] = df_grp.groupby('wafer_names')[col].diff(1)
        df_grp[f'{col}_rolling_mean3'] = df_grp.groupby('wafer_names')[col].rolling(window=3).mean().reset_index(level=0, drop=True)
        df_grp[f'{col}_rolling_std3'] = df_grp.groupby('wafer_names')[col].rolling(window=3).std().reset_index(level=0, drop=True)
        df_grp[f'{col}_rolling_mean5'] = df_grp.groupby('wafer_names')[col].rolling(window=5).mean().reset_index(level=0, drop=True)
        df_grp[f'{col}_rolling_std5'] = df_grp.groupby('wafer_names')[col].rolling(window=5).std().reset_index(level=0, drop=True)
    df_grp.fillna(0, inplace=True)
    return df_grp

@st.cache_resource(show_spinner=True)
def train_and_explain(df: pd.DataFrame):
    unique_faults = sorted(df['fault_name'].unique())
    fault_map = {name: code for code, name in enumerate(unique_faults)}
    df_local = df.copy()
    df_local['target'] = df_local['fault_name'].map(fault_map)
    target_names = list(fault_map.keys())
    drop_cols = ['fault_name', 'wafer_names', 'wafer_group', 'target']
    if 'wafer_group' not in df_local.columns:
        drop_cols.remove('wafer_group')
    X = df_local.drop(columns=drop_cols)
    y = df_local['target'].values
    feature_names = X.columns.tolist()
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
    imputation_stats = X_train.median()
    X_train_proc = X_train.fillna(imputation_stats)
    X_test_proc = X_test.fillna(imputation_stats)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_proc)
    X_test_scaled = scaler.transform(X_test_proc)
    try:
        if len(np.unique(y_train)) > 1:
            X_train_resampled, y_train_resampled = ADASYN(random_state=42).fit_resample(X_train_scaled, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train
    except Exception:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train
    clf = LGBMClassifier(
        objective='multiclass',
        num_class=len(unique_faults),
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        verbosity=-1
    )
    clf.fit(X_train_resampled, y_train_resampled)
    pred_label = clf.predict(X_test_scaled)
    report = classification_report(y_test, pred_label, 
                                   labels=list(range(len(unique_faults))),
                                   target_names=target_names, 
                                   zero_division=0, 
                                   output_dict=True)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    bg_size = min(100, len(X_train_scaled_df))
    try:
        bg = shap.kmeans(X_train_scaled_df, bg_size, random_state=42)
    except Exception:
        bg = X_train_scaled_df.sample(bg_size, random_state=42)
    explainer = shap.TreeExplainer(clf, data=bg, feature_perturbation="interventional", model_output="raw")
    sv = explainer(X_test_scaled_df, check_additivity=False)
    return {
        "clf": clf, "pred_label": pred_label, "y_test": y_test, "report": report,
        "sv": sv, "X_test_scaled_df": X_test_scaled_df, "feature_names": feature_names,
        "target_names": target_names
    }

# ===============================
# ▶▶ 엔드포인트 예측 모델 (CatBoost) — 참고 스크립트와 동일 구간 사용
# ===============================
WAFER_KEY_FALLBACK = "_wafer"
SMOOTH_WIN = 5
EPS = 1e-6
LAGS = 5

def is_numeric_col(s: pd.Series, thr: float=0.9) -> bool:
    return pd.to_numeric(s, errors="coerce").notna().mean() >= thr

def smooth_series(a: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(a) < 2: return a
    return pd.Series(a).rolling(win, min_periods=1, center=True).mean().to_numpy()

def first_one(arr: np.ndarray):
    pos = np.where(arr==1)[0]
    return int(pos[0]) if len(pos)>0 else None

@st.cache_resource(show_spinner=True)
def build_endpoint_model(df_raw: pd.DataFrame):
    df_raw = normalize_columns(df_raw.copy())
    wafer_key = "wafer_names" if "wafer_names" in df_raw.columns else WAFER_KEY_FALLBACK
    if wafer_key == WAFER_KEY_FALLBACK and WAFER_KEY_FALLBACK not in df_raw.columns:
        df_raw[WAFER_KEY_FALLBACK] = "W0"
    if "Time" not in df_raw.columns: raise ValueError("final_merged_data4.csv에 'Time' 컬럼이 필요합니다(초 단위).")
    if "Step Number" not in df_raw.columns: raise ValueError("정답 라벨 생성을 위해 'Step Number' 컬럼이 필요합니다.")
    EXCLUDE = {"Step Number","wafer_names","time_window","Time"}
    feature_cols = [c for c in df_raw.columns if c not in EXCLUDE and is_numeric_col(df_raw[c])]
    if len(feature_cols) < 5: raise ValueError("사용 가능한 수치형 피처가 너무 적습니다. 파일 스키마를 확인하세요.")
    df2_list=[]
    for w, sub in df_raw.groupby(wafer_key, sort=False):
        tmp=sub.copy()
        for c in feature_cols:
            s = pd.to_numeric(tmp[c], errors="coerce").interpolate(limit_direction="both").to_numpy()
            s = smooth_series(s, SMOOTH_WIN); tmp[c]=s
        df2_list.append(tmp)
    df2 = pd.concat(df2_list, axis=0, ignore_index=True)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(df2[feature_cols].to_numpy(dtype=float))
    y_true_all=[]
    for w, sub in df2.groupby(wafer_key, sort=False):
        steps = sub["Step Number"].tolist(); ep=None
        for i in range(1,len(steps)):
            if steps[i-1]==4 and steps[i]==5: ep=i; break
        if ep is None: y_true_all.extend([0]*len(steps))
        else:          y_true_all.extend([0]*ep + [1]*(len(steps)-ep))
    y_true = np.asarray(y_true_all, dtype=int)
    rows, ys, times, wids = [], [], [], []
    F = X_all.shape[1]
    for w, sub in df2.groupby(wafer_key, sort=False):
        idx = sub.index.to_numpy()
        Xi = X_all[idx,:]; yi = y_true[idx]; ti = sub["Time"].to_numpy()
        Tw = len(idx)
        if Tw < LAGS: continue
        for t in range(LAGS-1, Tw):
            win = Xi[t-LAGS+1:t+1,:].reshape(-1)
            rows.append(win); ys.append(yi[t]); times.append(ti[t]); wids.append(w)
    X_super = np.vstack(rows) if rows else np.zeros((0, LAGS*F))
    y_super = np.asarray(ys, dtype=int)
    time_super = np.asarray(times)
    wid_super = np.asarray(wids)
    (X_train,X_test,
     y_train,y_test,
     t_train,t_test,
     w_train,w_test) = train_test_split(
        X_super, y_super, time_super, wid_super,
        test_size=0.2, stratify=y_super, random_state=42
    )
    cb = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1,
                            loss_function='Logloss', auto_class_weights='Balanced',
                            random_seed=42, verbose=0)
    cb.fit(X_train, y_train)
    y_pred = cb.predict(X_test).ravel().astype(int)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0, output_dict=True)
    accuracy = report.get("accuracy", float(np.mean(y_test == y_pred)))
    cls1 = report.get("1", {"precision":0, "recall":0, "f1-score":0})
    explainer = shap.TreeExplainer(cb)
    sv_raw = explainer.shap_values(X_test)
    sv_arr = sv_raw[1] if isinstance(sv_raw, list) and len(sv_raw)>1 else (sv_raw[0] if isinstance(sv_raw,list) else sv_raw)
    feat_names=[]
    for lag in range(LAGS):
        for c in feature_cols:
            feat_names.append(f"{c}_lag{LAGS-1-lag}")
    ctx = {
        "model": cb,
        "X_super": X_super, "y_super": y_super, "time_super": time_super, "wid_super": wid_super,
        "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
        "t_test": t_test, "w_test": w_test,
        "feature_cols": feature_cols, "feat_names_lag": feat_names, "F": F, "LAGS": LAGS,
        "report": report, "accuracy": accuracy, "cls1": cls1,
        "sv": sv_arr,
        "wafer_list": np.unique(wid_super.astype(str)).tolist()
    }
    return ctx

# ===============================
# 2-1) MSPC 함수 정의 (실시간 모니터링 탭에서 사용)
# ===============================
@st.cache_data
def calculate_mspc_model(df, labels_fault):
    normal_labels = ['none', 'calibration']
    normal_data = df[labels_fault.isin(normal_labels)]
    if normal_data.empty or normal_data.shape[0] <= 1:
        return None, "MSPC 분석을 위한 'none' 또는 'calibration' 데이터가 충분하지 않습니다."
    try:
        pca = PCA()
        pca.fit(normal_data)
        expl_var_cum = np.cumsum(pca.explained_variance_ratio_)
        target_cum = 0.90
        k = int(np.searchsorted(expl_var_cum, target_cum) + 1)
        k = max(1, min(k, normal_data.shape[1]))
        scores = pca.transform(df)[:, :k]
        reconstructed = scores @ pca.components_[:k, :]
        residuals = df.values - reconstructed
        q_statistic = np.sum(residuals**2, axis=1)
        epsilon = 1e-10
        t2_statistic = np.sum((scores**2) / (pca.explained_variance_[:k] + epsilon), axis=1)
        n, p = normal_data.shape
        def t2_limit(alpha, k, n):
            if n - k <= 0: return np.inf
            return (k * (n - 1) / (n - k)) * f.ppf(alpha, k, n - k)
        def q_limit(alpha, k, full_eigenvalues):
            lam_resid = full_eigenvalues[k:]
            if lam_resid.size == 0: return 0.0
            theta1 = np.sum(lam_resid)
            theta2 = np.sum(lam_resid**2)
            theta3 = np.sum(lam_resid**3)
            if theta2 == 0: return 0.0
            h0 = 1 - (2 * theta1 * theta3) / (3 * (theta2**2))
            h0 = max(h0, 0.001)
            z = norm.ppf(alpha)
            term_sqrt = 2 * theta2 * h0**2
            if term_sqrt < 0: term_sqrt = 0
            term1 = z * np.sqrt(term_sqrt) / theta1
            term2 = 1 + (theta2 * h0 * (h0 - 1)) / (theta1**2)
            term_power = term1 + term2
            if term_power < 0: return theta1
            return theta1 * (term_power**(1 / h0))
        T2_lim_95, T2_lim_99 = t2_limit(0.95, k, n), t2_limit(0.99, k, n)
        Q_lim_95, Q_lim_99 = q_limit(0.95, k, pca.explained_variance_), q_limit(0.99, k, pca.explained_variance_)
        results = {
            't2_statistic': t2_statistic,
            'q_statistic': q_statistic,
            't2_lim_95': T2_lim_95,
            't2_lim_99': T2_lim_99,
            'q_lim_95': Q_lim_95,
            'q_lim_99': Q_lim_99,
            'k': k,
            'expl_var_cum': expl_var_cum
        }
        return results, None
    except Exception as e:
        return None, f"MSPC 계산 중 오류가 발생했습니다: {e}"

def display_exceed_table():
    if st.session_state.mspc_results is None:
        return
    t2_lim_99 = st.session_state.mspc_results['t2_lim_99']
    q_lim_99 = st.session_state.mspc_results['q_lim_99']
    exceeding_df = st.session_state.displayed_df.copy()
    st.markdown("##### 99% 한계선 초과 데이터 개수 (실시간)")
    if not exceeding_df.empty:
        t2_exceed = exceeding_df[exceeding_df['T²'] > t2_lim_99].groupby(['Wafer Group', 'Fault Type']).size().reset_index(name='T2_Exceed_Count')
        q_exceed = exceeding_df[exceeding_df['Q'] > q_lim_99].groupby(['Wafer Group', 'Fault Type']).size().reset_index(name='Q_Exceed_Count')
        any_exceed_mask = (exceeding_df['T²'] > t2_lim_99) | (exceeding_df['Q'] > q_lim_99)
        total_sum_exceed_series = exceeding_df[any_exceed_mask].groupby(['Wafer Group', 'Fault Type']).size().reset_index(name='T2_Q_Total_Sum')
        final_table = pd.merge(t2_exceed, q_exceed, on=['Wafer Group', 'Fault Type'], how='outer').fillna(0)
        final_table = pd.merge(final_table, total_sum_exceed_series, on=['Wafer Group', 'Fault Type'], how='outer').fillna(0)
        final_table = final_table.rename(columns={
            'Wafer Group': '웨이퍼 그룹',
            'Fault Type': '결함 유형',
            'T2_Exceed_Count': 'T² 초과',
            'Q_Exceed_Count': 'Q 초과',
            'T2_Q_Total_Sum': 'T²&Q 동시 초과'
        })
        sorted_df = final_table.sort_values(by='T²&Q 동시 초과', ascending=False).reset_index(drop=True)
        final_display_cols = ['웨이퍼 그룹', '결함 유형', 'T² 초과', 'Q 초과', 'T²&Q 동시 초과']
        final_df = sorted_df[final_display_cols]
        st.dataframe(final_df)
    else:
        st.info("💡 99% 한계선을 초과하는 데이터가 없습니다.")
    st.markdown("---")


# ===============================
# 사이드바
# ===============================
#st.sidebar.title("📊 공정 모니터링 대시보드")

NAVY = "#0B5DBB"  # 로고 남색 톤

with st.sidebar:
    # --- 상단 로고 영역 ----
    st.image("image.png", use_container_width=True)
    st.markdown(
        """
        <style>
        .sidebar-logo {
            display: flex;
            justify-content: center;   /* 가운데 정렬 */
            margin-bottom: 10px;
        }
        [data-testid="stSidebar"] img {
            width: 50% !important;   /* 사이드바 대비 로고 크기 50% */
            height: auto !important;
            display: block;
            margin: 0 auto;          /* 가운데 정렬 */
        }       
        .sidebar-top{
            display:flex; justify-content:center; align-items:center;
            padding: 14px 8px 10px; margin-bottom: 6px;
        }
        .sidebar-divider{ height:1px; background:#EEF0F3; margin:6px 4px 10px; }
        /* 선택된 항목 흰색 처리 */
        div[role="radiogroup"] .nav-link-selected i,
        div[role="radiogroup"] .nav-link-selected span{
            color: #fff !important;
        }
        </style>
   
        """,
        unsafe_allow_html=True,
    )


    # --- 메뉴 ---
    page = option_menu(
        "대시보드",
        ["식각 시뮬레이션", "실시간 모니터링", "모델 결과 분석"],
        icons=["search", "activity", "cpu"],
        menu_icon="cast",
        default_index=0,
        styles={
            "icon": {"color": "#B0B7C3", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "4px -2px",        # 여백 줄이기
                "padding": "8px 10px",      # 패딩 줄이기
                "color": "#9CA3AF",
                "border-radius": "10px",
                "--hover-color": "#E9F1FF",
                "width": "170px",           # 🔹 nav-link 너비 확장
                "white-space": "nowrap",    # 🔹 줄바꿈 방지
            },
            "nav-link-selected": {
                "background-color": NAVY,
                "color": "#FFFFFF",
                "font-weight": "700",
            },
        },
    )



st.session_state.page = page

# ===============================
# 1) 식각 시뮬레이션
# ===============================
if page == "식각 시뮬레이션":
    # 페이지 설정 (이미 상단에서 한 번 호출되었을 수 있으므로 예외 무시)
    try:
        st.set_page_config(
            page_title="반도체 식각 시뮬레이션",
            page_icon="🔬",
            layout="wide"
        )
    except Exception:
        pass

    # CSS 스타일
    st.markdown("""
    <style>
        .main { padding-top: 1rem; }

        /* 아이콘형 버튼: 기본 네모 박스 제거하고 아이콘만 보이게 */
        .stButton > button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            width: 56px !important;
            height: 56px !important;
            font-size: 36px !important; /* 아이콘 크기 */
            line-height: 1 !important;
            cursor: pointer !important;
            border-radius: 50% !important; /* 호버 때 원형 하이라이트 */
            color: inherit !important;
        }
        .stButton > button:hover {
            background: rgba(0,0,0,0.06) !important;
            transform: scale(1.06);
        }
        .stButton > button:active {
            transform: scale(0.98);
        }
        .stButton > button:disabled {
            opacity: 0.4 !important;
            cursor: not-allowed !important;
            transform: none !important;
            background: transparent !important;
        }

        /* 공통 박스 스타일 */
        .stage-box, .layer-box {
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            border: 2px solid;
            font-size: 0.9em;
            min-height: 100px;
            box-sizing: border-box;
        }
        .stage-box h4, .layer-box h4 {
            font-size: 1em;
            margin: 0 0 0 0;
        }
        .stage-box p, .layer-box p {
            margin: 3px 0;
            font-size: 0.85em;
        }
        .stage-box { background-color: #f8f9fa; border-color: #ddd; }
        .al-box { background-color: #FFE4E8; border-color: #E8A0A8; }
        .tin-box { background-color: #E4E4FF; border-color: #9B8FCC; }
        .oxide-box { background-color: #E4F4FF; border-color: #87CEEB; }
    </style>
    """, unsafe_allow_html=True)

    # 제목
    st.title("실시간 반도체 식각 시뮬레이션")
    st.markdown("---")

    # 세션 상태 초기화
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'simulation_paused' not in st.session_state:
        st.session_state.simulation_paused = False
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'elapsed_time' not in st.session_state:
        st.session_state.elapsed_time = 0
    if 'paused_elapsed_time' not in st.session_state:
        st.session_state.paused_elapsed_time = 0
    if 'wafer_data' not in st.session_state:
        st.session_state.wafer_data = None
    if 'selected_wafer' not in st.session_state:
        st.session_state.selected_wafer = None

    # === (추가) 배속 & 가상시간 세션 상태 ===
    if 'speed' not in st.session_state:
        st.session_state.speed = 1.0
    if 'elapsed_virtual' not in st.session_state:
        st.session_state.elapsed_virtual = 0.0
    if 'time_origin_real' not in st.session_state:
        st.session_state.time_origin_real = None

    # CSV 파일 로드
    @st.cache_data
    def load_wafer_data():
        try:
            df = pd.read_csv('etch_stage_times.csv')
            return df
        except FileNotFoundError:
            st.error("etch_stage_times.csv 파일을 찾을 수 없습니다.")
            return None
        except Exception as e:
            st.error(f"파일 로드 중 오류 발생: {e}")
            return None

    # 데이터 로드
    wafer_df = load_wafer_data()

    # 레이어 정보 정의 (수정됨)
    layers = [
        {'name': 'PR', 'thickness': 0.8, 'color': '#FF8C42', 'label': 'PR'},
        {'name': 'Al', 'thickness': 2.6, 'color': '#E8A0A8', 'label': 'Al'},
        {'name': 'TiN', 'thickness': 0.5, 'color': '#9B8FCC', 'label': 'TiN'},
        {'name': 'Oxide', 'thickness': 1.5, 'color': '#A0C4F2', 'label': '산화막'},
        {'name': 'Wafer', 'thickness': 1.0, 'color': '#4A4A4A', 'label': '웨이퍼'}
    ]

    # ===== 콜백: 시작/일시정지 토글 =====
    def toggle_run():
        """버튼 클릭 시 실행/일시정지 상태를 정확히 1클릭으로 전환"""
        now = time.time()
        running = st.session_state.simulation_running
        paused  = st.session_state.simulation_paused

        if running and not paused:
            # 실행 → 일시정지: 지금까지의 가상 경과시간 저장
            st.session_state.simulation_paused = True
            st.session_state.simulation_running = False
            st.session_state.paused_elapsed_time = st.session_state.elapsed_virtual
        else:
            # 대기/일시정지 → 실행(재개)
            st.session_state.simulation_running = True
            st.session_state.simulation_paused = False
            # 최초 시작 또는 재개 기준 실제시간 원점
            st.session_state.time_origin_real = now
            if st.session_state.start_time is None:
                st.session_state.start_time = now
                st.session_state.elapsed_time = 0
                st.session_state.elapsed_virtual = 0.0
                st.session_state.paused_elapsed_time = 0.0

    # ===== 콜백: 웨이퍼 변경 시 '완전 초기화' =====
    def on_wafer_change():
        st.session_state.simulation_running = False
        st.session_state.simulation_paused = False
        st.session_state.start_time = None
        st.session_state.elapsed_time = 0
        st.session_state.paused_elapsed_time = 0
        # (추가) 배속 가상시간 초기화
        st.session_state.elapsed_virtual = 0.0
        st.session_state.time_origin_real = None

    # 메인 컨테이너 (3:2 비율 유지)
    col1, col2 = st.columns([3, 2])

    with col1:
        left_col, right_col = st.columns([2.5, 3])

        with left_col:
            col_wafer, col_speed = st.columns([3, 2])
            with col_wafer:
                if wafer_df is not None:
                    selected_wafer = st.selectbox(
                        "웨이퍼 선택",
                        wafer_df['wafer_names'].tolist(),
                        key='wafer_select',
                        on_change=on_wafer_change
                    )

                    st.session_state.selected_wafer = selected_wafer
                    wafer_row = wafer_df[wafer_df['wafer_names'] == selected_wafer].iloc[0]
                    total_time = wafer_row['stage5']
            with col_speed:
                    speed_options = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
                    # 현재 speed가 옵션에 없을 경우를 대비한 index 계산
                    try:
                        idx = speed_options.index(float(st.session_state.speed))
                    except Exception:
                        idx = speed_options.index(1.0)
                    st.session_state.speed = st.selectbox(
                        "속도 설정", options=speed_options, index=idx
                    )
                
        with right_col:
            st.markdown(
                "<div style='margin-bottom:-50px; padding-bottom:0px; font-size:14px; margin-left:15px;'>컨트롤 버튼</div>",
                unsafe_allow_html=True
            )
            icon_label = "⏸️" if (st.session_state.simulation_running and not st.session_state.simulation_paused) else "▶️"
            col_a, col_b = st.columns([1, 3])
            with col_a:
                col_start, col_stop= st.columns([1, 1])
                with col_start:
                    st.button(icon_label, key="toggle_btn", on_click=toggle_run, disabled=(wafer_df is None))
                with col_stop:
                    reset_button = st.button("⏹️", key="reset_btn")
                

            with col_b:
                st.markdown(
                    """
                    <style>
                    .custom-alert {
                        width: 120px;
                        padding: 0.3em 0.5em;
                        font-size: 0.7em;
                        margin-left:15px;
                        border-radius: 15px;
                        text-align: center;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                if st.session_state.simulation_running and not st.session_state.simulation_paused:
                    st.markdown('<div class="custom-alert" style="background-color:#d4edda; color:#155724;">식각 진행 중</div>', unsafe_allow_html=True)
                elif st.session_state.simulation_paused:
                    st.markdown('<div class="custom-alert" style="background-color:#fff3cd; color:#856404;">일시 중단</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="custom-alert" style="background-color:#f8d7da; color:#721c24;">식각 대기 중</div>', unsafe_allow_html=True)

        plot_placeholder = st.empty()

        # 전체 진행률
        st.markdown("###### 전체 진행률")
        progress_cols = st.columns([4, 1])
        progress_bar = progress_cols[0].progress(0)
        percent_text = progress_cols[1].markdown("0%")
        completion_cols = st.columns([4, 1])
        completion_placeholder = completion_cols[0].empty()

    with col2:
        process_time_placeholder = st.empty()
        stage1_placeholder = st.empty()
        stage2_placeholder = st.empty()
        al_placeholder = st.empty()
        tin_placeholder = st.empty()
        oxide_placeholder = st.empty()

    def calculate_current_heights(elapsed_time, wafer_row):
        stage2_time = wafer_row['stage2']
        stage3_time = wafer_row['stage3']
        stage4_time = wafer_row['stage4']
        stage5_time = wafer_row['stage5']
        center_heights = {
            'PR': layers[0]['thickness'],
            'Al': layers[1]['thickness'],
            'TiN': layers[2]['thickness'],
            'Oxide': layers[3]['thickness'],
            'Wafer': layers[4]['thickness']
        }
        if elapsed_time <= stage2_time:
            center_heights['Al'] = layers[1]['thickness']
        elif elapsed_time >= stage3_time:
            center_heights['Al'] = 0
        else:
            progress = (elapsed_time - stage2_time) / (stage3_time - stage2_time)
            center_heights['Al'] = layers[1]['thickness'] * (1 - progress)
        if elapsed_time <= stage3_time:
            center_heights['TiN'] = layers[2]['thickness']
        elif elapsed_time >= stage4_time:
            center_heights['TiN'] = 0
        else:
            progress = (elapsed_time - stage3_time) / (stage4_time - stage3_time)
            center_heights['TiN'] = layers[2]['thickness'] * (1 - progress)
        if elapsed_time <= stage4_time:
            center_heights['Oxide'] = layers[3]['thickness']
        elif elapsed_time >= stage5_time:
            center_heights['Oxide'] = 0
        else:
            progress = (elapsed_time - stage4_time) / (stage5_time - stage4_time)
            center_heights['Oxide'] = layers[3]['thickness'] * (1 - progress)
        return center_heights

    def draw_layers(elapsed_time, wafer_row):
        fig, ax = plt.subplots(figsize=(8, 10))
        center_heights = calculate_current_heights(elapsed_time, wafer_row)
        ax.set_facecolor('#f0f0f0')
        fig.patch.set_facecolor('white')
        pr_left_x = 0
        pr_right_x = 4
        pr_width = 2.5
        center_gap = 1.5
        arrow_y = 6.5
        for x in [2.875, 3.25, 3.625]:
            ax.arrow(x, arrow_y, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
        y_position = 0
        rect = patches.Rectangle((pr_left_x, y_position), pr_width * 2 + center_gap, layers[4]['thickness'],
                                 linewidth=2, edgecolor='none', facecolor=layers[4]['color'])
        ax.add_patch(rect)
        ax.text(3.25, y_position + layers[4]['thickness']/2, layers[4]['label'],
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        y_position += layers[4]['thickness']
        rect = patches.Rectangle((pr_left_x, y_position), pr_width, layers[3]['thickness'],
                                 linewidth=2, edgecolor='none', facecolor=layers[3]['color'])
        ax.add_patch(rect)
        rect = patches.Rectangle((pr_right_x, y_position), pr_width, layers[3]['thickness'],
                                 linewidth=2, edgecolor='none', facecolor=layers[3]['color'])
        ax.add_patch(rect)
        if center_heights['Oxide'] > 0:
            rect = patches.Rectangle((pr_left_x + pr_width, y_position), center_gap, center_heights['Oxide'],
                                     linewidth=2, edgecolor='none', facecolor=layers[3]['color'])
            ax.add_patch(rect)
            if center_heights['Oxide'] > 0.3:
                ax.text(3.25, y_position + center_heights['Oxide']/2, layers[3]['label'],
                        ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        y_position_oxide = y_position + layers[3]['thickness']
        rect = patches.Rectangle((pr_left_x, y_position_oxide), pr_width, layers[2]['thickness'],
                                 linewidth=2, edgecolor='none', facecolor=layers[2]['color'])
        ax.add_patch(rect)
        rect = patches.Rectangle((pr_right_x, y_position_oxide), pr_width, layers[2]['thickness'],
                                 linewidth=2, edgecolor='none', facecolor=layers[2]['color'])
        ax.add_patch(rect)
        if center_heights['TiN'] > 0:
            rect = patches.Rectangle((pr_left_x + pr_width, y_position + center_heights['Oxide']),
                                     center_gap, center_heights['TiN'],
                                     linewidth=2, edgecolor='none', facecolor=layers[2]['color'])
            ax.add_patch(rect)
            if center_heights['TiN'] > 0.2:
                ax.text(3.25, y_position + center_heights['Oxide'] + center_heights['TiN']/2, layers[2]['label'],
                        ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        y_position_tin = y_position_oxide + layers[2]['thickness']
        rect = patches.Rectangle((pr_left_x, y_position_tin), pr_width, layers[1]['thickness'],
                                 linewidth=2, edgecolor='none', facecolor=layers[1]['color'])
        ax.add_patch(rect)
        rect = patches.Rectangle((pr_right_x, y_position_tin), pr_width, layers[1]['thickness'],
                                 linewidth=2, edgecolor='none', facecolor=layers[1]['color'])
        ax.add_patch(rect)
        if center_heights['Al'] > 0:
            rect = patches.Rectangle((pr_left_x + pr_width, y_position + center_heights['Oxide'] + center_heights['TiN']),
                                     center_gap, center_heights['Al'],
                                     linewidth=2, edgecolor='none', facecolor=layers[1]['color'])
            ax.add_patch(rect)
            if center_heights['Al'] > 0.3:
                ax.text(3.25, y_position + center_heights['Oxide'] + center_heights['TiN'] + center_heights['Al']/2,
                        layers[1]['label'], ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        y_position_al = y_position_tin + layers[1]['thickness']
        rect = patches.Rectangle((pr_left_x, y_position_al), pr_width, layers[0]['thickness'],
                                 linewidth=2, edgecolor='none', facecolor=layers[0]['color'])
        ax.add_patch(rect)
        ax.text(pr_left_x + pr_width/2, y_position_al + layers[0]['thickness']/2, 'PR',
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        rect = patches.Rectangle((pr_right_x, y_position_al), pr_width, layers[0]['thickness'],
                                 linewidth=2, edgecolor='none', facecolor=layers[0]['color'])
        ax.add_patch(rect)
        ax.text(pr_right_x + pr_width/2, y_position_al + layers[0]['thickness']/2, 'PR',
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        ax.set_xlim(-0.5, 8)
        ax.set_ylim(0, 7)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        return fig

    def update_info_display(elapsed_time, wafer_row):
        total_time = wafer_row['stage5']
        process_time_placeholder.metric("⏱️ 공정시간", f"{elapsed_time:.1f} / {total_time:.1f} 초")
        stage1_time = wafer_row['stage1']
        if elapsed_time > 0:
            if elapsed_time <= stage1_time:
                progress = (elapsed_time / stage1_time) * 100
                stage1_placeholder.markdown(f"""
                <div class="stage-box">
                    <h4>1. 가스 유량 & 압력 안정화 🔄</h4>
                    <p style="font-size: 11px;">식각을 위한 최적의 환경을 조성하기 위해 챔버 내의 가스 유량과 압력 안정화</p>
                    <p>진행률: {progress:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                stage1_placeholder.markdown(f"""
                <div class="stage-box">
                    <h4>1. 가스 유량 & 압력 안정화 ✅</h4>
                    <p style="font-size: 11px;">식각을 위한 최적의 환경을 조성하기 위해 챔버 내의 가스 유량과 압력 안정화</p>
                    <p>완료: {stage1_time:.1f}초</p>
                </div>
                """, unsafe_allow_html=True)
        stage2_time = wafer_row['stage2']
        if elapsed_time > stage1_time:
            if elapsed_time <= stage2_time:
                progress = ((elapsed_time - stage1_time) / (stage2_time - stage1_time)) * 100
                stage2_placeholder.markdown(f"""
                <div class="stage-box">
                    <h4>2. 플라스마 점화 🔄</h4>
                    <p style="font-size: 11px;">RF 전력을 인가하여 유도결합 플라즈마를 발생시켜 식각에 필요한 이온 활성화</p>
                    <p>진행률: {progress:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                stage2_placeholder.markdown(f"""
                <div class="stage-box">
                    <h4>2. 플라스마 점화 ✅</h4>
                    <p style="font-size: 11px;">RF 전력을 인가하여 유도결합 플라즈마를 발생시켜 식각에 필요한 이온 활성화</p>
                    <p>완료: {stage2_time:.1f}초</p>
                </div>
                """, unsafe_allow_html=True)
        stage3_time = wafer_row['stage3']
        if elapsed_time > stage2_time:
            if elapsed_time <= stage3_time:
                progress = ((elapsed_time - stage2_time) / (stage3_time - stage2_time)) * 100
                al_placeholder.markdown(f"""
                <div class="layer-box al-box">
                    <h4>3. Al 식각 🔄</h4>
                    <p style="font-size: 11px;">유도 결합된 플라즈마를 사용하여 주 식각 대상인 알루미늄 층을 식각</p>
                    <p>식각률: {progress:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                al_placeholder.markdown(f"""
                <div class="layer-box al-box">
                    <h4>3. Al 식각 ✅</h4>
                    <p style="font-size: 11px;">유도 결합된 플라즈마를 사용하여 주 식각 대상인 알루미늄 층을 식각</p>
                    <p>완료: {stage3_time:.1f}초</p>
                </div>
                """, unsafe_allow_html=True)
        stage4_time = wafer_row['stage4']
        if elapsed_time > stage3_time:
            if elapsed_time <= stage4_time:
                progress = ((elapsed_time - stage3_time) / (stage4_time - stage3_time)) * 100
                tin_placeholder.markdown(f"""
                <div class="layer-box tin-box">
                    <h4>4. TiN 식각 🔄</h4>
                    <p style="font-size: 11px;">Al 층 아래에 있는 TiN을 추가 식각하여 잔여물 제거 및 식각 균일성 확보</p>
                    <p>식각률: {progress:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                tin_placeholder.markdown(f"""
                <div class="layer-box tin-box">
                    <h4>4. TiN 식각 ✅</h4>
                    <p style="font-size: 11px;">Al 층 아래에 있는 TiN을 추가 식각하여 잔여물 제거 및 식각 균일성 확보</p>
                    <p>완료: {stage4_time:.1f}초</p>
                </div>
                """, unsafe_allow_html=True)
        stage5_time = wafer_row['stage5']
        if elapsed_time > stage4_time:
            if elapsed_time < stage5_time:
                progress = ((elapsed_time - stage4_time) / (stage5_time - stage4_time)) * 100
                oxide_placeholder.markdown(f"""
                <div class="layer-box oxide-box">
                    <h4>5. 산화막 식각 🔄</h4>
                    <p style="font-size: 11px;">TiN 층 아래 있는 산화막을 추가로 식각, 산화물 손실 주의</p>
                    <p>식각률: {progress:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                oxide_placeholder.markdown(f"""
                <div class="layer-box oxide-box">
                    <h4>5. 산화막 식각 ✅</h4>
                    <p style="font-size: 11px;">TiN 층 아래 있는 산화막을 추가로 식각, 산화물 손실 주의</p>
                    <p>완료: {stage5_time:.1f}초</p>
                </div>
                """, unsafe_allow_html=True)

    # ===== 리셋 처리 =====
    if 'reset_btn' in st.session_state and st.session_state.get('reset_btn'):
        st.session_state.simulation_running = False
        st.session_state.simulation_paused = False
        st.session_state.start_time = None
        st.session_state.elapsed_time = 0
        st.session_state.paused_elapsed_time = 0
        # (추가) 배속 가상시간 초기화
        st.session_state.elapsed_virtual = 0.0
        st.session_state.time_origin_real = None
        st.rerun()

    # ===== 시뮬레이션 루프 =====
    if st.session_state.simulation_running and not st.session_state.simulation_paused and wafer_df is not None and st.session_state.selected_wafer:
        wafer_row = wafer_df[wafer_df['wafer_names'] == st.session_state.selected_wafer].iloc[0]
        total_time = wafer_row['stage5']
        while st.session_state.simulation_running and not st.session_state.simulation_paused:
            now = time.time()
            if st.session_state.time_origin_real is None:
                st.session_state.time_origin_real = now
            # (변경) 가상 경과시간 = 누적(일시정지까지) + (재개 후 실제경과 × 배속)
            st.session_state.elapsed_virtual = (
                st.session_state.paused_elapsed_time +
                (now - st.session_state.time_origin_real) * st.session_state.speed
            )
            elapsed = st.session_state.elapsed_virtual

            # (기존 elapsed_time은 유지하되, 표시에만 가상시간 사용)
            st.session_state.elapsed_time = elapsed

            if elapsed >= total_time:
                st.session_state.simulation_running = False
                st.session_state.elapsed_virtual = total_time
                elapsed = total_time

            fig = draw_layers(elapsed, wafer_row)
            plot_placeholder.pyplot(fig)
            plt.close(fig)
            update_info_display(elapsed, wafer_row)

            progress = min(1.0, elapsed / total_time)
            progress_bar.progress(progress)
            percent_text.markdown(f"**{int(progress*100)}%**")

            time.sleep(0.1)
            if not st.session_state.simulation_running:
                break
        if st.session_state.elapsed_virtual >= total_time:
            completion_placeholder.success("🎉 식각 시뮬레이션이 완료되었습니다!")
    elif wafer_df is not None and st.session_state.selected_wafer:
        wafer_row = wafer_df[wafer_df['wafer_names'] == st.session_state.selected_wafer].iloc[0]
        # (변경) 미실행 상태에서도 가상시간 기준으로 표시
        elapsed = st.session_state.elapsed_virtual
        fig = draw_layers(elapsed, wafer_row)
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        update_info_display(elapsed, wafer_row)
        total_time = wafer_row['stage5']
        if total_time > 0:
            progress = min(1.0, elapsed / total_time)
            progress_bar.progress(progress)
            percent_text.markdown(f"**{int(progress*100)}%**")
            if progress >= 1.0:
                completion_placeholder.success("🎉 식각 시뮬레이션이 완료되었습니다!")
            else:
                completion_placeholder.empty()



# ===============================
# 2) 실시간 공정 모니터링
# ===============================
elif page == "실시간 모니터링":
    st.title("실시간 공정 모니터링")
    tab1, tab2 = st.tabs(["MSPC", "센서 데이터"])
    with tab1:
        file_path = "final_data_with_rfm3.csv"
        if "is_running" not in st.session_state:
            st.session_state.is_running = False
            st.session_state.current_index = 0
            st.session_state.mspc_results = None
            st.session_state.scaled_df = None
            st.session_state.labels_fault = None
            st.session_state.selected_option = None
            st.session_state.displayed_df = pd.DataFrame(columns=['Index', 'Wafer Group', 'T²', 'Q', 'Fault Type'])
            st.session_state.stop_thresholds = {'29': 15, '31': 20, '33': 9}
        try:
            p_mspc = find_existing_path(file_path)
            df = robust_read_csv(p_mspc)
            #st.success(f"'{file_path}' 파일이 성공적으로 로드되었습니다. ✅")
            if 'fault_name' not in df.columns or 'wafer_names' not in df.columns:
                st.error("오류: 데이터프레임에 'fault_name' 또는 'wafer_names' 열이 없습니다. 컬럼 이름을 확인해 주세요.")
                st.stop()
            df['wafer_group'] = df['wafer_names'].astype(str).str[:2]
            unique_groups = sorted(list(df['wafer_group'].unique()))
            selection_options = ['전체 웨이퍼'] + [f"그룹 {g}" for g in unique_groups]
            selected_option = st.selectbox("웨이퍼 그룹을 선택하세요:", selection_options, key="wafer_group_selector")
            if selected_option != st.session_state.selected_option:
                st.session_state.is_running = False
                st.session_state.current_index = 0
                st.session_state.mspc_results = None
                st.session_state.scaled_df = None
                st.session_state.selected_option = selected_option
                st.session_state.displayed_df = pd.DataFrame(columns=['Index', 'Wafer Group', 'T²', 'Q', 'Fault Type'])
                st.rerun()
            if selected_option == '전체 웨이퍼':
                filtered_df = df.copy()
            elif selected_option.startswith('그룹 '):
                group_name = selected_option.replace('그룹 ', '')
                filtered_df = df[df['wafer_group'] == group_name].copy()
            if filtered_df.empty:
                st.warning(f"{selected_option}에 해당하는 데이터가 없습니다. 다른 웨이퍼를 선택해주세요.")
                st.stop()
            filtered_df['sort_key'] = filtered_df['fault_name'].apply(lambda x: 1 if x in ['calibration', 'none'] else 0)
            filtered_df = filtered_df.sort_values(by=['sort_key', 'wafer_names']).reset_index(drop=True)
            filtered_df = filtered_df.drop(columns='sort_key')
            labels_fault = filtered_df['fault_name']
            wafer_groups = filtered_df['wafer_group']
            data_columns = filtered_df.select_dtypes(include=np.number).columns.drop(['lot_id', 'wafer_id', 'wafer_names'], errors='ignore')
            numerical_df = filtered_df[data_columns]
            scaler = StandardScaler()
            scaled_df = pd.DataFrame(scaler.fit_transform(numerical_df), columns=numerical_df.columns)
            st.session_state.scaled_df = scaled_df
            st.session_state.labels_fault = labels_fault
            st.session_state.wafer_names = filtered_df['wafer_names']
            st.session_state.wafer_groups = wafer_groups
            if st.session_state.mspc_results is None:
                with st.spinner("MSPC 모델을 구축하고 있습니다..."):
                    st.session_state.mspc_results, error_message = calculate_mspc_model(scaled_df, labels_fault)
                if error_message:
                    st.warning(error_message)
                    st.stop()
            st.markdown("---")
            st.subheader(f"📊 {selected_option} MSPC 관리도 시뮬레이션")
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("시작", key="start"):
                    if not st.session_state.is_running and st.session_state.current_index > 0:
                        last_processed_idx = st.session_state.current_index - 1 
                        if last_processed_idx >= 0 and last_processed_idx < len(st.session_state.wafer_names):
                            current_wafer_name = st.session_state.wafer_names.iloc[last_processed_idx]
                            current_group_end_index = st.session_state.wafer_names[st.session_state.wafer_names == current_wafer_name].index[-1]
                            next_start_index = current_group_end_index + 1
                        else:
                            next_start_index = 0
                        if next_start_index < len(st.session_state.scaled_df):
                            st.session_state.current_index = next_start_index
                            st.session_state.displayed_df = pd.DataFrame(columns=['Index', 'Wafer Group', 'T²', 'Q', 'Fault Type'])
                            st.session_state.is_running = True
                            st.success(f"다음 웨이퍼 데이터부터 시뮬레이션을 재개합니다. (시작 인덱스: {next_start_index})")
                        else:
                            st.info("시뮬레이션 데이터의 끝에 도달했습니다. 초기화 버튼을 눌러 다시 시작하세요.")
                            st.session_state.is_running = False
                    else:
                        st.session_state.is_running = True
                        if st.session_state.current_index > 0:
                             st.session_state.current_index = 0
                             st.session_state.displayed_df = pd.DataFrame(columns=['Index', 'Wafer Group', 'T²', 'Q', 'Fault Type'])
                    st.rerun()
            with col2:
                if st.button("중지", key="stop"):
                    st.session_state.is_running = False
            with col3:
                if st.button("초기화", key="reset"):
                    st.session_state.is_running = False
                    st.session_state.current_index = 0
                    st.session_state.displayed_df = pd.DataFrame(columns=['Index', 'Wafer Group', 'T²', 'Q', 'Fault Type'])
                    st.rerun()
            placeholder = st.empty()
            if st.session_state.is_running and st.session_state.current_index < len(scaled_df):
                with placeholder.container():
                    current_idx = st.session_state.current_index
                    t2_val = st.session_state.mspc_results['t2_statistic'][current_idx]
                    q_val = st.session_state.mspc_results['q_statistic'][current_idx]
                    fault_type = st.session_state.labels_fault.iloc[current_idx]
                    wafer_group = st.session_state.wafer_groups.iloc[current_idx]
                    t2_lim_99 = st.session_state.mspc_results['t2_lim_99']
                    q_lim_99 = st.session_state.mspc_results['q_lim_99']
                    new_row = pd.DataFrame([{'Index': current_idx, 'Wafer Group': wafer_group, 'T²': t2_val, 'Q': q_val, 'Fault Type': fault_type}])
                    st.session_state.displayed_df = pd.concat([st.session_state.displayed_df, new_row], ignore_index=True)
                    total_sum_exceed = 0
                    if fault_type not in ['calibration', 'none'] and wafer_group in st.session_state.stop_thresholds:
                        current_group_data = st.session_state.displayed_df[
                            (st.session_state.displayed_df['Wafer Group'] == wafer_group) & 
                            (st.session_state.displayed_df['Fault Type'] == fault_type)
                        ].copy() 
                        any_exceed_mask_for_group = (current_group_data['T²'] > t2_lim_99) | (current_group_data['Q'] > q_lim_99)
                        total_sum_exceed = any_exceed_mask_for_group.sum()
                    
                            
                            
                    col_header, col_status = st.columns([6, 2])  # 비율 조정 (왼쪽 넓게, 오른쪽 좁게)
                    is_any_exceedance = (t2_val > t2_lim_99) or (q_val > q_lim_99)
                    with col_header:
                        st.markdown(
                            f"##### **현재 데이터 포인트**: `{current_idx}` "
                            f"| **웨이퍼**: `{st.session_state.wafer_names.iloc[current_idx]}` "
                            f"| **T²&Q 동시 초과 개수**: `{total_sum_exceed}`"
                        )

                    with col_status:
                        if is_any_exceedance:
                            is_simultaneous_exceedance = (t2_val > t2_lim_99) and (q_val > q_lim_99)
                            if is_simultaneous_exceedance:
                                st.markdown(
                                    "<p style='color:red; font-size:25px; margin-left:-8px;'><b>⚠️ T² & Q 동시 초과!</p>",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    "<p style='color:orange; font-size:25px; margin-left:-8px;'><b>❗ 한계선 초과</p>",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.markdown(
                                "<p style='color:green; font-size:25px; margin-left:-8px;'><b>✅ 정상</p>",
                                unsafe_allow_html=True
                            )
        

                    
                    st.markdown("---")
                    if fault_type not in ['calibration', 'none'] and wafer_group in st.session_state.stop_thresholds:
                        threshold = st.session_state.stop_thresholds.get(wafer_group)
                        if total_sum_exceed >= threshold:
                            st.markdown(f"<h4 style='color:red;'>🚨 이상 감지! 공정을 즉시 중단합니다. (결함 유형: <span style='color:blue;'>{fault_type}</span>, 임계치 초과: {threshold}개 이상)</h3>", unsafe_allow_html=True)
                            st.session_state.is_running = False
                    st.session_state.current_index += 1
                    t2_lim_95 = st.session_state.mspc_results['t2_lim_95']
                    q_lim_95 = st.session_state.mspc_results['q_lim_95']
                    col_plot1, col_plot2 = st.columns(2)
                    with col_plot1:
                        fig1, ax1 = plt.subplots(figsize=(10, 5))
                        ax1.plot(st.session_state.displayed_df['Index'], st.session_state.displayed_df['T²'], marker='o', linestyle='-', label='T² 통계량', alpha=0.7)
                        ax1.axhline(t2_lim_99, color='red', linestyle='--', label=f'99% 한계선 ({t2_lim_99:.2f})')
                        ax1.set_title('Hotelling’s T² 관리도', fontsize=18)
                        ax1.set_ylabel('T² 값')
                        ax1.set_xlabel('데이터 인덱스')
                        ax1.legend()
                        ax1.grid(True, linestyle='--', alpha=0.6)
                        st.pyplot(fig1)
                    with col_plot2:
                        fig2, ax2 = plt.subplots(figsize=(10, 5))
                        ax2.plot(st.session_state.displayed_df['Index'], st.session_state.displayed_df['Q'], marker='o', linestyle='-', color='g', label='Q 통계량 (SPE)', alpha=0.7)
                        ax2.axhline(q_lim_99, color='red', linestyle='--', label=f'99% 한계선 ({q_lim_99:.2e})')
                        ax2.set_title('Q 통계량 (SPE) 관리도', fontsize=18)
                        ax2.set_ylabel('Q 값')
                        ax2.set_xlabel('데이터 인덱스')
                        ax2.legend()
                        ax2.grid(True, linestyle='--', alpha=0.6)
                        st.pyplot(fig2)
                    display_exceed_table()
                if st.session_state.is_running:
                    time.sleep(0.05)
                    st.rerun()
            else:
                if not st.session_state.displayed_df.empty:
                    with placeholder.container():
                        t2_lim_95 = st.session_state.mspc_results['t2_lim_95']
                        t2_lim_99 = st.session_state.mspc_results['t2_lim_99']
                        q_lim_95 = st.session_state.mspc_results['q_lim_95']
                        q_lim_99 = st.session_state.mspc_results['q_lim_99']
                        
                        #st.subheader("마지막 업데이트")
                        last_idx = st.session_state.displayed_df.index[-1]
                        last_t2_val = st.session_state.displayed_df.loc[last_idx, 'T²']
                        last_q_val = st.session_state.displayed_df.loc[last_idx, 'Q']
                        col_header, col_status = st.columns([3, 1])
                        is_any_exceedance_last = (last_t2_val > t2_lim_99) or (last_q_val > q_lim_99)
                        with col_header:
                            st.markdown(f"##### 마지막 데이터 포인트: `{st.session_state.displayed_df.loc[last_idx, 'Index']}` | 웨이퍼: `{st.session_state.wafer_names.iloc[st.session_state.displayed_df.loc[last_idx, 'Index']]}`")
                        with col_status:
                            if is_any_exceedance_last:
                                is_simultaneous_exceedance_last = (last_t2_val > t2_lim_99) and (last_q_val > q_lim_99)
                                if is_simultaneous_exceedance_last:
                                    st.markdown(
                                    "<p style='color:red; font-size:25px; margin-left:-8px;'><b>⚠️ T² & Q 동시 초과!</p>",
                                    unsafe_allow_html=True
                                )
                                else:
                                    st.markdown(
                                    "<p style='color:orange; font-size:25px; margin-left:-8px;'><b>❗ 한계선 초과</p>",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                "<p style='color:green; font-size:25px; margin-left:-8px;'><b>✅ 정상</p>",
                                unsafe_allow_html=True
                            )
                        st.markdown("---")
                        col_plot1, col_plot2 = st.columns(2)
                        with col_plot1:
                            fig1, ax1 = plt.subplots(figsize=(10, 5))
                            ax1.plot(st.session_state.displayed_df['Index'], st.session_state.displayed_df['T²'], marker='o', linestyle='-', label='T² 통계량', alpha=0.7)
                            ax1.axhline(t2_lim_95, color='orange', linestyle='--', label=f'95% 한계선 ({t2_lim_95:.2f})')
                            ax1.axhline(t2_lim_99, color='red', linestyle='--', label=f'99% 한계선 ({t2_lim_99:.2f})')
                            ax1.set_title('Hotelling’s T² 관리도', fontsize=18)
                            ax1.set_ylabel('T² 값')
                            ax1.set_xlabel('데이터 인덱스')
                            ax1.legend()
                            ax1.grid(True, linestyle='--', alpha=0.6)
                            st.pyplot(fig1)
                        with col_plot2:
                            fig2, ax2 = plt.subplots(figsize=(10, 5))
                            ax2.plot(st.session_state.displayed_df['Index'], st.session_state.displayed_df['Q'], marker='o', linestyle='-', color='g', label='Q 통계량 (SPE)', alpha=0.7)
                            ax2.axhline(q_lim_95, color='orange', linestyle='--', label=f'95% 한계선 ({q_lim_95:.2e})')
                            ax2.axhline(q_lim_99, color='red', linestyle='--', label=f'99% 한계선 ({q_lim_99:.2e})')
                            ax2.set_title('Q 통계량 (SPE) 관리도', fontsize=18)
                            ax2.set_ylabel('Q 값')
                            ax2.set_xlabel('데이터 인덱스')
                            ax2.legend()
                            ax2.grid(True, linestyle='--', alpha=0.6)
                            st.pyplot(fig2)
                        display_exceed_table()
                        #st.subheader("실시간 데이터 테이블 (최근 10개)")
                        #st.dataframe(st.session_state.displayed_df.tail(10))
                elif st.session_state.current_index >= len(scaled_df) and not st.session_state.displayed_df.empty:
                    st.info("✅ 시뮬레이션이 완료되었습니다. 초기화 후 다시 시작해 주세요.")
                elif st.session_state.mspc_results is not None:
                    st.info("시뮬레이션을 시작하거나 '초기화' 후 다시 시도해 주세요.")
        except FileNotFoundError:
            st.error(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 해당 파일을 이 스크립트와 동일한 폴더에 넣어주세요.")
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")

    with tab2:
        #st.subheader("센서 데이터 모니터링")

        #with st.expander("결함 유형 분포 확인"):
            #st.dataframe(df.drop_duplicates(subset=['wafer_names'])['fault_name'].value_counts())
        
        unique_wafers = sorted(df['wafer_names'].unique())
        # 탭 간 위젯 충돌을 피하기 위해 고유한 key를 부여합니다.
        selected_wafer = st.selectbox("분석할 웨이퍼를 선택하세요:", unique_wafers, key="sensor_wafer_selector")
        df_wafer = df[df['wafer_names'] == selected_wafer].reset_index(drop=True)
        
        st.markdown("---")

        st.markdown(f"### 🔬 **{selected_wafer}** 웨이퍼 모니터링")
        #st.info(f"선택된 웨이퍼의 결함 유형: **{df_wafer['fault_name'].iloc[0]}**")
        
        # 모니터링할 컬럼 정의
        proc_cols = ['Endpt A', 'Vat Valve', 'Pressure', 'TCP Top Pwr', 'RF Btm Pwr']
        rfm_cols = ['S1I3', 'S1I5', 'S2I5', 'S34PV1', 'S34V5']
        oes_cols = ['250.0', '336.98', '395.8', '725.0', '773.2']
        
        # OES 제목 및 y축 단위(unit)를 위한 딕셔너리 정의
        oes_title_map = {
            '250.0': 'Si (250.0 nm)', '336.98': 'Ti (336.98 nm)', '395.8': 'Al (395.8 nm)',
            '725.0': 'Cl (725.0 nm)', '773.2': 'O (773.2 nm)'
        }
        
        unit_map = {
            'Endpt A': 'Endpoint Signal', 'Vat Valve': 'Position (%)', 'RF Btm Pwr': 'Power (P)',
            'Pressure': 'Pressure (mTorr)', 'TCP Top Pwr': 'Power (P)', 'S1I3': 'Current (A)',
            'S1I5': 'Current (A)', 'S2I5': 'Current (A)', 'S34PV1': 'Power (W)', 'S34V5': 'Voltage (V)',
            '250.0': 'Intensity (a.u.)', '336.98': 'Intensity (a.u.)', '395.8': 'Intensity (a.u.)',
            '725.0': 'Intensity (a.u.)', '773.2': 'Intensity (a.u.)'
        }

        if st.button("▶️ 시작", key="start_monitoring_button"):
            # Y축 범위 및 유효 컬럼 미리 계산
            y_domains = {}
            existing_cols_map = {}
            
            def precompute_domains_and_cols(cols, group_name):
                existing = [col for col in cols if col in df_wafer.columns]
                existing_cols_map[group_name] = existing
                if not existing: return
                
                for col in existing:
                    min_val, max_val = df_wafer[col].min(), df_wafer[col].max()
                    margin = (max_val - min_val) * 0.1 if max_val > min_val else 0.5
                    y_domains[col] = [min_val - margin, max_val + margin]
            
            precompute_domains_and_cols(proc_cols, "proc")
            precompute_domains_and_cols(rfm_cols, "rfm")
            precompute_domains_and_cols(oes_cols, "oes")

            # UI 레이아웃 미리 설정
            st.markdown("#### EV(공정&설비 모니터링)")
            proc_placeholder = st.empty()
            st.markdown("#### RFM(플라즈마 발생기 모니터링)")
            rfm_placeholder = st.empty()
            st.markdown("#### OES(플라즈마 파장 모니터링)")
            oes_placeholder = st.empty()

            placeholder_map = {"proc": proc_placeholder, "rfm": rfm_placeholder, "oes": oes_placeholder}
            
            # 성능 최적화를 위해 Figure와 Axes 객체를 루프 밖에서 한 번만 생성
            figs, axes_map = {}, {}
            for group in ["proc", "rfm", "oes"]:
                existing_vars = existing_cols_map.get(group, [])
                if existing_vars:
                    num_cols = len(existing_vars)
                    fig, axes = plt.subplots(1, num_cols, figsize=(3 * num_cols, 3))
                    axes = [axes] if num_cols == 1 else axes
                    figs[group], axes_map[group] = fig, axes
            
            def create_animated_plot(data, col, y_domain, color, ax, title, y_label):
                ax.clear()  # 기존 축을 지우고 새로 그림
                ax.plot(data.index, data[col], color=color, linewidth=2)
                ax.set_title(title, fontsize=14)
                ax.set_ylabel(y_label)
                if y_domain: ax.set_ylim(y_domain)
                ax.grid(True, linestyle='--', alpha=0.6)

            # 애니메이션 루프
            for i in range(1, len(df_wafer) + 1):
                current_data = df_wafer.iloc[:i]
                
                # 각 섹션 차트 업데이트
                for group, color in [("proc", "royalblue"), ("rfm", "green"), ("oes", "darkorange")]:
                    if group in figs:
                        fig, axes = figs[group], axes_map[group]
                        existing_vars = existing_cols_map[group]
                        
                        for idx, var in enumerate(existing_vars):
                            plot_title = oes_title_map.get(var, var) if group == "oes" else var
                            y_label = unit_map.get(var, 'Value')
                            create_animated_plot(current_data, var, y_domains.get(var), color, axes[idx], title=plot_title, y_label=y_label)
                        
                        fig.tight_layout()
                        placeholder_map[group].pyplot(fig)
                
                time.sleep(0.01) # 애니메이션 속도 조절

            # 루프 종료 후 모든 Figure 객체 메모리 해제
            for fig in figs.values():
                plt.close(fig)


# ===============================
# 3) 모델 결과 분석
# ===============================
elif page == "모델 결과 분석":
    st.title("모델 결과 분석")
    tab1, tab2 = st.tabs(["결함 유형 분류 모델", "엔드포인트 예측 모델"])

    # ---------- Tab1: 결함 분류 ----------
    with tab1:
        st.subheader("📌 결함 유형 분류 모델")
        df_merged, load_msg = load_fault_data()
        st.caption(load_msg)

        group_choice = st.selectbox("웨이퍼 그룹 선택", ["전체", "29", "31", "33"], index=0)
        df_sel = make_features_by_group(df_merged, group_choice)

        if df_sel.empty:
            st.warning("선택한 그룹에 해당하는 데이터가 없습니다.")
            st.stop()

        results_file = "ml_results_all.pkl"

        # 모든 그룹 학습 후 저장 (최초 1회만 실행)
        if not Path(results_file).exists():
            results = {}
            for g in ["전체", "29", "31", "33"]:
                df_g = make_features_by_group(df_merged, g)
                if not df_g.empty:
                    with st.spinner(f"{g} 그룹 모델 학습 중..."):
                        results[g] = train_and_explain(df_g)
            with open(results_file, "wb") as f:
                pickle.dump(results, f)
            st.success("모든 그룹의 모델 학습 및 저장 완료 ✅")

        # 저장된 결과 불러오기
        with open(results_file, "rb") as f:
            results = pickle.load(f)

        # 현재 선택한 그룹의 결과 꺼내오기
        ctx = results.get(group_choice)
        if ctx is None:
            st.error(f"{group_choice} 그룹 모델 결과가 없습니다.")
            st.stop()

        report = ctx["report"]
        y_test, pred_label = ctx["y_test"], ctx["pred_label"]

        macro = report.get("macro avg", {}); accuracy = report.get("accuracy", float(np.mean(y_test==pred_label)))
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Macro F1-score", f"{macro.get('f1-score',0):.4f}")
        c2.metric("Macro Precision", f"{macro.get('precision',0):.4f}")
        c3.metric("Macro Recall", f"{macro.get('recall',0):.4f}")
        c4.metric("Accuracy", f"{accuracy:.4f}")
        sv = ctx["sv"]; X_test_scaled_df = ctx["X_test_scaled_df"]; feature_names = ctx["feature_names"]; target_names = ctx["target_names"]
        st.markdown("---")
        st.markdown("#### Feature Importance (Top-10)")
        st.caption("💡 각 결함 유형별로 색상이 구분되어 표시됩니다. 막대의 길이는 해당 특징의 중요도를 나타냅니다.")
        fig_bar = plt.figure(figsize=(12, 7))
        shap.summary_plot(sv, features=X_test_scaled_df, feature_names=feature_names,
                          plot_type="bar", class_names=target_names, max_display=10, show=False)
        #plt.title("SHAP Feature Importance (Top-10)", fontsize=14, pad=20)
        legend = plt.gca().get_legend()
        if legend:
            legend.set_bbox_to_anchor((1.05, 1))
            plt.legend(fontsize=7)
        cbar = plt.gcf().get_axes()[-1] if len(plt.gcf().get_axes()) > 1 else None
        if cbar and hasattr(cbar, 'tick_params'):
            cbar.tick_params(labelsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=10)
        plt.xlabel(plt.gca().get_xlabel(), fontsize=8)
        plt.ylabel(plt.gca().get_ylabel(), fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_bar, use_container_width=True)
        
        st.markdown("---"); st.markdown("### 결함 유형별 상세 성능 & 해석")
        fault_choice = st.selectbox("결함 유형 선택", target_names, index=0); ci = target_names.index(fault_choice)
        cls_metrics = report.get(fault_choice, {}); c1,c2,c3 = st.columns(3)
        c1.metric(f"[{fault_choice}] F1-score", f"{cls_metrics.get('f1-score',0):.4f}")
        c2.metric(f"[{fault_choice}] Precision", f"{cls_metrics.get('precision',0):.4f}")
        c3.metric(f"[{fault_choice}] Recall", f"{cls_metrics.get('recall',0):.4f}")
        from collections import Counter
        mask_ci = (y_test==ci); wrong_pred = pred_label[mask_ci & (pred_label!=ci)]
        if len(wrong_pred)==0:
            st.success(f"'{fault_choice}' 클래스에 대한 오분류가 없습니다. 🎉")
        else:
            mis_map = Counter(wrong_pred)
            mis_table = pd.DataFrame({"예측 클래스":[target_names[k] for k in mis_map.keys()],
                                      "건수": list(mis_map.values())}).sort_values("건수", ascending=False)
            st.markdown("#### 오분류표")
            st.caption("💡 선택한 결함 유형에 대해 어떤 다른 유형으로 잘못 예측했는지를 보여줍니다. ")
            st.dataframe(mis_table, use_container_width=True)
        st.markdown("#### SHAP Beeswarm ")
        st.caption("💡 각 점은 하나의 테스트 샘플을 나타냅니다. x축은 SHAP 값, 색상은 특징 값(빨강: 높음, 파랑: 낮음)을 의미합니다.")
        vals = sv.values
        if len(vals.shape) == 3:
            sv_c = shap.Explanation(
                values=vals[:,:,ci],
                base_values=sv.base_values[:,ci] if np.ndim(sv.base_values)>1 else sv.base_values,
                data=sv.data, feature_names=feature_names
            )
        else:
            sv_c = shap.Explanation(
                values=vals,
                base_values=sv.base_values,
                data=sv.data, feature_names=feature_names
            )
        fig_bee = plt.figure(figsize=(12, 7))
        shap.plots.beeswarm(sv_c, max_display=10, show=False)
        #plt.title(f"SHAP Beeswarm - {fault_choice}", fontsize=14, pad=20)
        cbar = plt.gcf().get_axes()[-1] if len(plt.gcf().get_axes()) > 1 else None
        if cbar and hasattr(cbar, 'tick_params'):
            cbar.tick_params(labelsize=8)
            if hasattr(cbar, 'set_ylabel'):
                cbar.set_ylabel('Feature value', fontsize=9)
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig_bee, use_container_width=True)

        # llm_payload = create_beeswarm_payload(sv_c, fault_choice, max_display=10)  # joy
        llm_payload = create_summarized_beeswarm_payload(sv_c, fault_choice, max_display=10)  # joy
        # with open('llm_payload.pkl', 'wb') as f:  # joy (디버깅용) -- 위 함수의 결과를 파일로 저장하는 부분. 
        #     pickle.dump(llm_payload, f)
        # st.write(json.dumps(llm_payload, indent=2)) -- 위 함수의 결과를 streamlit 앱에 출력하는 부분.
        import openai
        client = openai.OpenAI(api_key="")
        prompt = f"""
                # [역할]

                너는 LAM Research의 플라즈마 식각 장비 전문가이자, 반도체 수율 및 공정 제어를 총괄하는 최고의 엔지니어(Principal Engineer)다. 너의 임무는 다중 결함유형 분류 모델의 SHAP 분석 결과를 해석하여, 특정 공정 불량의 근본 원인을 물리적/화학적 관점에서 진단하고, 현장 엔지니어들이 즉시 실행할 수 있는 구체적인 Action Item을 제시하는 것이다.

                # [목표]

                아래에 제공된 선택한 타겟 결함(target_class_name)에 대한 SHAP Beeswarm 요약(상위 특성, 구간별 통계) 데이터를 분석하여, 경영진과 동료 엔지니어들을 위한 기술 진단 보고서를 작성한다. 보고서는  **"1. 요약"****"2. 근본 원인 분석"**과 **"3. 조치 계획"**이라는 명확한 두 개의 섹션으로 구성되어야 한다. 

                # [Process & Data Context]

                - **공정 개요**: LAM사의 플라즈마 식각 장비를 활용한 알루미늄 스택(Al-0.5wt.%Cu / TiN / Oxide) 식각 공정이다. TCP(Transformer Coupled Plasma) 소스를 사용하며, 공정은 가스 안정화, 플라즈마 점화, (Al)주 식각, (TiN/산화물)과식각, 배기의 6단계로 구성된다.
                -실험 설계(결함 유형)
                TCP 전력, RF 전력, 압력, Cl₂/BCl₃ 유량, He 척 압력 등을 변경하여 결함을 의도적으로 유도한 세 차례 실험(29, 31, 33)을 수행
                
                리셋 방식 분석
                의도적으로 변경한 제어 변수는 데이터에서 평균을 정상값으로 되돌려 리셋
                ex) TCP+50: TCP 전력을 350→400 W로 변경했을 때, 데이터 파일의 평균을 다시 350 W로 맞춰 처리(예시일 뿐 실제 평균 및 변경한 값은 뭔지 모름. 정상값(설정값)에서 몇 올렸는지만 알고 있다.)
                결과적으로 제어 변수는 정상처럼 보이지만, 다른 변수와의 관계 왜곡으로 결함 영향이 드러나도록 설계
                - 결함 유형 분류 모델: 모델은 변경된 변수 자체가 아닌, 다른 변수들에서 나타나는 이상 패턴을 통해 결함을 탐지하도록 평가됨.
                - **데이터 소스**: 식각 장비의 세가지 센서 시스템에서 수집된 세 종류의 데이터를 통합했다. `fault_name`은 불량 유형을 나타내는 가장 중요한 타겟 변수이다.

                1.**EV (Engineering Variables)**: 에칭 장비 자체를 **제어하는 설정값**과 장비의**상태**를 직접 측정한 기본 설비/공정 변수.

                변수: 

                "BCl3 Flow": "BCl₃ 유량", "Cl2 Flow": "Cl₂ 유량",
                "RF Btm Pwr": "RF 하부 전력", "RF Btm Rfl Pwr": "RF 하부 반사파 전력",
                "RF Pwr": "RF 전력", "RF Phase Err": "RF 위상 오차",
                "RF Impedance": "RF 임피던스", "RF Tuner": "RF 튜너 위치",
                "RF Load": "RF 로드", "TCP Top Pwr": "TCP 상부 전력",
                "TCP Rfl Pwr": "TCP 반사파 전력", "TCP Phase Err": "TCP 위상오차",
                "TCP Impedance": "TCP 임피던스", "TCP Tuner": "TCP 튜너 위치",
                "TCP Load": "TCP 로드", "He Press": "헬륨 척 압력",
                "Pressure": "챔버 압력", "Vat Valve": "배기 밸브 개도",
                "Endpt A": "엔드포인트 검출 신호 A",

                2. **RFM (Radio Frequency Monitoring)**

                -플라즈마를 발생시키는 RF 생성기의 전력 및 위상 관계를 측정.

                -플라스마 공정에서는 기본 주파수 전원이 인가될 때 플라스마 내부의 비선형적인 반응 때문에 다양한 고조파가 자연스럽게 생성됨
                -고조파 정보는 기본 주파수만으로는 포착하기 어려운 미묘한 **공정 이상이나 결함을 감지**하는 데 중요한 역할을 함
                

                변수: ‘센서 위치 + 전압/전류/위상 + 주파수’ 형태로 되어있음 
                    ex) S1V2 : S1 지점의 제2 고조파의 전압

                    S2I1 : S2 지점 제1 고조파(기본주파수)의  전류

                S34PV5: S3 & S4 간 제5 고조파의 전압 위상차

                **** S34I3: S3+S4 센서간 제3 고조파에 대한 전류 합성값

                3. **OES (Optical Emission Spectroscopy)**

                - 플라즈마 내 화학종의 광학 방출 스펙트럼을 측정.
                - 금속 식각 장비는**세 개의 서로 다른 위치**에 OES 센서를 장착하고 있다. 이 센서들은 각기 다른 시야각(viewing port)을 통해 플라스마를 측정한다. **공정 가스와 웨이퍼에서 발생하는 화학종**에 해당하는 43개의 핵심 피크를 통합하여 분석에 활용한다.

                변수:  **'725.0'** 처럼 **숫자(파장대, nm)**로 시작한다. 43개의 파장 컬럼(250.0~791.5)이 3번 반복되어 나타나는데, 이는 각 센서(3개 위치)가 43개 파장의 데이터를 측정했음을 의미. 
                    
                    
                ex) 261.8, 261.8.1, 261.8.2 → 파장값 뒤에 붙은 .1, .2 로 센서 위치를 구분한다.
                식각에 참여하는 물질(원소)별 고유 방출 파장대
                *참고) 식각 대상: 알루미늄 스택(Al-0.5wt.%Cu / TiN / Oxide) 
                 - '395.8': Al (알루미늄) - 식각 대상(주식각)
                - '336.98': Ti (티타늄) - 식각 대상(과식각)
                - '250.0': Si (실리콘) - 식각 대상(과식각)
                - '725.0': Cl (염소) - 식각 물질(Cl2 식각 가스)
                - '781.0': O (산소) - 식각 대상(과식각)
                - '748.5': N (질소) - 식각 대상(과식각)
                - '324.8': Cu (구리) - 식각 대상(주식각)

                # [AI의 사고 과정 (Thought Process)]

                너는 보고서를 작성하기 위해 내부적으로 다음 4단계의 사고 과정을 거쳐야 한다.

                1.**핵심 변수별 통계적 관계 해석**: 상위 4개 중요 변수에 대해, 변수 값(Feature Value)과 불량 기여도(SHAP Value) 사이의 명확한 패턴을 파악한다. ("이 변수의 값이 높을 때/낮을 때, 불량 가능성을 높이는가, 낮추는가?")

                2. **공학적 인과관계 추론 (가설 수립)**: 1단계의 통계적 관계를 실제 Al/TiN 식각 공정의 물리/화학적 메커니즘과 연결한다. ("이 현상은 왜 발생하는가? 플라즈마 밀도, 이온 에너지, 가스 해리, 식각 부산물, 챔버 컨디션 등 어떤 물리적 변화와 연관되는가?")

                3. **종합적인 근본 원인 진단**: 각 변수별 분석을 종합하여, 불량을 유발한 가장 가능성 높은 단일 혹은 복합 시나리오를 구성한다. ("모든 단서를 종합했을 때, 가장 설득력 있는 불량 발생 스토리는 무엇인가?")

                4. **단계별 조치 계획 수립**: 진단된 근본 원인을 해결하기 위한 구체적이고 실행 가능한 계획을 단기/중장기로 구분하여 수립한다.

                # [보고서 출력 구조 (Output Structure)]

                위 사고 과정을 바탕으로, 최종 보고서는 반드시 아래 세 개의 섹션으로 나누어 명확하고 간결하게 작성하라.\
                -각 소제목은 bold체 + 조금 크게 작성
                - 말투는 '~습니다' 체로 고정 
                - 보고서 제목은 따로 생성하지 않음
                -Quantile 1/4' 같은 표현은 쓰지 말고, **"하위 25% 구간"** 처럼 사람 친화적으로 바꾸세요.
                -근본 원인 분석 및 조치 계획을 작성할 때 [Process & Data Context] 내용을 참고해서 도메인적 지식을 기반으로 논리적으로 생각.
                -여기서 말하는 결함유형(타겟)은 웨이퍼 상 또는 식각에서의 불량이 발생했다기보다는 특정 제어변수를 기본 센서 교정값보다 크거나 작게 의도적으로 변경했을 때 다른 변수(인자)들로 그 변경(의도적 결함)을 탐지할 수 있는지를 말하는거야. 
                따라서, **해당 타겟 결함이 생기면(예: BCl3 +10, BCl3 가스 유량을 10sccm 높게 설정하면) 주요 인자(변수)들이 증가/감소 할 수 있고 그럼 이러한 문제가 생길 수 있다**라는 플로우로 근본 원인 분석 및 조치사항을 작성해줘.
                -OES데이터의 파장대 같은 경우 [Process & Data Context] 설명에 이 파장대에 해당하는 물질이 나와있으면 그 정보 활용해줘.
                1. 요약

                아래 형식을 토대로 간단하게 bullet형태로 작성, 데이터 소스에 있는 변수에 대한 설명 및 도메인 지식을 조사해서 변수에 대한 설명을 간단하게 작성한다.
                *요약 부분만 개조식으로 작성(명사로 끝나도록)

                - **타겟 결함**: {llm_payload['target_class_name']} 
                    타겟 결함에 대한 간단한 설명(예: BCl3 +10, BCl3 가스 유량을 10sccm 높게 설정)

                -**핵심 유발 요인 TOP 3**: 
                1~3. 변수명: 변수에 대한 간단한 도메인 기반 설명 / 해당 변수가 감소 or 증가할 때 해당 결함이 발생할 확률이 높아지는지 

                ---

                2. 근본 원인 분석 

                이 섹션에서는 [AI의 사고 과정] 1, 2, 3단계에서 도출한 내용을 종합하여 작성한다.

                **[분석]**
                    각 핵심 변수의 데이터 패턴(e.g., 특정 값 구간에서의 SHAP 값 변화)이 실제 공정에서 어떤 물리적, 화학적 현상(e.g., 이온 에너지 증가, 특정 라디칼 밀도 감소)을 의미하는지 공학적으로 해석한다.
                    핵심 유발 요인 TOP 3 순서대로 1.2.3 번호를 매겨 작성
                **[결론]**
                    위의 분석들을 종합하여 해당 불량을 유발한 최종적인 원인을 하나의 통합된 시나리오로 결론 내린다.

                3. 조치 계획 

                이 섹션에서는 [AI의 사고 과정] 4단계 결과를 바탕으로 작성한다. 현장 엔지니어가 바로 실행할 수 있도록 명확한 행동 중심으로 기술한다.

                **[단기 조치: 즉시 확인 및 대응]**

                -  확인 항목: 즉시 분석해야 할 장비 데이터 로그 및 웨이퍼 계측 항목을 구체적으로 명시한다.

                -  긴급 조치: 추가적인 불량 발생을 막기 위한 즉각적인 조치 사항을 제시한다.

                **[중장기 조치: 개선 및 예방]**

                -  공정 레시피 최적화: 재발 방지를 위해 수정 또는 최적화가 필요한 공정 변수와 목표 값을 제시한다.

                -  장비 관리 및 모니터링 강화: 향후 유사 문제를 조기에 감지하기 위한 모니터링 강화 방안 또는 장비 유지보수(PM) 개선 항목을 제안한다.

---




        {json.dumps(llm_payload, indent=2)}"""
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content": prompt}],
            temperature=0.1,
            max_tokens=1000)
        with st.expander("SHAP 분석"):
            st.write(resp.choices[0].message.content)



    # ---------- Tab2: 엔드포인트 ----------
    with tab2:
        st.subheader("📌 엔드포인트 예측 모델")

        df_ep, ep_msg = load_endpoint_data()

        # ------------------------------
        # ✅ 모델 최초 1회만 학습 후 저장 / 이후에는 불러오기
        # ------------------------------
        results_file = "ml_results_endpoint.pkl"

        if not Path(results_file).exists():
            st.info("최초 실행: 모든 그룹의 모델을 학습하고 저장합니다.")
            results = {}
            for g in ["전체", "29", "31", "33"]:
                df_g = make_features_by_group(df_ep, g)
                if not df_g.empty:
                    with st.spinner(f"{g} 그룹 모델 학습 중..."):
                        results[g] = train_and_explain(df_g)
            with open(results_file, "wb") as f:
                pickle.dump(results, f)
            st.success("모든 그룹의 모델 학습 및 저장 완료 ✅")

        # 저장된 결과 불러오기
        with open(results_file, "rb") as f:
            results = pickle.load(f)

        try:
            with st.spinner("모델 학습 중..."):
                ep_ctx = build_endpoint_model(df_ep)
        except Exception as e:
            st.error(f"엔드포인트 모델 준비 중 오류: {e}")
            st.stop()

        cls1 = ep_ctx["cls1"]; acc = ep_ctx["accuracy"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1-score (1)", f"{cls1.get('f1-score', 0):.4f}")
        c2.metric("Precision (1)", f"{cls1.get('precision', 0):.4f}")
        c3.metric("Recall (1)", f"{cls1.get('recall', 0):.4f}")
        c4.metric("Accuracy", f"{acc:.4f}")

        wafer_list = ep_ctx["wafer_list"]
        if len(wafer_list) == 0:
            st.warning("생성된 샘플이 없습니다. 데이터 스키마를 확인하세요.")
            st.stop()

        sel_wafer = st.selectbox("웨이퍼 선택", wafer_list, index=0, key="ep_wafer")

        X_super = ep_ctx["X_super"]; y_super = ep_ctx["y_super"]
        time_super = ep_ctx["time_super"]; wid_super = ep_ctx["wid_super"]
        F = ep_ctx["F"]; cb = ep_ctx["model"]

        mask_full = (wid_super.astype(str) == str(sel_wafer))
        if mask_full.sum() == 0:
            st.warning("선택한 웨이퍼의 샘플이 없습니다.")
            st.stop()

        Xw = X_super[mask_full]
        tw = time_super[mask_full]
        yw = y_super[mask_full]
        yhat_w = cb.predict(Xw).ravel().astype(int)

        cur_block = Xw[:, -F:]
        signal = cur_block.mean(axis=1)

        def idx_to_time(tarr, idx):
            return float(tarr[idx]) if idx is not None and 0 <= idx < len(tarr) else None

        def first_one(arr: np.ndarray):
            pos = np.where(arr == 1)[0]
            return int(pos[0]) if len(pos) > 0 else None

        true_ep_idx = first_one(yw)
        pred_ep_idx = first_one(yhat_w)
        true_t = idx_to_time(tw, true_ep_idx)
        pred_t = idx_to_time(tw, pred_ep_idx)
        delta = (pred_t - true_t) if (true_t is not None and pred_t is not None) else None

        col_plot, col_stats = st.columns([3, 1])
        with col_plot:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(tw, signal, label='Mean of current-frame features (std-scaled)')
            if true_t is not None:
                ax.axvline(true_t, color='g', linestyle='-', linewidth=2, label='True EP')
            if pred_t is not None:
                ax.axvline(pred_t, color='r', linestyle='--', linewidth=2, label='Pred EP')

            def add_value_only_label(ax, xv, color):
                ymin, ymax = ax.get_ylim()
                y_text = ymin + 0.03 * (ymax - ymin)
                ax.text(xv, y_text, f"{float(xv):.1f}s", color=color, ha='center', va='bottom', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

            if true_t is not None: add_value_only_label(ax, true_t, "g")
            if pred_t is not None: add_value_only_label(ax, pred_t, "r")
            #plt.title(f"Wafer: {sel_wafer} | Endpoint Prediction (CatBoost + lag features, frame-wise)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Mean signal (standardized)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with col_stats:
            st.markdown("#### ⏱️ EP 시점")
            st.metric("실제 EP", f"{true_t:.1f}s" if true_t is not None else "N/A")
            st.metric("예측 EP", f"{pred_t:.1f}s" if pred_t is not None else "N/A")
            st.metric("편차(예측-실제)", f"{delta:.1f}s" if delta is not None else "N/A")

        st.markdown("#### SHAP Beeswarm")
        st.caption("💡 각 점은 하나의 테스트 샘플을 나타냅니다. x축은 SHAP 값, 색상은 특징 값(빨강: 높음, 파랑: 낮음)을 의미합니다.")
        sv = ep_ctx["sv"]
        feat_names = ep_ctx["feat_names_lag"]

        sv_exp = shap.Explanation(
            values=sv.values if hasattr(sv, 'values') else sv,
            base_values=getattr(sv, 'base_values', np.zeros(len(ep_ctx["X_test"]))),
            data=ep_ctx["X_test"],
            feature_names=feat_names
        )

        llm_payload2 = create_summarized_beeswarm_payload(sv_exp, None, max_display=10)
        #with open('llm_payload2.pkl', 'wb') as f:   # 디버깅용
            #pickle.dump(llm_payload2, f)
            #st.write(json.dumps(llm_payload2, indent=2)) 
        fig_bee = plt.figure(figsize=(10, 6))
        shap.summary_plot(sv_exp, ep_ctx["X_test"], feature_names=feat_names, max_display=10, show=False)
        #plt.title("SHAP Beeswarm (Endpoint model, Top-10)")
        cbar = plt.gcf().get_axes()[-1] if len(plt.gcf().get_axes()) > 1 else None
        if cbar and hasattr(cbar, 'tick_params'):
            cbar.tick_params(labelsize=8)
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig_bee, use_container_width=True)
        

        client = openai.OpenAI(api_key="")
        prompt = f"""
                # [역할]

                너는 LAM Research의 플라즈마 식각 장비 전문가이자, 반도체 수율 및 공정 제어를 총괄하는 최고의 엔지니어(Principal Engineer)다. 너의 임무는 다중 엔드포인트 예측 모델의 SHAP 분석 결과를 해석하여, 특정 공정 엔드포인트 오차의 근본 원인을 물리적/화학적 관점에서 진단하고, 현장 엔지니어들이 즉시 실행할 수 있는 구체적인 Action Item을 제시하는 것이다.

                # [목표]

                아래에 제공된 엔드포인트 예측 모델의 SHAP Beeswarm 요약(상위 특성, 구간별 통계) 데이터를 분석하여, 경영진과 동료 엔지니어들을 위한 기술 진단 보고서를 작성한다. 보고서는 **"1. 요약"**, **"2. 근본 원인 분석"**과 **"3. 조치 계획"**이라는 명확한 세 개의 섹션으로 구성되어야 한다.
                
                # [Process & Data Context]

                - **공정 개요**: LAM사의 플라즈마 식각 장비를 활용한 알루미늄 스택(Al-0.5wt.%Cu / TiN / Oxide) 식각 공정이다. TCP(Transformer Coupled Plasma) 소스를 사용하며, 공정은 가스 안정화, 플라즈마 점화, (Al)주 식각, (TiN/산화물)과식각, 배기의 6단계로 구성된다.

                - **모델 특징 (Lag Feature)**: 엔드포인트 예측 모델은 현재 시점의 데이터뿐만 아니라 **Lag feature(과거 시점 데이터)**를 종합하여 판단합니다. 피처 이름에 **'lag'**가 붙은 변수(예: 'Cl2 Flow\_lag\_5')는 **현재 시점 기준 5 프레임 이전의 데이터**가 현재 EP 예측에 영향을 미쳤음을 의미합니다. 따라서 분석 시 **시간 경과에 따른 공정 드리프트(Drift)** 관점에서 해석해야 합니다.

                - **SHAP 값 해석**: SHAP 값이 **양수(+)**이면 EP 예측(클래스 1)을 **가속화/촉진**하고, **음수(-)**이면 EP 예측을 **지연/방해**한다.

                - **데이터 소스**: 식각 장비의 세가지 센서 시스템에서 수집된 세 종류의 데이터를 통합했다. 
                
                1.**EV (Engineering Variables)**: 에칭 장비 자체를 **제어하는 설정값**과 장비의**상태**를 직접 측정한 기본 설비/공정 변수.

                변수: 

                "BCl3 Flow": "BCl₃ 유량", "Cl2 Flow": "Cl₂ 유량",
                "RF Btm Pwr": "RF 하부 전력", "RF Btm Rfl Pwr": "RF 하부 반사파 전력",
                "RF Pwr": "RF 전력", "RF Phase Err": "RF 위상 오차",
                "RF Impedance": "RF 임피던스", "RF Tuner": "RF 튜너 위치",
                "RF Load": "RF 로드", "TCP Top Pwr": "TCP 상부 전력",
                "TCP Rfl Pwr": "TCP 반사파 전력", "TCP Phase Err": "TCP 위상오차",
                "TCP Impedance": "TCP 임피던스", "TCP Tuner": "TCP 튜너 위치",
                "TCP Load": "TCP 로드", "He Press": "헬륨 척 압력",
                "Pressure": "챔버 압력", "Vat Valve": "배기 밸브 개도",
                "Endpt A": "엔드포인트 검출 신호 A",

                2. **RFM (Radio Frequency Monitoring)**

                -플라즈마를 발생시키는 RF 생성기의 전력 및 위상 관계를 측정.

                -플라스마 공정에서는 기본 주파수 전원이 인가될 때 플라스마 내부의 비선형적인 반응 때문에 다양한 고조파가 자연스럽게 생성됨
                -고조파 정보는 기본 주파수만으로는 포착하기 어려운 미묘한 **공정 이상이나 결함을 감지**하는 데 중요한 역할을 함
                

                변수: ‘센서 위치 + 전압/전류/위상 + 주파수’ 형태로 되어있음 
                    ex) S1V2 : S1 지점의 제2 고조파의 전압

                    S2I1 : S2 지점 제1 고조파(기본주파수)의  전류

                S34PV5: S3 & S4 간 제5 고조파의 전압 위상차

                **** S34I3: S3+S4 센서간 제3 고조파에 대한 전류 합성값

                3. **OES (Optical Emission Spectroscopy)**

                - 플라즈마 내 화학종의 광학 방출 스펙트럼을 측정.
                - 금속 식각 장비는**세 개의 서로 다른 위치**에 OES 센서를 장착하고 있다. 이 센서들은 각기 다른 시야각(viewing port)을 통해 플라스마를 측정한다. **공정 가스와 웨이퍼에서 발생하는 화학종**에 해당하는 43개의 핵심 피크를 통합하여 분석에 활용한다.

                변수:  **'725.0'** 처럼 **숫자(파장대, nm)**로 시작한다. 43개의 파장 컬럼(250.0~791.5)이 3번 반복되어 나타나는데, 이는 각 센서(3개 위치)가 43개 파장의 데이터를 측정했음을 의미. 
                    
                    
                ex) 261.8, 261.8.1, 261.8.2 → 파장값 뒤에 붙은 .1, .2 로 센서 위치를 구분한다.
                식각에 참여하는 물질(원소)별 고유 방출 파장대
                *참고) 식각 대상: 알루미늄 스택(Al-0.5wt.%Cu / TiN / Oxide) 
                 - '395.8': Al (알루미늄) - 식각 대상(주식각)
                - '336.98': Ti (티타늄) - 식각 대상(과식각)
                - '250.0': Si (실리콘) - 식각 대상(과식각)
                - '725.0': Cl (염소) - 식각 물질(Cl2 식각 가스)
                - '781.0': O (산소) - 식각 대상(과식각)
                - '748.5': N (질소) - 식각 대상(과식각)
                - '324.8': Cu (구리) - 식각 대상(주식각)
                - '669.5' : Al (알루미늄) - 식각 대상(주식각) (강한 피크는 아님)
                - '261.8' : Ti (티타늄) - 식각 대상(과식각) (강한 피크는 아님)
                
                ex) 각 파장대의 값을 이야기할 때 669.5 파장이면 669.5 파장대이면 669.5(Al) 이런 식으로 파장대 숫자 먼저 작성한 후 괄호 안에 물질(원소)을 작성해서 설명해줘.

                # [AI의 사고 과정 (Thought Process)]

                너는 보고서를 작성하기 위해 내부적으로 다음 4단계의 사고 과정을 거쳐야 한다.
                
                1.**핵심 변수별 통계적 관계 해석**: 상위 4개 중요 변수에 대해, 제공된 JSON의 'value_shap_summary' 통계를 분석하여, 해당 변수의 값(feature_value)이 낮은 구간(min)과 높은 구간(max)에서의 SHAP 평균값(shap_mean)을 비교한다. 이를 통해 '변수 값이 높을 때' 또는 '낮을 때' 중 어느 쪽이 EP 예측(클래스 1)을 가장 강하게 **가속화(+ SHAP)**하거나 **지연(- SHAP)**시키는지 **구체적인 조건과 결과**를 도출한다.
                
                2. **공학적 인과관계 추론 (가설 수립)**: 1단계의 통계적 관계를 실제 Al/TiN 식각 공정의 물리/화학적 메커니즘과 연결한다. **특히 Lag Feature가 포함된 경우**, 그 과거 데이터가 현재의 EP 예측을 어떻게 변화시키는지(ex: 과거의 챔버 오염이 현재의 식각 속도를 높임)에 대한 **시간적 인과관계를 추론**한다. ("이 현상은 왜 발생하는가? 플라즈마 밀도, 이온 에너지, 가스 해리, 식각 부산물, 챔버 컨디션 변화 등 어떤 물리적 변화와 연관되어 EP 시점을 앞당기거나 늦추는가?")

                3. **종합적인 근본 원인 진단**: 각 변수별 분석을 종합하여, EP 시점을 통제하는 가장 가능성 높은 단일 혹은 복합 시나리오를 구성한다. ("모든 단서를 종합했을 때, 가장 설득력 있는 EP 시점 변동 스토리는 무엇인가?")

                4. **단계별 조치 계획 수립**: 진단된 근본 원인을 해결하기 위한 구체적이고 실행 가능한 계획을 단기/중장기로 구분하여 수립한다.

                
                
                # [보고서 출력 구조 (Output Structure)]

                위 사고 과정을 바탕으로, 최종 보고서는 반드시 아래 세 개의 섹션으로 나누어 명확하고 간결하게 작성하라.
                -각 소제목은 bold체 + 조금 크게 작성
                - 말투는 '~습니다' 체로 고정 
                -근본 원인 분석 및 조치 계획을 작성할 때 [Process & Data Context] 내용을 참고해서 도메인적 지식을 기반으로 논리적으로 생각.
                -여기서 말하는 결함유형(타겟)은 웨이퍼 상 또는 식각에서의 불량이 발생했다기보다는 특정 제어변수를 기본 센서 교정값보다 크거나 작게 의도적으로 변경했을 때 다른 변수(인자)들로 그 변경(의도적 결함)을 탐지할 수 있는지를 말하는거야. 
                따라서, **해당 타겟 결함이 생기면(예: BCl3 +10, BCl3 가스 유량을 10sccm 높게 설정하면) 주요 인자(변수)들이 증가/감소 할 수 있고 그럼 이러한 문제가 생길 수 있다**라는 플로우로 근본 원인 분석 및 조치사항을 작성해줘.
                -OES데이터의 파장대 같은 경우 [Process & Data Context] 설명에 이 파장대에 해당하는 물질이 나와있으면 그 정보 활용해줘.
                1. 요약

                아래 형식을 토대로 간단하게 bullet형태로 작성, 데이터 소스에 있는 변수에 대한 설명 및 도메인 지식을 조사해서 변수에 대한 설명을 간단하게 작성한다.
                *요약 부분만 개조식으로 작성(명사로 끝나도록)

                - **분석 목표**: 엔드포인트 예측 모델의 주요 변동 요인 파악 및 EP 시점 통제 방안 모색
                - **분석 대상 웨이퍼**: {sel_wafer} (주요 요인 분석은 전체 데이터 기반)

                
                -**EP 예측 시점 가속/지연 요인 TOP 3**: 
                **반드시** 아래 '요약 데이터'의 `"top_features_summary"` 목록에서 **상위 3개 항목**을 추출하여, 각 항목의 `"value_shap_summary"`를 분석한 뒤, **가장 강력한 조건**과 **영향**을 도출하여 다음 형식으로 작성합니다.
                1. **TCP Tuner\_lag0** - [변수가 가속화/지연시키는 구체적인 조건과 도메인 기반 설명] (명사형)
                2. **669.5\_lag0** - [변수가 가속화/지연시키는 구체적인 조건과 도메인 기반 설명] (명사형)
                3. **261.8.1\_lag3** - [변수가 가속화/지연시키는 구체적인 조건과 도메인 기반 설명] (명사형)
                
                ---

                2. 근본 원인 분석 

                이 섹션에서는 [AI의 사고 과정] 1, 2, 3단계에서 도출한 내용을 종합하여 작성한다.

                - **[분석]**
                    각 핵심 변수의 데이터 패턴(e.g., 특정 값 구간에서의 SHAP 값 변화)이 실제 공정에서 어떤 물리적, 화학적 현상(e.g., 이온 에너지 증가, 특정 라디칼 밀도 감소)을 의미하는지 공학적으로 해석한다.
                    핵심 유발 요인 TOP 3 순서대로 1.2.3 번호를 매겨 작성
                    
                - **[결론]**
                    ( 한 줄 띄우고 아래 내용 작성)
                    위의 분석들을 종합하여 해당 불량을 유발한 최종적인 원인을 하나의 통합된 시나리오로 결론 내린다.
                
                ---

                3. 조치 계획 

                이 섹션에서는 [AI의 사고 과정] 4단계 결과를 바탕으로 작성한다. 현장 엔지니어가 바로 실행할 수 있도록 명확한 행동 중심으로 기술한다.
                
                **[단기 조치: 즉시 확인 및 대응]**

                -  확인 항목: EP 예측 오류 가능성이 높을 때 즉시 분석해야 할 장비 데이터 로그 및 웨이퍼 계측 항목을 구체적으로 명시한다.

                -  긴급 조치: EP 시점 예측 편차를 줄이기 위한 즉각적인 조치 사항을 제시한다. (예: 해당 웨이퍼는 Hold 처리)

                **[중장기 조치: 개선 및 예방]**

                -  공정 레시피 최적화: EP 시점 변동성 재발 방지를 위해 수정 또는 최적화가 필요한 공정 변수와 목표 범위를 제시한다.

                -  장비 관리 및 모니터링 강화: 엔드포인트 예측 정확도를 높이고 향후 유사 문제를 조기에 감지하기 위한 모니터링 강화 방안 또는 장비 유지보수(PM) 개선 항목을 제안한다.

"""


        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content": prompt}],
            temperature=0.1,
            max_tokens=1000)
        with st.expander("SHAP 분석"):
            st.write(resp.choices[0].message.content)

    

