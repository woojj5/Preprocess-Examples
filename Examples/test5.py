import os
import re
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (윈도우 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==== 데이터 요약 함수들 ====
def summarize_before_preprocessing(df, name='원본 데이터'):
    print(f"=== {name} 요약 ===")
    print(f"총 행(row) 수: {len(df):,}")
    print(f"총 열(column) 수: {len(df.columns):,}")
    missing = df.isnull().sum()
    missing_ratio = (missing / len(df)) * 100
    missing_summary = pd.DataFrame({'missing_count': missing, 'missing_ratio(%)': missing_ratio})
    print("결측치 컬럼 (결측 존재 컬럼만):")
    print(missing_summary[missing_summary['missing_count'] > 0].sort_values(by='missing_ratio(%)', ascending=False))
    duplicated_count = df.duplicated().sum()
    print(f"중복 행 개수: {duplicated_count}")
    if 'device_no' in df.columns:
        print(f"총 차량(device_no) 개수: {df['device_no'].nunique():,}")
    else:
        print("device_no 컬럼 없음")
    if 'cartype' in df.columns:
        print(f"총 차종(cartype) 개수: {df['cartype'].nunique():,}")
    else:
        print("cartype 컬럼 없음")
    voltage_cols = [c for c in df.columns if 'cell_volt' in c]
    if voltage_cols:
        print("셀 전압 컬럼 일부 통계:")
        print(df[voltage_cols].describe(percentiles=[0.01,0.5,0.99]).T.head())

def summarize_after_preprocessing(df, name='전처리 후 데이터'):
    print(f"=== {name} 요약 ===")
    print(f"총 행(row) 수: {len(df):,}")
    print(f"총 열(column) 수: {len(df.columns):,}")
    missing = df.isnull().sum()
    missing_ratio = (missing / len(df)) * 100
    missing_summary = pd.DataFrame({'missing_count': missing, 'missing_ratio(%)': missing_ratio})
    print("결측치 컬럼 (결측 존재 컬럼만):")
    print(missing_summary[missing_summary['missing_count'] > 0].sort_values(by='missing_ratio(%)', ascending=False))
    duplicated_count = df.duplicated().sum()
    print(f"중복 행 개수: {duplicated_count}")
    key_cols = ['cellvolt_mean_no_zero', 'cellvolt_imbalance', 'zero_cell_count']
    present_key_cols = [c for c in key_cols if c in df.columns]
    if present_key_cols:
        print("주요 파생 컬럼 통계:")
        print(df[present_key_cols].describe().T)
    if 'device_no' in df.columns:
        print(f"총 차량(device_no) 개수: {df['device_no'].nunique():,}")
    else:
        print("device_no 컬럼 없음")
    if 'cartype' in df.columns:
        print(f"총 차종(cartype) 개수: {df['cartype'].nunique():,}")
    else:
        print("cartype 컬럼 없음")

# ==== BMS 전처리 ====
def preprocess_bms(df):
    df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
    if 'seq' in df.columns: df.drop(columns=['seq'], inplace=True)

    cell_voltage_cols = [col for col in df.columns if re.match(r'cell[_]?volt[_]?\d+', col)]
    mod_temp_cols = [col for col in df.columns if re.match(r'mod_temp[_]?\d+', col)]

    if cell_voltage_cols: df[cell_voltage_cols] = df[cell_voltage_cols].fillna(0)
    if mod_temp_cols: df[mod_temp_cols] = df[mod_temp_cols].fillna(0)

    cols_for_ffill = [c for c in df.columns if c not in cell_voltage_cols + mod_temp_cols + ['time','msg_time','device_no','measured_month','measured_ym']]
    if cols_for_ffill: df[cols_for_ffill] = df[cols_for_ffill].fillna(method='ffill')

    df.dropna(inplace=True)

    drop_dup_cols = [c for c in ['device_no', 'measured_month', 'time', 'msg_time'] if c in df.columns]
    if drop_dup_cols: df.drop_duplicates(subset=drop_dup_cols, inplace=True)

    if 'time' in df.columns: df['time'] = pd.to_datetime(df['time'], errors='coerce')
    if 'msg_time' in df.columns: df['msg_time'] = pd.to_datetime(df['msg_time'], errors='coerce')
    if 'device_no' in df.columns: df['device_no'] = df['device_no'].astype(str)

    if cell_voltage_cols:
        df['zero_cell_count'] = (df[cell_voltage_cols] == 0).sum(axis=1)
        df['has_zero_cell'] = df['zero_cell_count'] > 0
        def mean_excluding_zeros(row):
            vals = row[cell_voltage_cols]
            vals_no_zero = vals[vals != 0]
            return vals_no_zero.mean() if len(vals_no_zero) > 0 else np.nan
        df['cellvolt_mean_no_zero'] = df.apply(mean_excluding_zeros, axis=1)
        df['cellvolt_max'] = df[cell_voltage_cols].max(axis=1)
        df['cellvolt_min'] = df[cell_voltage_cols].min(axis=1)
        df['cellvolt_imbalance'] = df['cellvolt_max'] - df['cellvolt_min']
        
        df = df[df['cellvolt_min'] > 0]
        df['z_cellvolt_max'] = stats.zscore(df['cellvolt_max'].astype(float), nan_policy='omit')
        df = df[df['z_cellvolt_max'].abs() < 3]
    else:
        df['zero_cell_count'] = 0
        df['has_zero_cell'] = False
        df['cellvolt_mean_no_zero'] = np.nan
        df['cellvolt_max'] = np.nan
        df['cellvolt_min'] = np.nan
        df['cellvolt_imbalance'] = np.nan

    sort_cols = [c for c in ['device_no','time'] if c in df.columns]
    if sort_cols: df = df.sort_values(by=sort_cols)

    if 'cumul_current_chrgd' in df.columns: df['cumul_current_chrgd_diff'] = df['cumul_current_chrgd'].diff()
    if 'cumul_energy_chrgd' in df.columns: df['cumul_energy_chrgd_diff'] = df['cumul_energy_chrgd'].diff()

    if 'device_no' in df.columns:
        df['device_no_cat'] = df['device_no'].astype('category').cat.codes

    if 'cartype' not in df.columns:
        df['cartype'] = 'Unknown'

    if 'measured_month' in df.columns:
        df['measured_ym'] = df['measured_month'].astype(str)
    elif 'time' in df.columns:
        df['measured_ym'] = df['time'].dt.strftime('%Y-%m')
    else:
        df['measured_ym'] = pd.NaT

    return df.reset_index(drop=True)

# ==== GPS 전처리 ====
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def preprocess_gps(df):
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['lat','lng','speed'])
    df = df[(df['lat']>=-90) & (df['lat']<=90) & (df['lng']>=-180) & (df['lng']<=180)]
    if 'hdop' in df.columns: df = df[df['hdop']<5]
    df = df.sort_values(by=['device_no','time'])
    df['lat_shift'] = df.groupby('device_no')['lat'].shift(1)
    df['lng_shift'] = df.groupby('device_no')['lng'].shift(1)
    df['dist_km'] = haversine(df['lat_shift'], df['lng_shift'], df['lat'], df['lng'])
    df['dist_km'] = df['dist_km'].fillna(0)
    df = df[(df['speed']>=0) & (df['speed']<=150)]
    df['direction_shift'] = df.groupby('device_no')['direction'].shift(1)
    raw_delta = (df['direction'] - df['direction_shift']).abs()
    df['direction_delta'] = raw_delta.apply(lambda x: min(x,360-x) if pd.notnull(x) else 0)
    df['is_stopped'] = df['speed'] == 0
    if 'state' in df.columns: df['state_cat'] = df['state'].astype('category').cat.codes
    if 'mode' in df.columns: df['mode_cat'] = df['mode'].astype('category').cat.codes
    df['measured_ym'] = df['time'].dt.strftime('%Y-%m')
    return df.reset_index(drop=True)

# ==== 자동저장+3초뒤 자동닫히는 plot 함수 ====
def save_and_show_plot(fig, folder, fname, show_sec=3):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, fname)
    fig.savefig(path, bbox_inches='tight')
    print(f"그래프 저장 완료: {path}")
    plt.show(block=False)
    plt.pause(show_sec)
    plt.close(fig)

def plot_monthly_bms(df, save_dir='plots/bms'):
    if df is not None and 'cellvolt_mean_no_zero' in df.columns:
        monthly = df.groupby('measured_ym')['cellvolt_mean_no_zero'].mean()
        fig, ax = plt.subplots(figsize=(10,5))
        monthly.plot(marker='o', ax=ax)
        ax.set_title('BMS - 월별 0 제외 평균 cellvolt_mean_no_zero')
        ax.set_xlabel('측정 연-월'); ax.set_ylabel('평균 cellvolt_mean_no_zero (0 제외)')
        plt.xticks(rotation=45); ax.grid(True); plt.tight_layout()
        save_and_show_plot(fig, save_dir, 'bms_monthly_cellvolt_mean_no_zero.png')
    else:
        print("cellvolt_mean_no_zero 컬럼이 없어 그래프를 그릴 수 없습니다.")

def plot_monthly_gps(df, save_dir='plots/gps'):
    if df is not None and 'dist_km' in df.columns:
        monthly = df.groupby('measured_ym')['dist_km'].sum()
        fig, ax = plt.subplots(figsize=(10,5))
        monthly.plot(marker='o', color='orange', ax=ax)
        ax.set_title('GPS - 월별 총 이동거리')
        ax.set_xlabel('측정 연-월'); ax.set_ylabel('총 이동거리 (km)')
        plt.xticks(rotation=45); ax.grid(True); plt.tight_layout()
        save_and_show_plot(fig, save_dir, 'gps_monthly_total_distance.png')
    else:
        print("dist_km 컬럼이 없어 그래프를 그릴 수 없습니다.")

def plot_bms_soh_hist(df, save_dir='plots/bms'):
    if df is not None and 'soh' in df.columns:
        fig, ax = plt.subplots(figsize=(8,5))
        df['soh'].hist(bins=30, ax=ax)
        ax.set_title('BMS SOH 분포 히스토그램')
        ax.set_xlabel('SOH')
        plt.tight_layout()
        save_and_show_plot(fig, save_dir, 'bms_soh_hist.png')
    else:
        print("SOH 컬럼이 없어 히스토그램을 그릴 수 없습니다.")

# ==== 파일 탐색 및 전처리 ====
def scan_all_csv_and_preprocess(root_dir, data_type):
    dfs = []
    file_count = 0
    for dirpath, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith('.csv'):
                path_lower = dirpath.lower()
                if data_type in path_lower or os.path.basename(path_lower) == data_type:
                    fpath = os.path.join(dirpath, fname)
                    print(f"[{data_type.upper()}] {fpath} 처리 중...")
                    try:
                        df = pd.read_csv(fpath, encoding='utf-8', low_memory=False)
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(fpath, encoding='cp949', low_memory=False)
                        except Exception as e:
                            print(f"파일 읽기 실패: {fpath} ({e})")
                            continue
                    except Exception as e:
                        print(f"파일 읽기 실패: {fpath} ({e})")
                        continue
                    print(f"처리 전 컬럼: {df.columns.tolist()}, 행수: {len(df)}")
                    summarize_before_preprocessing(df, name=f'{fname} 전처리 전')
                    try:
                        if data_type == 'bms':
                            df_proc = preprocess_bms(df)
                        else:
                            df_proc = preprocess_gps(df)
                        dfs.append(df_proc)
                        file_count += 1
                    except Exception as e:
                        print(f"전처리 오류: {fpath} ({e})")
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        print(f"총 {data_type.upper()} 파일 수: {file_count}")
        summarize_after_preprocessing(all_df, name=f"전체 {data_type.upper()} 데이터 전처리 후")
        return all_df
    else:
        print(f"{data_type.upper()} 데이터 없음")
        return None

# ==== 메인 실행 ====
if __name__ == "__main__":
    root_dir = r""  # 환경에 맞게 변경

    print("\n[BMS 데이터 처리 시작]")
    bms_all = scan_all_csv_and_preprocess(root_dir, 'bms')

    print("\n[GPS 데이터 처리 시작]")
    gps_all = scan_all_csv_and_preprocess(root_dir, 'gps')

    print("\n[BMS 데이터 분석]")
    plot_monthly_bms(bms_all, save_dir='plots/bms')
    plot_bms_soh_hist(bms_all, save_dir='plots/bms')

    print("\n[GPS 데이터 분석]")
    plot_monthly_gps(gps_all, save_dir='plots/gps')
