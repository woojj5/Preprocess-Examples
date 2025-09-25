import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path
import psutil
import gc
import zipfile
import io
import re


# --- 1. 설정 영역 ---


# 현재 스크립트가 있는 디렉토리
BASE_DIR = Path(__file__).parent
# 처리된 데이터를 저장할 폴더
PROCESSED_DATA_DIR = BASE_DIR / "pickle_data"
# 차종 정보 파일 경로
CARTYPE_CSV_PATH = BASE_DIR / ".csv"


# ZIP 파일명에 포함될 경우 처리에서 제외할 키워드 목록 (대소문자 무관)
EXCLUDE_KEYWORDS = ["log", "empty"]


# 처리된 데이터 저장 폴더 생성
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# --- 2. 컬럼 정의 ---
FINAL_BMS_COLUMN_ORDER = [
    'device_no', 'measured_month', 'time', 'msg_time', 'acceptable_chrg_pw', 'acceptable_dischrg_pw', 'airbag_hwire_duty',
    'batt_coolant_inlet_temp', 'batt_fan_running', 'batt_internal_temp', 'batt_ltr_rear_temp', 'batt_pra_busbar_temp',
    'batt_pw', 'bms_running', 'cellvolt_dispersion', 'cell_volt_1', 'cell_volt_2', 'cell_volt_3', 'cell_volt_4',
    'cell_volt_5', 'cell_volt_6', 'cell_volt_7', 'cell_volt_8', 'cell_volt_9', 'cell_volt_10', 'cell_volt_11',
    'cell_volt_12', 'cell_volt_13', 'cell_volt_14', 'cell_volt_15', 'cell_volt_16', 'cell_volt_17', 'cell_volt_18',
    'cell_volt_19', 'cell_volt_20', 'cell_volt_21', 'cell_volt_22', 'cell_volt_23', 'cell_volt_24', 'cell_volt_25',
    'cell_volt_26', 'cell_volt_27', 'cell_volt_28', 'cell_volt_29', 'cell_volt_30', 'cell_volt_31', 'cell_volt_32',
    'cell_volt_33', 'cell_volt_34', 'cell_volt_35', 'cell_volt_36', 'cell_volt_37', 'cell_volt_38', 'cell_volt_39',
    'cell_volt_40', 'cell_volt_41', 'cell_volt_42', 'cell_volt_43', 'cell_volt_44', 'cell_volt_45', 'cell_volt_46',
    'cell_volt_47', 'cell_volt_48', 'cell_volt_49', 'cell_volt_50', 'cell_volt_51', 'cell_volt_52', 'cell_volt_53',
    'cell_volt_54', 'cell_volt_55', 'cell_volt_56', 'cell_volt_57', 'cell_volt_58', 'cell_volt_59', 'cell_volt_60',
    'cell_volt_61', 'cell_volt_62', 'cell_volt_63', 'cell_volt_64', 'cell_volt_65', 'cell_volt_66', 'cell_volt_67',
    'cell_volt_68', 'cell_volt_69', 'cell_volt_70', 'cell_volt_71', 'cell_volt_72', 'cell_volt_73', 'cell_volt_74',
    'cell_volt_75', 'cell_volt_76', 'cell_volt_77', 'cell_volt_78', 'cell_volt_79', 'cell_volt_80', 'cell_volt_81',
    'cell_volt_82', 'cell_volt_83', 'cell_volt_84', 'cell_volt_85', 'cell_volt_86', 'cell_volt_87', 'cell_volt_88',
    'cell_volt_89', 'cell_volt_90', 'cell_volt_91', 'cell_volt_92', 'cell_volt_93', 'cell_volt_94', 'cell_volt_95',
    'cell_volt_96', 'cell_volt_97', 'cell_volt_98', 'cell_volt_99', 'cell_volt_100', 'cell_volt_101', 'cell_volt_102',
    'cell_volt_103', 'cell_volt_104', 'cell_volt_105', 'cell_volt_106', 'cell_volt_107', 'cell_volt_108',
    'cell_volt_109', 'cell_volt_110', 'cell_volt_111', 'cell_volt_112', 'cell_volt_113', 'cell_volt_114',
    'cell_volt_115', 'cell_volt_116', 'cell_volt_117', 'cell_volt_118', 'cell_volt_119', 'cell_volt_120',
    'cell_volt_121', 'cell_volt_122', 'cell_volt_123', 'cell_volt_124', 'cell_volt_125', 'cell_volt_126',
    'cell_volt_127', 'cell_volt_128', 'cell_volt_129', 'cell_volt_130', 'cell_volt_131', 'cell_volt_132',
    'cell_volt_133', 'cell_volt_134', 'cell_volt_135', 'cell_volt_136', 'cell_volt_137', 'cell_volt_138',
    'cell_volt_139', 'cell_volt_140', 'cell_volt_141', 'cell_volt_142', 'cell_volt_143', 'cell_volt_144',
    'cell_volt_145', 'cell_volt_146', 'cell_volt_147', 'cell_volt_148', 'cell_volt_149', 'cell_volt_150',
    'cell_volt_151', 'cell_volt_152', 'cell_volt_153', 'cell_volt_154', 'cell_volt_155', 'cell_volt_156',
    'cell_volt_157', 'cell_volt_158', 'cell_volt_159', 'cell_volt_160', 'cell_volt_161', 'cell_volt_162',
    'cell_volt_163', 'cell_volt_164', 'cell_volt_165', 'cell_volt_166', 'cell_volt_167', 'cell_volt_168',
    'cell_volt_169', 'cell_volt_170', 'cell_volt_171', 'cell_volt_172', 'cell_volt_173', 'cell_volt_174',
    'cell_volt_175', 'cell_volt_176', 'cell_volt_177', 'cell_volt_178', 'cell_volt_179', 'cell_volt_180',
    'cell_volt_181', 'cell_volt_182', 'cell_volt_183', 'cell_volt_184', 'cell_volt_185', 'cell_volt_186',
    'cell_volt_187', 'cell_volt_188', 'cell_volt_189', 'cell_volt_190', 'cell_volt_191', 'cell_volt_192',
    'chrg_cable_conn', 'chrg_cnt', 'chrg_cnt_q', 'cumul_current_chrgd', 'cumul_current_dischrgd', 'cumul_energy_chrgd',
    'cumul_energy_chrgd_q', 'cumul_pw_chrgd', 'cumul_pw_dischrgd', 'drive_motor_spd_1', 'drive_motor_spd_2',
    'emobility_spd', 'est_chrg_time', 'ext_temp', 'fast_chrg_port_conn', 'fast_chrg_relay_on', 'hvac_list_1',
    'hvac_list_2', 'insul_resistance', 'int_temp', 'inverter_capacity_volt', 'main_relay_conn', 'max_cell_volt',
    'max_cell_volt_no', 'max_deter_cell_no', 'min_cell_volt', 'min_cell_volt_no', 'min_deter', 'min_deter_cell_no',
    'mod_avg_temp', 'mod_max_temp', 'mod_min_temp', 'mod_temp_1', 'mod_temp_2', 'mod_temp_3', 'mod_temp_4',
    'mod_temp_5', 'mod_temp_6', 'mod_temp_7', 'mod_temp_8', 'mod_temp_9', 'mod_temp_10', 'mod_temp_11', 'mod_temp_12',
    'mod_temp_13', 'mod_temp_14', 'mod_temp_15', 'mod_temp_16', 'mod_temp_17', 'mod_temp_18', 'msg_id', 'odometer',
    'op_time', 'pack_current', 'pack_volt', 'seq', 'slow_chrg_port_conn', 'soc', 'socd', 'soh', 'start_time',
    'sub_batt_volt', 'trip_chrg_pw', 'trip_dischrg_pw', 'v2l'
]
_gps_split_col = ['device_no', 'time', 'direction', 'fuel_pct', 'hdop', 'lat', 'lng', 'mode', 'source', 'speed', 'state']


# --- 3. 유틸리티 및 헬퍼 함수 ---


def detect_file_category(csv_filename):
    filename = Path(csv_filename).name.lower()
    if 'bms' in filename: return 'bms'
    if 'gps' in filename: return 'gps'
    return 'unknown'


def _split_comma_series(series: pd.Series) -> pd.Series:
    return series.fillna('').astype(str).str.split(',').apply(lambda lst: [v.strip() for v in lst if v.strip()])


def expand_list_columns(df: pd.DataFrame, log_container) -> pd.DataFrame:
    out = df.copy()
    
    cell_volt_list_col, mod_temp_list_col = None, None
    for col in df.columns:
        if 'cell_volt_list' in col.lower(): cell_volt_list_col = col
        elif 'mod_temp_list' in col.lower(): mod_temp_list_col = col

    if cell_volt_list_col:
        col = cell_volt_list_col
        split_values = _split_comma_series(out[col])
        max_len = split_values.str.len().max() if len(split_values) > 0 else 0
        if max_len > 0:
            col_names = [f'cell_volt_{i+1}' for i in range(max_len)]
            cell_df = pd.DataFrame(split_values.tolist(), columns=col_names, index=out.index).apply(pd.to_numeric, errors='coerce').fillna(0)
            out = pd.concat([out.drop(columns=[col]), cell_df], axis=1)
        else:
            out = out.drop(columns=[col])

    if mod_temp_list_col:
        col = mod_temp_list_col
        split_values = _split_comma_series(out[col])
        max_len = split_values.str.len().max() if len(split_values) > 0 else 0
        if max_len > 0:
            col_names = [f'mod_temp_{i+1}' for i in range(max_len)]
            temp_df = pd.DataFrame(split_values.tolist(), columns=col_names, index=out.index).apply(pd.to_numeric, errors='coerce').fillna(0)
            out = pd.concat([out.drop(columns=[col]), temp_df], axis=1)
        else:
            out = out.drop(columns=[col])
    return out


def extract_date_from_filename(filename):
    filename_str = str(Path(filename).stem)
    match = re.search(r'(\d{4}-\d{2})', filename_str)
    if match: return match.group(1)
    match = re.search(r'_(\d{4})_', filename_str)
    if match:
        yymm = match.group(1)
        return f"20{yymm[:2]}-{yymm[2:]}"
    return "UNKNOWN_DATE"


@st.cache_data
def load_cartype_map(csv_path):
    try:
        df = pd.read_csv(csv_path, header=0, dtype={'device_no': str})
        df = df.dropna(subset=['device_no', 'car_type'])
        return df.set_index('device_no')['car_type'].to_dict()
    except FileNotFoundError:
        st.error(f"❌ 차종 정보 파일({csv_path})을 찾을 수 없습니다. 스크립트와 같은 폴더에 `aicar_cartype_list.csv` 파일이 있는지 확인하세요.")
        st.stop()
    except Exception as e:
        st.error(f"❌ 차종 정보 파일 처리 중 에러 발생: {e}")
        st.stop()


def get_output_path(date_str, category, car_type, original_filename_str):
    cleaned_car_type = car_type
    output_subdir = PROCESSED_DATA_DIR / str(date_str) / str(category) / cleaned_car_type
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{Path(original_filename_str).stem}_processed.pkl"
    return output_subdir / output_filename


# --- 4. 핵심 데이터 처리 함수 ---


def data_split_by_category(file_buffer, source_name, cartype_map, force_reprocess, log_container):
    category = detect_file_category(source_name)
    if category == 'unknown':
        log_container.warning(f"⚠️ `{source_name}`: 카테고리(bms/gps)를 알 수 없어 건너뜁니다.")
        return [("failed", source_name)]

    date_str = extract_date_from_filename(source_name)
    result_df = pd.DataFrame()

    if category == 'bms':
        try:
            file_buffer.seek(0)
            bms_df = pd.read_csv(file_buffer, sep='|', dtype=str, encoding='utf-8-sig', 
                                 skipinitialspace=True, on_bad_lines='skip', low_memory=False)
            if bms_df.empty:
                log_container.warning(f"⚠️ `{source_name}`: 비어있는 BMS 파일입니다.")
                return [("failed", source_name)]
            
            if len(bms_df) > 0 and bms_df.iloc[0].astype(str).str.contains('-').any(): bms_df = bms_df.drop(0).reset_index(drop=True)
            if len(bms_df) > 0 and bms_df.iloc[-1].nunique() <= 2: bms_df = bms_df.iloc[:-1].reset_index(drop=True)
            bms_df = bms_df.dropna(how='all').reset_index(drop=True)
            if bms_df.empty: return [("failed", source_name)]
            bms_df.columns = [col.strip() for col in bms_df.columns]
            
            try:
                col_list = bms_df.columns.tolist()
                insul_idx = col_list.index('insul_resistance')
                if insul_idx >= 2:
                    bms_df['hvac_list_1'] = bms_df.iloc[:, insul_idx - 1]
                    bms_df['hvac_list_2'] = bms_df.iloc[:, insul_idx - 2]
            except ValueError:
                pass 

            bms_df = bms_df.replace('', pd.NA)
            bms_df = expand_list_columns(bms_df, log_container)
            result_df = bms_df
        except Exception as e:
            log_container.error(f"❌ `{source_name}` BMS 처리 오류: {e}")
            return [("failed", source_name)]

    elif category == 'gps':
        try:
            file_buffer.seek(0)
            df_raw = pd.read_csv(file_buffer, header=None, dtype=str, encoding='utf-8-sig', on_bad_lines='skip', low_memory=False)
            if df_raw.empty or len(df_raw.columns) == 0:
                log_container.warning(f"⚠️ `{source_name}`: 비어있는 GPS 파일입니다.")
                return [("failed", source_name)]
            split_rows = df_raw[0].str.split('|').apply(lambda x: [s.strip() for s in x])
            split_rows = split_rows[split_rows.apply(lambda x: len(x) > 1 and x[0] != '')].reset_index(drop=True)
            if len(split_rows) > 2: split_rows = split_rows[2:].reset_index(drop=True)
            if len(split_rows) == 0: return [("failed", source_name)]
            gps_df = pd.DataFrame(split_rows.tolist())
            if len(gps_df.columns) >= len(_gps_split_col): gps_df = gps_df.iloc[:, :len(_gps_split_col)]
            gps_df.columns = _gps_split_col[:len(gps_df.columns)]
            result_df = gps_df.replace('', np.nan).replace('nan', np.nan)
        except Exception as e:
            log_container.error(f"❌ `{source_name}` GPS 처리 오류: {e}")
            return [("failed", source_name)]
    
    if category == 'bms' and 'msg_time' in result_df.columns and 'seq' in result_df.columns:
        result_df['msg_time'] = pd.to_datetime(result_df['msg_time'], errors='coerce')
        result_df['seq'] = pd.to_numeric(result_df['seq'], errors='coerce')
        result_df.sort_values(by=['msg_time', 'seq'], ascending=True, inplace=True, na_position='first')

    if 'device_no' not in result_df.columns:
        log_container.error(f"❌ `{source_name}`: 'device_no' 컬럼이 없어 차종 분리 불가.")
        return [("failed", source_name)]
    
    result_df['car_type'] = result_df['device_no'].astype(str).map(cartype_map).fillna('UNKNOWN_CAR_TYPE')
    grouped = result_df.groupby('car_type')
    output_results = []
    
    for car_type, group_df in grouped:
        group_df_to_save = group_df.drop(columns=['car_type'])
        
        if category == 'bms':
            existing_cols = [col for col in FINAL_BMS_COLUMN_ORDER if col in group_df_to_save.columns]
            other_cols = [col for col in group_df_to_save.columns if col not in existing_cols]
            group_df_to_save = group_df_to_save[existing_cols + other_cols]

        save_path = get_output_path(date_str, category, car_type, source_name)
        
        if not force_reprocess and save_path.exists():
            output_results.append(("skipped", save_path))
            continue
        
        try:
            group_df_to_save = group_df_to_save.drop_duplicates()
            group_df_to_save.to_pickle(save_path)
            output_results.append(("processed", save_path))
        except Exception as e:
            log_container.error(f"❌ Pickle 저장 실패: {save_path}, 오류: {e}")
            output_results.append(("failed", save_path))
            
    del result_df, grouped
    gc.collect()
    return output_results


# --- 5. Streamlit 애플리케이션 ---


def process_recursively(file_buffer, file_name, cartype_map, force_reprocess, log_container, results_summary):
    if file_name.lower().endswith('.csv'):
        log_container.info(f"📄 CSV 파일 처리 중: {file_name}")
        results = data_split_by_category(file_buffer, file_name, cartype_map, force_reprocess, log_container)
        for status, path_or_name in results:
            results_summary[status] += 1
            if status == "processed": log_container.success(f"✅ 성공: {Path(path_or_name).relative_to(BASE_DIR)}")
            elif status == "skipped": log_container.warning(f"⏭️ 스킵: {Path(path_or_name).relative_to(BASE_DIR)}")
        return

    if file_name.lower().endswith('.zip'):
        try:
            with zipfile.ZipFile(file_buffer, 'r') as zf:
                for member_info in zf.infolist():
                    if member_info.is_dir() or member_info.filename.lower().startswith('__macosx/'):
                        continue
                    
                    member_name = Path(member_info.filename).name
                    if any(keyword in member_name.lower() for keyword in EXCLUDE_KEYWORDS):
                        log_container.warning(f"🚫 `{member_name}`: 제외 키워드가 포함되어 건너뜁니다.")
                        continue
                    
                    with zf.open(member_info.filename) as member_file:
                        member_buffer = io.BytesIO(member_file.read())
                        process_recursively(member_buffer, member_info.filename, cartype_map, force_reprocess, log_container, results_summary)

        except zipfile.BadZipFile:
            log_container.error(f"❌ 손상된 ZIP 파일입니다: {file_name}")
            results_summary["failed"] += 1
        return

# ---- 메인 앱 ----
st.set_page_config(page_title="차량 데이터 전처리기", layout="wide")
st.title("🚗 차량 운행 데이터(BMS/GPS) 전처리기")
st.markdown("---")

cartype_map = load_cartype_map(CARTYPE_CSV_PATH)

tab1, tab2 = st.tabs(["📁 파일 업로드 및 전처리", "📊 전처리 결과 확인"])

# --- TAB 1: 파일 업로드 및 처리 ---
with tab1:
    st.header("1. 파일 업로드")
    st.info("처리할 CSV 또는 ZIP 파일을 업로드하세요. ZIP 파일 안에 다른 ZIP 파일이 포함되어 있어도 자동으로 처리됩니다.")
    uploaded_files = st.file_uploader("파일 선택", type=['csv', 'zip'], accept_multiple_files=True, label_visibility="collapsed")
    
    st.header("2. 처리 옵션")
    force_reprocess = st.checkbox("강제 재처리 (이미 처리된 파일도 덮어쓰기)", value=False)
    
    st.markdown("---")
    
    if st.button("🚀 전처리 시작", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("⚠️ 파일을 먼저 업로드해주세요.")
        else:
            results_summary = {"processed": 0, "skipped": 0, "failed": 0}
            log_container = st.container()

            with st.spinner("파일 처리 중... 잠시만 기다려주세요."):
                log_container.info(f"총 {len(uploaded_files)}개의 파일 처리를 시작합니다.")
                for uploaded_file in uploaded_files:
                    log_container.markdown(f"--- \n### 🗂️ 처리 대상: **{uploaded_file.name}**")
                    file_buffer = io.BytesIO(uploaded_file.getvalue())
                    process_recursively(file_buffer, uploaded_file.name, cartype_map, force_reprocess, log_container, results_summary)

            st.markdown("---")
            st.header("🎉 처리 완료!")
            col1, col2, col3 = st.columns(3)
            col1.metric("✅ 성공", results_summary["processed"], "개")
            col2.metric("⏭️ 스킵", results_summary["skipped"], "개")
            col3.metric("❌ 실패", results_summary["failed"], "개")

# --- TAB 2: 결과 확인 ---
with tab2:
    st.header("저장된 Pickle 파일 확인")
    
    pkl_files = list(PROCESSED_DATA_DIR.rglob("*.pkl"))
    
    if not pkl_files:
        st.info("처리된 파일이 없습니다. 먼저 '파일 업로드 및 전처리' 탭에서 작업을 완료해주세요.")
    else:
        pkl_file_paths_str = sorted([str(p.relative_to(BASE_DIR)) for p in pkl_files])
        
        selected_pkl_path = st.selectbox(
            "확인할 파일을 선택하세요:", 
            pkl_file_paths_str,
            index=None,
            placeholder="목록에서 파일을 선택하세요..."
        )

        if st.button("📥 선택한 파일 로드", use_container_width=True):
            if selected_pkl_path:
                try:
                    full_path = BASE_DIR / selected_pkl_path
                    st.session_state.loaded_df = pd.read_pickle(full_path)
                    st.session_state.loaded_path = selected_pkl_path
                except Exception as e:
                    st.error(f"파일을 불러오는 중 오류가 발생했습니다: {e}")
                    st.session_state.loaded_df = None
                    st.session_state.loaded_path = None
            else:
                st.warning("⚠️ 파일을 먼저 목록에서 선택해주세요.")
                st.session_state.loaded_df = None
                st.session_state.loaded_path = None

        if 'loaded_df' in st.session_state and st.session_state.loaded_df is not None:
            df = st.session_state.loaded_df
            path = st.session_state.loaded_path

            st.markdown("---")
            st.markdown(f"#### 📄 표시 중인 파일: `{path}`")
            st.markdown(f"데이터 형태: **{df.shape[0]}** 행, **{df.shape[1]}** 열")
            
            st.dataframe(df, height=400)
            
            @st.cache_data
            def convert_df_to_csv(df_to_convert):
                return df_to_convert.to_csv(index=False).encode('utf-8-sig')

            csv_data = convert_df_to_csv(df)
            
            st.download_button(
                label="이 테이블을 CSV로 다운로드",
                data=csv_data,
                file_name=f"{Path(path).stem}.csv",
                mime='text/csv',
            )
