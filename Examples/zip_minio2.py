# zip_minio.py
# ZIP → (전처리, 중간 CSV 미저장) → PostgreSQL COPY + MERGE
# 요구사항:
#  - MinIO에서 origin_zip/ 아래 모든 zip을 순회하고, 중첩 zip 내부의 csv까지 탐색
#  - BMS/GPS 전처리 후 곧장 DB에 COPY → MERGE
#  - 이상치 처리 없음
#  - --direct 모드 전용
#  - 스크린샷의 폴더 aicar_2212_origin_old_zip/ 은 스캔 제외

import io
import os
import re
import gc
import sys
import argparse
import zipfile
from pathlib import Path
from typing import List, Tuple, Iterable, Optional, Dict, Any

import boto3
import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
import psycopg2
from io import StringIO

# =========================
# 상수/스키마
# =========================
SCHEMA_CELL_VOLT_MAX = 192
SCHEMA_MOD_TEMP_MAX  = 18

BMS_COLUMN_ORDER = [
    'device_no','measured_month','time','msg_time','acceptable_chrg_pw','acceptable_dischrg_pw','airbag_hwire_duty',
    'batt_coolant_inlet_temp','batt_fan_running','batt_internal_temp','batt_ltr_rear_temp','batt_pra_busbar_temp',
    'batt_pw','bms_running','cellvolt_dispersion'
] + [f'cell_volt_{i}' for i in range(1, SCHEMA_CELL_VOLT_MAX+1)] + [
    'chrg_cable_conn','chrg_cnt','chrg_cnt_q','cumul_current_chrgd','cumul_current_dischrgd','cumul_energy_chrgd',
    'cumul_energy_chrgd_q','cumul_pw_chrgd','cumul_pw_dischrgd','drive_motor_spd_1','drive_motor_spd_2',
    'emobility_spd','est_chrg_time','ext_temp','fast_chrg_port_conn','fast_chrg_relay_on',
    'hvac_list_1','hvac_list_2','insul_resistance','int_temp','inverter_capacity_volt','main_relay_conn',
    'max_cell_volt','max_cell_volt_no','max_deter_cell_no','min_cell_volt','min_cell_volt_no','min_deter',
    'min_deter_cell_no','mod_avg_temp','mod_max_temp','mod_min_temp'
] + [f'mod_temp_{i}' for i in range(1, SCHEMA_MOD_TEMP_MAX+1)] + [
    'msg_id','odometer','op_time','pack_current','pack_volt','seq','slow_chrg_port_conn',
    'soc','socd','soh','start_time','sub_batt_volt','trip_chrg_pw','trip_dischrg_pw','v2l',
    # 통계 컬럼(스키마에 존재)
    'cell_volt_mode','cell_volt_min','cell_volt_max','cell_volt_median','cell_volt_mean',
    'mod_temp_mode','mod_temp_min','mod_temp_max','mod_temp_median','mod_temp_mean',
    'cell_volt_count','mod_temp_count'
]

GPS_COLUMNS = ['device_no','time','direction','fuel_pct','hdop','lat','lng','mode','source','speed','state']

TS_COLS  = ['time','measured_month','msg_time','start_time']
STR_COLS = {'device_no','msg_id','mode','source','state'}

HEADER_SYNONYMS_LOWER = {'device_no','device no','deviceno','time'}

MIN_ROWS_TO_SAVE = 2

# =========================
# MinIO (boto3)
# =========================
def get_s3(service_name: str, endpoint_url: str, access_key: str, secret_key: str):
    s3 = boto3.resource(
        service_name=service_name,
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    s3_client = boto3.client(
        service_name=service_name,
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    return s3, s3_client

def list_zip_keys(s3_client, bucket: str, prefix: str) -> List[str]:
    """
    MinIO에서 ZIP 키 목록을 가져온다.
    - 'aicar_2212_origin_old_zip/' 하위는 전부 제외한다.
    """
    EXCLUDE_SUBDIRS = {"aicar_2212_origin_old_zip"}

    keys = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3_client.list_objects_v2(**kwargs)

        for it in resp.get("Contents", []):
            k = it["Key"]              # 예: aicar_data/origin_zip/xxx.zip
            if it["Size"] <= 0:
                continue
            # 경로 세그먼트 기준으로 제외 폴더 포함 여부 확인
            segs = [seg for seg in k.split('/') if seg]
            if any(seg in EXCLUDE_SUBDIRS for seg in segs):
                continue
            if k.lower().endswith(".zip"):
                keys.append(k)

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys

def get_object_bytes(s3_client, bucket: str, key: str) -> bytes:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

# =========================
# ZIP 탐색/CSV 청크 반복
# =========================
def _is_trash_member(name: str) -> bool:
    lower = name.lower()
    if lower.startswith('__macosx/'):
        return True
    if Path(name).name.startswith('._'):
        return True
    return False

def list_members_in_chain(root_zip_bytes: bytes, chain: List[str]) -> List[str]:
    curr = zipfile.ZipFile(io.BytesIO(root_zip_bytes), 'r')
    try:
        for part in chain:
            with curr.open(part) as f:
                data = f.read()
            curr.close()
            curr = zipfile.ZipFile(io.BytesIO(data), 'r')
        out = []
        for info in curr.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if _is_trash_member(name):
                continue
            if info.file_size == 0:
                continue
            lower = name.lower()
            if lower.endswith('.csv') or lower.endswith('.zip'):
                out.append(name)
        return out
    finally:
        try: curr.close()
        except Exception: pass

def walk_all_csv_chains(root_zip_bytes: bytes, include_nested: bool=True) -> List[Tuple[List[str], str]]:
    result: List[Tuple[List[str], str]] = []
    stack: List[List[str]] = [[]]
    while stack:
        chain = stack.pop()
        members = list_members_in_chain(root_zip_bytes, chain)
        for m in members:
            if m.lower().endswith('.csv'):
                result.append((chain + [m], m))
            elif include_nested and m.lower().endswith('.zip'):
                stack.append(chain + [m])
    return result

def iter_csv_chunks_from_chain(root_zip_bytes: bytes, entry_chain: List[str], *, sep: str, header: Optional[int], chunksize: int) -> Iterable[pd.DataFrame]:
    last_csv = entry_chain[-1]
    curr = zipfile.ZipFile(io.BytesIO(root_zip_bytes), 'r')
    opened_stack = []
    try:
        for part in entry_chain[:-1]:
            with curr.open(part) as f:
                data = f.read()
            opened_stack.append(curr)
            curr = zipfile.ZipFile(io.BytesIO(data), 'r')
        with curr.open(last_csv, 'r') as raw:
            try:
                for chunk in pd.read_csv(
                    raw, sep=sep, header=header, chunksize=chunksize,
                    low_memory=False, dtype=str
                ):
                    yield chunk
            except EmptyDataError:
                return
    finally:
        try: curr.close()
        except Exception: pass
        for z in opened_stack:
            try: z.close()
            except Exception: pass

# =========================
# 전처리(간단/이상치 없음)
# =========================
def _fix_year_vectorized(s: pd.Series) -> pd.Series:
    s = s.astype('string').str.strip()
    mask_two = s.str.match(r'^\d{2}[-/.]')  # 23-... → 2023-...
    s = s.mask(mask_two, s.str.replace(r'^(\d{2})([-/.])', r'20\1\2', regex=True))
    return pd.to_datetime(s, errors='coerce')

def drop_trailing_footer_if_any(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        last = df.iloc[-1]
        s = ' '.join([str(x) for x in last.tolist()]).strip().lower()
        footer_patterns = [r'\(\s*\d+\s*rows?\s*\)', r'\brows?\b', r'#+', r'\bsummary\b', r'\btotal\b', r'합계', r'\b총\s*\d+']
        if s and any(re.search(p, s) for p in footer_patterns):
            return df.iloc[:-1].copy()
    except Exception:
        pass
    return df

def detect_file_category(name: str) -> str:
    f = str(name).lower()
    if 'bms' in f: return 'bms'
    if 'gps' in f: return 'gps'
    return 'skip'

def _drop_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df
    if 'device_no' in d.columns:
        mask_header = d['device_no'].astype(str).str.strip().str.lower().isin(HEADER_SYNONYMS_LOWER)
        d = d[~mask_header]
    if 'time' in d.columns:
        mask_time_hdr = d['time'].astype(str).str.strip().str.lower().isin({'time'})
        d = d[~mask_time_hdr]
    return d

def expand_list_fixed(df_in: pd.DataFrame, list_keyword: str, base_prefix: str, fixed_len: int) -> pd.DataFrame:
    df = df_in.copy()
    cols = list(df.columns)
    list_col = next((c for c in cols if list_keyword in str(c).lower()), None)
    if not list_col:
        return df
    ser = df[list_col].astype('object')
    ser_str = pd.Series(ser, index=df.index).astype(str).str.strip()
    empty_mask = ser_str.eq('') | ser_str.str.lower().isin(['nan', 'none'])
    if empty_mask.all():
        tmp = pd.DataFrame({f"{base_prefix}_{i+1}": pd.NA for i in range(fixed_len)}, index=df.index)
    else:
        tmp = ser_str.str.split(',', n=fixed_len-1, expand=True)
        want_cols = list(range(fixed_len))
        tmp = tmp.reindex(columns=want_cols)
        tmp.columns = [f"{base_prefix}_{i+1}" for i in range(fixed_len)]
        try:
            tmp = tmp.apply(pd.to_numeric, errors='coerce')
        except Exception:
            pass
    insert_at = cols.index(list_col)
    out = pd.concat([df.iloc[:, :insert_at], tmp, df.iloc[:, insert_at+1:]], axis=1)
    return out

# ---- 헤더 표준화 (BMS) ----
HEADER_ALIAS_MAP_BMS = {
    # 디바이스 PK
    'device no': 'device_no',
    'deviceno': 'device_no',
    'device_no': 'device_no',
    'device-id': 'device_no',
    'deviceid': 'device_no',
    # 타임스탬프
    'time': 'time',
    'msgtime': 'msg_time',
    'msg_time': 'msg_time',
    'measuredmonth': 'measured_month',
    'measured_month': 'measured_month',
    'starttime': 'start_time',
    'start_time': 'start_time',
    # 리스트 컬럼
    'cell_volt_list': 'cell_volt_list',
    'cellvolt_list': 'cell_volt_list',
    'mod_temp_list': 'mod_temp_list',
    'module_temp_list': 'mod_temp_list',
}

def normalize_bms_headers(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    new_cols = []
    for c in df.columns:
        k = str(c).strip().lower()
        new_cols.append(HEADER_ALIAS_MAP_BMS.get(k, c))
    df = df.copy()
    df.columns = new_cols
    return df
# ---------------------------

def preprocess_bms_chunk(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    # 1) 헤더 표준화
    df = normalize_bms_headers(df)

    # 2) 리스트 확장(표준화 이후)
    if any('cell_volt_list' in str(c).lower() for c in df.columns):
        df = expand_list_fixed(df, 'cell_volt_list', 'cell_volt', SCHEMA_CELL_VOLT_MAX)
    if any('mod_temp_list' in str(c).lower() for c in df.columns):
        df = expand_list_fixed(df, 'mod_temp_list', 'mod_temp', SCHEMA_MOD_TEMP_MAX)

    # 3) 푸터/재헤더 제거
    df = drop_trailing_footer_if_any(df)
    if len(df) > 0:
        row0 = df.iloc[0].astype(str).str.strip().str.lower()
        if any(v in HEADER_SYNONYMS_LOWER for v in row0.values):
            df = df.iloc[1:].reset_index(drop=True)
    df = df.replace('', pd.NA)
    df = _drop_header_rows(df)

    # 4) 숫자/문자 단순 정리
    non_num = {'device_no','measured_month','time','msg_time','start_time','int_temp'}
    for col in df.columns:
        if col not in non_num:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5) 시간열 파싱
    for dt_col in TS_COLS:
        if dt_col in df.columns:
            df[dt_col] = _fix_year_vectorized(df[dt_col])

    # 6) device_no 정리
    if 'device_no' in df.columns:
        df['device_no'] = df['device_no'].astype(str).str.replace(r'[\r\n]', '', regex=True).str.strip()

    df = drop_trailing_footer_if_any(df)
    return df.drop_duplicates()

def preprocess_gps_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=GPS_COLUMNS)
    df = drop_trailing_footer_if_any(df)

    # 첫 컬럼 전체 문자열 → '|' split
    if len(df.columns) >= 1:
        gps_df = df.iloc[:,0].astype(str).str.split('|', expand=True)
    else:
        gps_df = pd.DataFrame()

    n = min(len(gps_df.columns), len(GPS_COLUMNS))
    gps_df = gps_df.iloc[:, :n]
    gps_df.columns = GPS_COLUMNS[:n]

    # 타입
    if 'time' in gps_df.columns:
        gps_df['time'] = _fix_year_vectorized(gps_df['time'])
    for col in ['lat','lng','speed','direction','hdop','fuel_pct']:
        if col in gps_df.columns:
            gps_df[col] = pd.to_numeric(gps_df[col], errors='coerce')

    if 'device_no' in gps_df.columns:
        gps_df['device_no'] = gps_df['device_no'].astype(str).str.replace(r'[\r\n]', '', regex=True).str.strip()

    for c in GPS_COLUMNS:
        if c not in gps_df.columns:
            gps_df[c] = pd.NA

    gps_df = _drop_header_rows(gps_df)
    gps_df = drop_trailing_footer_if_any(gps_df)
    gps_df = gps_df.drop_duplicates()
    try:
        gps_df = gps_df.sort_values(by='time', ascending=True, na_position='last').reset_index(drop=True)
    except Exception:
        pass
    return gps_df[GPS_COLUMNS]

def clean_data_for_postgres(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    dfc = _drop_header_rows(dfc)

    # 시간 파싱 재확인
    for col in TS_COLS:
        if col in dfc.columns:
            dfc[col] = pd.to_datetime(dfc[col], errors='coerce')

    # 반드시 time 있어야 함
    if 'time' in dfc.columns:
        dfc = dfc[~dfc['time'].isna()]

    # 숫자/문자
    for col in dfc.columns:
        if col not in STR_COLS and col not in TS_COLS:
            dfc[col] = pd.to_numeric(dfc[col], errors='coerce').replace([np.inf,-np.inf], np.nan)
    for col in STR_COLS:
        if col in dfc.columns:
            s = dfc[col].astype(str).replace(['nan','NaN','None'],'').str.strip()
            dfc[col] = s

    # 완전 공백 행 제거
    dfc = dfc.dropna(how='all')
    return dfc

# =========================
# DB: 연결/스테이지/머지
# =========================
def connect_db(host: str, port: int, dbname: str, user: str, password: str):
    conn = psycopg2.connect(host=host, port=int(port), dbname=dbname, user=user, password=password)
    conn.set_client_encoding("UTF8")
    with conn.cursor() as cur:
        cur.execute("SET search_path TO aicar, public;")
        cur.execute("SET LOCAL synchronous_commit = OFF;")
        cur.execute("SET LOCAL wal_compression = ON;")
        cur.execute("SET LOCAL temp_buffers = '128MB';")
        cur.execute("SET LOCAL work_mem = '256MB';")
    conn.commit()
    return conn

def ensure_base_tables(conn):
    ddl = """
    CREATE SCHEMA IF NOT EXISTS aicar;
    SET search_path TO aicar, public;

    CREATE TABLE IF NOT EXISTS device (
      device_no  TEXT PRIMARY KEY,
      created_at TIMESTAMPTZ NULL,
      updated_at TIMESTAMPTZ NULL
    );

    CREATE TABLE IF NOT EXISTS bms (
      device_no TEXT NOT NULL,
      "time" TIMESTAMPTZ NOT NULL,
      measured_month TIMESTAMPTZ NULL,
      msg_time TIMESTAMPTZ NULL,
      acceptable_chrg_pw NUMERIC,
      acceptable_dischrg_pw NUMERIC,
      airbag_hwire_duty NUMERIC,
      batt_coolant_inlet_temp NUMERIC,
      batt_fan_running NUMERIC,
      batt_internal_temp NUMERIC,
      batt_ltr_rear_temp NUMERIC,
      batt_pra_busbar_temp NUMERIC,
      batt_pw NUMERIC,
      bms_running NUMERIC,
      cellvolt_dispersion NUMERIC,
      """ + ",".join([f"cell_volt_{i} NUMERIC" for i in range(1,193)]) + """,
      chrg_cable_conn NUMERIC,
      chrg_cnt NUMERIC,
      chrg_cnt_q NUMERIC,
      cumul_current_chrgd NUMERIC,
      cumul_current_dischrgd NUMERIC,
      cumul_energy_chrgd NUMERIC,
      cumul_energy_chrgd_q NUMERIC,
      cumul_pw_chrgd NUMERIC,
      cumul_pw_dischrgd NUMERIC,
      drive_motor_spd_1 NUMERIC,
      drive_motor_spd_2 NUMERIC,
      emobility_spd NUMERIC,
      est_chrg_time NUMERIC,
      ext_temp NUMERIC,
      fast_chrg_port_conn NUMERIC,
      fast_chrg_relay_on NUMERIC,
      hvac_list_1 NUMERIC,
      hvac_list_2 NUMERIC,
      insul_resistance NUMERIC,
      int_temp NUMERIC,
      inverter_capacity_volt NUMERIC,
      main_relay_conn NUMERIC,
      max_cell_volt NUMERIC,
      max_cell_volt_no NUMERIC,
      max_deter_cell_no NUMERIC,
      min_cell_volt NUMERIC,
      min_cell_volt_no NUMERIC,
      min_deter NUMERIC,
      min_deter_cell_no NUMERIC,
      mod_avg_temp NUMERIC,
      mod_max_temp NUMERIC,
      mod_min_temp NUMERIC,
      """ + ",".join([f"mod_temp_{i} NUMERIC" for i in range(1,19)]) + """,
      msg_id TEXT,
      odometer NUMERIC,
      op_time NUMERIC,
      pack_current NUMERIC,
      pack_volt NUMERIC,
      seq NUMERIC,
      slow_chrg_port_conn NUMERIC,
      soc NUMERIC,
      socd NUMERIC,
      soh NUMERIC,
      start_time TIMESTAMPTZ NULL,
      sub_batt_volt NUMERIC,
      trip_chrg_pw NUMERIC,
      trip_dischrg_pw NUMERIC,
      v2l NUMERIC,
      cell_volt_mode NUMERIC,
      cell_volt_min NUMERIC,
      cell_volt_max NUMERIC,
      cell_volt_median NUMERIC,
      cell_volt_mean NUMERIC,
      mod_temp_mode NUMERIC,
      mod_temp_min NUMERIC,
      mod_temp_max NUMERIC,
      mod_temp_median NUMERIC,
      mod_temp_mean NUMERIC,
      cell_volt_count NUMERIC,
      mod_temp_count NUMERIC,
      CONSTRAINT bms_pk PRIMARY KEY (device_no, "time")
    );

    CREATE TABLE IF NOT EXISTS gps (
      device_no TEXT NOT NULL,
      "time"      TIMESTAMPTZ NOT NULL,
      direction NUMERIC,
      fuel_pct  NUMERIC,
      hdop      NUMERIC,
      lat       NUMERIC,
      lng       NUMERIC,
      mode      TEXT,
      source    TEXT,
      speed     NUMERIC,
      state     TEXT,
      CONSTRAINT gps_pk PRIMARY KEY (device_no, "time")
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()

def _create_stage_with_index(conn, stage: str, target: str, schema_cols: List[str]):
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {stage};")
        cur.execute(f"CREATE TEMP TABLE {stage} (LIKE {target} INCLUDING DEFAULTS);")
        if {'device_no','time','seq'}.issubset(set(schema_cols)):
            cur.execute(f"CREATE INDEX ON {stage} (device_no, \"time\", seq DESC);")
        elif {'device_no','time'}.issubset(set(schema_cols)):
            cur.execute(f"CREATE INDEX ON {stage} (device_no, \"time\");")
    conn.commit()

def _copy_into_stage_buffered(conn, stage: str, columns: List[str], buf: StringIO):
    buf.seek(0)
    sql = f"COPY {stage} ({', '.join(columns)}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
    with conn.cursor() as cur:
        cur.copy_expert(sql=sql, file=buf)
    conn.commit()

def ensure_devices_from_stage(conn, stage: str):
    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO aicar.device (device_no, created_at, updated_at)
            SELECT DISTINCT s.device_no, NOW(), NOW()
            FROM {stage} s
            WHERE s.device_no IS NOT NULL
              AND length(btrim(s.device_no)) > 0
              AND lower(btrim(s.device_no)) NOT IN ('device_no','device no','deviceno')
            ON CONFLICT (device_no) DO UPDATE
              SET updated_at = EXCLUDED.updated_at;
        """)
    conn.commit()

def _merge_stage_into_target(conn, stage: str, target: str, columns: list, pk_cols: list, do_update: bool) -> int:
    col_list = ", ".join(columns)
    pk_list = ", ".join(pk_cols)

    order_keys = []
    if 'msg_time' in columns and 'time' in columns:
        order_keys.append("COALESCE(msg_time, time) DESC NULLS LAST")
    elif 'msg_time' in columns:
        order_keys.append("msg_time DESC NULLS LAST")
    elif 'time' in columns:
        order_keys.append("\"time\" DESC NULLS LAST")
    if 'seq' in columns:
        order_keys.append("seq DESC NULLS LAST")
    order_clause = ", ".join([*pk_cols, *order_keys]) if order_keys else ", ".join(pk_cols)

    where_conds = [f"{pk} IS NOT NULL" for pk in pk_cols]
    if 'device_no' in pk_cols:
        where_conds.append("length(btrim(device_no)) > 0")
        where_conds.append("lower(btrim(device_no)) NOT IN ('device_no','device no','deviceno')")
    where_clause = " AND ".join(where_conds)

    dedup = f"{stage}_dedup"
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {dedup};")
        cur.execute(f"""
            CREATE TEMP TABLE {dedup} AS
            SELECT DISTINCT ON ({pk_list}) {col_list}
            FROM {stage}
            WHERE {where_clause}
            ORDER BY {order_clause};
        """)
        non_pk_update_cols = [c for c in columns if c not in pk_cols]
        if do_update and non_pk_update_cols:
            set_clause = ", ".join([f"{c}=EXCLUDED.{c}" for c in non_pk_update_cols])
            merge_sql = f"""
                INSERT INTO {target} ({col_list})
                SELECT {col_list} FROM {dedup}
                ON CONFLICT ({pk_list}) DO UPDATE SET {set_clause};
            """
        else:
            merge_sql = f"""
                INSERT INTO {target} ({col_list})
                SELECT {col_list} FROM {dedup}
                ON CONFLICT ({pk_list}) DO NOTHING;
            """
        cur.execute(merge_sql)
        affected = cur.rowcount
    conn.commit()
    return affected or 0

# =========================
# 원패스 처리 루틴
# =========================
def process_bms_chain_direct(conn, root_zip_bytes: bytes, chain: List[str], chunksize: int, do_update: bool) -> int:
    stage = 'bms_stage'
    target = 'aicar.bms'
    schema_cols = [c for c in BMS_COLUMN_ORDER]
    pk = ['device_no','time']

    _create_stage_with_index(conn, stage, target, schema_cols)

    total = 0
    # 타깃 실제 컬럼 순서 조회
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.attname
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = %s AND n.nspname = %s
              AND a.attnum > 0 AND NOT a.attisdropped
            ORDER BY a.attnum;
        """, (target.split('.')[-1], 'aicar'))
        target_order = [r[0] for r in cur.fetchall()]

    batch_buf = StringIO()
    batch_cols: Optional[List[str]] = None
    rows_in_batch = 0
    BATCH_SIZE = max(50_000, chunksize)

    for chunk in iter_csv_chunks_from_chain(root_zip_bytes, chain, sep='|', header=0, chunksize=chunksize):
        if chunk is None or len(chunk)==0:
            continue

        proc = preprocess_bms_chunk(chunk)
        if proc.empty or len(proc) < MIN_ROWS_TO_SAVE:
            del chunk, proc; gc.collect(); continue

        df = clean_data_for_postgres(proc)

        # ★ PK 보장: 필수 컬럼 없으면 스킵
        if 'device_no' not in df.columns or 'time' not in df.columns:
            print("[SKIP] BMS chunk without required PK columns (device_no/time)")
            del chunk, proc, df; gc.collect(); continue

        # ★ PK 유효값 필터
        dev = df['device_no'].astype(str).str.strip()
        df = df[(dev.str.len() > 0) & (~df['time'].isna())]

        # 완전 공백 행 제거
        df = df.dropna(how='all')

        if df.empty:
            del chunk, proc, df; gc.collect(); continue

        # 배치 COPY
        if batch_cols is None:
            batch_cols = [c for c in target_order if c in df.columns]
            # 반드시 PK 포함
            if not {'device_no','time'}.issubset(set(batch_cols)):
                print("[SKIP] BMS batch_cols missing PK after header normalization")
                del chunk, proc, df; gc.collect()
                batch_cols = None
                continue

            # 디버깅 로그
            # print("[DEBUG][BMS] batch_cols:", batch_cols)

        df[batch_cols].to_csv(batch_buf, index=False, header=False, na_rep='\\N')
        rows_in_batch += len(df)

        if rows_in_batch >= BATCH_SIZE:
            _copy_into_stage_buffered(conn, stage, batch_cols, batch_buf)
            batch_buf = StringIO(); rows_in_batch = 0

        total += len(df)
        del chunk, proc, df; gc.collect()

    # flush remain
    if rows_in_batch > 0 and batch_cols is not None:
        _copy_into_stage_buffered(conn, stage, batch_cols, batch_buf)

    if total > 0 and batch_cols is not None:
        ensure_devices_from_stage(conn, stage)
        _merge_stage_into_target(conn, stage, target, batch_cols, pk, do_update)

    # cleanup
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {stage}_dedup;")
            cur.execute(f"DROP TABLE IF EXISTS {stage};")
        conn.commit()
    except Exception:
        try: conn.rollback()
        except Exception: pass

    return total

def process_gps_chain_direct(conn, root_zip_bytes: bytes, chain: List[str], chunksize: int, do_update: bool) -> int:
    stage = 'gps_stage'
    target = 'aicar.gps'
    schema_cols = [c for c in GPS_COLUMNS]
    pk = ['device_no','time']

    _create_stage_with_index(conn, stage, target, schema_cols)

    total = 0
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.attname
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = %s AND n.nspname = %s
              AND a.attnum > 0 AND NOT a.attisdropped
            ORDER BY a.attnum;
        """, (target.split('.')[-1], 'aicar'))
        target_order = [r[0] for r in cur.fetchall()]

    batch_buf = StringIO()
    batch_cols: Optional[List[str]] = None
    rows_in_batch = 0
    BATCH_SIZE = max(50_000, chunksize)

    for raw in iter_csv_chunks_from_chain(root_zip_bytes, chain, sep='\t', header=None, chunksize=chunksize):
        if raw is None or len(raw)==0:
            continue
        raw = drop_trailing_footer_if_any(raw)
        if len(raw) < MIN_ROWS_TO_SAVE:
            del raw; gc.collect(); continue
        final_df = preprocess_gps_df(raw)
        if final_df.empty or len(final_df) < MIN_ROWS_TO_SAVE:
            del raw, final_df; gc.collect(); continue

        df = clean_data_for_postgres(final_df)

        if 'device_no' not in df.columns or 'time' not in df.columns:
            print("[SKIP] GPS chunk without required PK columns (device_no/time)")
            del raw, final_df, df; gc.collect(); continue

        dev = df['device_no'].astype(str).str.strip()
        df = df[(dev.str.len() > 0) & (~df['time'].isna())]
        df = df.dropna(how='all')

        if df.empty:
            del raw, final_df, df; gc.collect(); continue

        if batch_cols is None:
            batch_cols = [c for c in target_order if c in df.columns]
            if not {'device_no','time'}.issubset(set(batch_cols)):
                print("[SKIP] GPS batch_cols missing PK")
                del raw, final_df, df; gc.collect()
                batch_cols = None
                continue
            # print("[DEBUG][GPS] batch_cols:", batch_cols)

        df[batch_cols].to_csv(batch_buf, index=False, header=False, na_rep='\\N')
        rows_in_batch += len(df)

        if rows_in_batch >= BATCH_SIZE:
            _copy_into_stage_buffered(conn, stage, batch_cols, batch_buf)
            batch_buf = StringIO(); rows_in_batch = 0

        total += len(df)
        del raw, final_df, df; gc.collect()

    if rows_in_batch > 0 and batch_cols is not None:
        _copy_into_stage_buffered(conn, stage, batch_cols, batch_buf)

    if total > 0 and batch_cols is not None:
        ensure_devices_from_stage(conn, stage)
        _merge_stage_into_target(conn, stage, target, batch_cols, pk, do_update)

    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {stage}_dedup;")
            cur.execute(f"DROP TABLE IF EXISTS {stage};")
        conn.commit()
    except Exception:
        try: conn.rollback()
        except Exception: pass

    return total

# =========================
# main
# =========================
def main():
    ap = argparse.ArgumentParser(description="MinIO ZIP → (전처리) → PostgreSQL 원패스 적재")
    # MinIO
    ap.add_argument('--input-bucket', required=True)
    ap.add_argument('--input-prefix', required=True)
    ap.add_argument('--service-name', default=os.getenv('MINIO_SERVICE_NAME','s3'))
    ap.add_argument('--endpoint-url', default=os.getenv('MINIO_ENDPOINT',''))
    ap.add_argument('--access-key', default=os.getenv('MINIO_ACCESS_KEY',''))
    ap.add_argument('--secret-key', default=os.getenv('MINIO_SECRET_KEY',''))

    # 옵션
    ap.add_argument('--include-nested', action='store_true', default=True)
    ap.add_argument('--no-include-nested', dest='include_nested', action='store_false')
    ap.add_argument('--chunk', type=int, default=100_000)
    ap.add_argument('--update', type=int, default=1, help='1=UPSERT, 0=DO NOTHING')

    # Direct(DB)
    ap.add_argument('--direct', action='store_true', help='중간 CSV 미저장, 바로 DB 적재')
    ap.add_argument('--db-host', default=os.getenv('PGHOST'))
    ap.add_argument('--db-port', type=int, default=int(os.getenv('PGPORT', '5432')))
    ap.add_argument('--db-name', default=os.getenv('PGDATABASE'))
    ap.add_argument('--db-user', default=os.getenv('PGUSER'))
    ap.add_argument('--db-pass', default=os.getenv('PGPASSWORD'))

    args = ap.parse_args()

    if not args.direct:
        print("[ERR] 이 스크립트는 현재 --direct 원패스 모드만 지원합니다. --direct 를 지정하세요.")
        sys.exit(1)

    # DB 필수 체크
    for k in ('db_host','db_port','db_name','db_user','db_pass'):
        if getattr(args, k) in (None, ''):
            print(f"[ERR] --{k.replace('_','-')} 설정이 필요합니다.")
            sys.exit(2)

    # MinIO 연결
    s3, s3_client = get_s3(args.service_name, args.endpoint_url, args.access_key, args.secret_key)

    # ZIP 목록
    print(f"[SCAN] bucket={args.input_bucket}, prefix={args.input_prefix} 에서 zip 찾는 중..")
    zip_keys = list_zip_keys(s3_client, args.input_bucket, args.input_prefix)
    print(f"[INFO] zip files: {len(zip_keys)}")

    if not zip_keys:
        print("[WARN] ZIP 없음. 종료.")
        sys.exit(0)

    # DB 연결 및 테이블 보장
    conn = connect_db(args.db_host, args.db_port, args.db_name, args.db_user, args.db_pass)
    try:
        ensure_base_tables(conn)
    except Exception as e:
        print(f"[ERR] ensure_base_tables: {e}")
        try: conn.rollback()
        except Exception: pass
        conn.close()
        sys.exit(3)

    processed_files = 0
    inserted_rows = 0

    for i, key in enumerate(zip_keys, 1):
        print(f"\n=== [{i}/{len(zip_keys)}] {key} ===")
        try:
            root_zip_bytes = get_object_bytes(s3_client, args.input_bucket, key)
        except Exception as e:
            print(f"[ERR] get_object {key}: {e}")
            continue

        chains = walk_all_csv_chains(root_zip_bytes, include_nested=args.include_nested)
        if not chains:
            print("[INFO] CSV 멤버 없음")
            continue

        for chain, csvname in chains:
            member = "/".join(chain)
            cat = detect_file_category(member)
            if cat == 'skip':
                continue
            print(f"[LOAD] {member} ({cat})")

            try:
                if cat == 'bms':
                    rows = process_bms_chain_direct(conn, root_zip_bytes, chain, chunksize=args.chunk, do_update=bool(args.update))
                else:
                    rows = process_gps_chain_direct(conn, root_zip_bytes, chain, chunksize=args.chunk, do_update=bool(args.update))
                inserted_rows += rows
                processed_files += 1
                print(f"[DONE] rows staged→merged: {rows:,}")
            except Exception as e:
                print(f"[ERR] load {member}: {e}")
                try: conn.rollback()
                except Exception: pass

        # 가벼운 통계 갱신(실패해도 무시)
        try:
            with conn.cursor() as cur:
                cur.execute("ANALYZE aicar.bms;")
                cur.execute("ANALYZE aicar.gps;")
            conn.commit()
        except Exception:
            try: conn.rollback()
            except Exception: pass

    print("\n========================")
    print(f"Processed files: {processed_files}")
    print(f"Inserted rows  : {inserted_rows:,}")
    print("========================\n")

    try: conn.close()
    except Exception: pass


if __name__ == '__main__':
    main()

