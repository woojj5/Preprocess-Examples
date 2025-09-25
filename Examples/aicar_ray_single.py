# aicar_ray_single_notemp.py
# - E:\origin만 스캔 (드라이브 가드)
# - 임시 ZIP 미사용: 중첩 ZIP은 메모리 BytesIO로만 접근
# - 중첩 ZIP 재귀 → entry_chain 기반 병렬 처리 (Ray)
# - BMS/GPS 전처리 + 이상치 처리 + 체크포인트 + 인코딩 폴백
# - 출력 네이밍 모드(기본 zip_stem), 개별 CSV 저장
# ------------------------------------------------------------

import os
import io
import re
import gc
import csv
import json
import time
import hashlib
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import ray

# =========================
# 기본 설정 (환경변수로 덮어쓰기 가능)
# =========================
DEFAULT_IN_ROOT = r""
DEFAULT_OUT_DIR = r""
DEFAULT_CPUS = max(1, min(6, (os.cpu_count() or 4)))
DEFAULT_CHUNKSIZE = 25_000
DEFAULT_INCLUDE_NESTED_ZIP = True
MIN_ROWS_TO_SAVE = 2

# === 출력 네이밍 모드 ===
# 'zip_stem'   : <out>/<ZIP파일이름(확장자X)>/<zip내부경로>.csv (추천)
# 'rel_zip'    : <out>/<in_root 기준 ZIP의 상대경로>/...        (폴더 구조 유지)
# 'member_only': <out>/<zip내부경로>.csv                        (ZIP간 충돌 주의)
# 'flat_dunder': <out>/<zipStem>__<member경로를__로평탄화>.csv  (평탄, 충돌 적음)
OUTPUT_NAMING = 'zip_stem'

GPS_COLUMNS = ['device_no', 'time', 'direction', 'fuel_pct', 'hdop', 'lat', 'lng', 'mode', 'source', 'speed', 'state']

# =========================
# 공통 유틸 & 전처리 함수
# =========================
def _fix_year_vectorized(s: pd.Series) -> pd.Series:
    s = s.astype('string').str.strip()
    mask_two = s.str.match(r'^\d{2}[-/.]')
    s = s.mask(mask_two, s.str.replace(r'^(\d{2})([-/.])', r'20\1\2', regex=True))
    return pd.to_datetime(s, errors='coerce')

def looks_like_footer_row(row: pd.Series) -> bool:
    try:
        s = ' '.join([str(x) for x in row.tolist()]).strip().lower()
    except Exception:
        s = str(row.astype(str).to_list()).lower()
    footer_patterns = [
        r'\(\s*\d+\s*rows?\s*\)', r'\brows?\b', r'#+', r'\bsummary\b', r'\btotal\b', r'합계', r'\b총\s*\d+'
    ]
    if s and any(re.search(p, s) for p in footer_patterns):
        return True
    try:
        num = pd.to_numeric(row, errors='coerce')
        zero_ratio = (num.fillna(0) == 0).mean()
    except Exception:
        zero_ratio = 0.0
    blank_ratio = row.astype(str).str.strip().isin(['', 'nan', 'none']).mean()
    return (zero_ratio >= 0.95 and blank_ratio >= 0.80)

def drop_trailing_footer_if_any(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        last = df.iloc[-1]
        if looks_like_footer_row(last):
            return df.iloc[:-1].copy()
    except Exception:
        pass
    return df

def has_zero_latlng(df: pd.DataFrame) -> bool:
    if 'lat' not in df.columns or 'lng' not in df.columns:
        return False
    lat = pd.to_numeric(df['lat'], errors='coerce')
    lng = pd.to_numeric(df['lng'], errors='coerce')
    return ((lat == 0) | (lng == 0)).fillna(False).any()

def detect_file_category(filename: str) -> str:
    f = filename.lower()
    if 'bms' in f: return 'bms'
    if 'gps' in f: return 'gps'
    return 'skip'

def drop_vim(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if str(c).strip().lower() == 'vin'], errors='ignore')

def expand_list_inplace(df_in: pd.DataFrame, list_keyword: str, base_prefix: str, fixed_len: Optional[int] = None) -> pd.DataFrame:
    df = df_in.copy()
    cols = list(df.columns)
    list_col = next((c for c in cols if list_keyword in str(c).lower()), None)
    if not list_col:
        return df
    ser = df[list_col].fillna('').astype('string')
    if fixed_len is None:
        max_len = ser.str.count(',').fillna(0).astype(int).max() + 1 if len(ser) else 0
    else:
        max_len = int(fixed_len)
    if max_len <= 0:
        return df.drop(columns=[list_col])
    tmp = ser.str.split(',', n=max_len-1, expand=True)
    if tmp.shape[1] < max_len:
        for _ in range(max_len - tmp.shape[1]):
            tmp[tmp.shape[1]] = pd.NA
    tmp = tmp.iloc[:, :max_len]
    tmp.columns = [f"{base_prefix}_{i+1}" for i in range(max_len)]
    tmp = tmp.apply(pd.to_numeric, errors='coerce')
    insert_at = cols.index(list_col)
    left = df.iloc[:, :insert_at]
    right = df.iloc[:, insert_at+1:]
    df_out = pd.concat([left, tmp, right], axis=1)
    return df_out

def apply_outliers_bms(df: pd.DataFrame) -> pd.DataFrame:
    if 'odometer' in df.columns:
        df.loc[(pd.to_numeric(df['odometer'], errors='coerce') <= 0) |
               (pd.to_numeric(df['odometer'], errors='coerce') > 2_000_000), 'odometer'] = np.nan
    num_cols = [c for c in df.columns if c not in {'device_no','measured_month','time','msg_time','start_time'}]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df

def apply_outliers_gps(df: pd.DataFrame) -> pd.DataFrame:
    for c in ['speed','hdop','direction','fuel_pct','lat','lng']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if 'speed' in df.columns:
        s = df['speed']
        df.loc[(s < 0) | (s > 300), 'speed'] = np.nan
    if 'hdop' in df.columns:
        h = df['hdop']
        df.loc[(h < 0) | (h > 100), 'hdop'] = np.nan
    return df

def preprocess_bms_chunk_keep_order(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = drop_trailing_footer_if_any(df)
    df = df.replace('', pd.NA)
    non_num = {'device_no', 'measured_month', 'time', 'msg_time', 'start_time', 'int_temp'}
    for col in df.columns:
        if col not in non_num:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = apply_outliers_bms(df)
    for dt_col in ['time', 'msg_time', 'measured_month', 'start_time']:
        if dt_col in df.columns:
            df[dt_col] = _fix_year_vectorized(df[dt_col])
    if 'device_no' in df.columns:
        df['device_no'] = df['device_no'].astype(str).str.replace(r'[\r\n]', '', regex=True).str.strip()
    df = drop_vim(df)
    df = drop_trailing_footer_if_any(df)
    return df.drop_duplicates()

def preprocess_gps_df_keep_order(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=GPS_COLUMNS)
    df = drop_trailing_footer_if_any(df)
    if len(df.columns) == 1 and df.iloc[:, 0].astype(str).str.contains(r'\|').any():
        split_rows = df.iloc[:, 0].str.split('|', expand=True)
        if len(split_rows.columns) > 2:
            split_rows = split_rows.iloc[2:].reset_index(drop=True)
        df = split_rows
    n = min(len(df.columns), len(GPS_COLUMNS))
    df = df.iloc[:, :n]
    df.columns = GPS_COLUMNS[:n]
    df = df.replace('', pd.NA)
    if 'time' in df.columns:
        df['time'] = _fix_year_vectorized(df['time'])
    for col in ['lat', 'lng', 'speed', 'direction', 'hdop', 'fuel_pct']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if 'device_no' in df.columns:
        df['device_no'] = df['device_no'].astype(str).str.replace(r'[\r\n]', '', regex=True).str.strip()
    if 'time' in df.columns:
        try:
            df = df.sort_values(by='time', ascending=True, na_position='last').reset_index(drop=True)
        except Exception:
            pass
    for c in GPS_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df = apply_outliers_gps(df)
    df = drop_vim(df)
    df = drop_trailing_footer_if_any(df)
    return df.drop_duplicates()

def clean_data_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    dfc = drop_trailing_footer_if_any(dfc)
    ts_cols = ['time', 'measured_month', 'msg_time', 'start_time']
    for col in ts_cols:
        if col in dfc.columns:
            dfc[col] = dfc[col].ffill().fillna(pd.Timestamp('1970-01-01 00:00:00'))
    string_cols = {'device_no', 'msg_id', 'mode', 'source', 'state'}
    ts_set = set(ts_cols)
    for col in dfc.columns:
        if col not in string_cols and col not in ts_set:
            dfc[col] = pd.to_numeric(dfc[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
            dfc[col] = dfc[col].fillna(0).astype('float32')
    for col in string_cols:
        if col in dfc.columns:
            ser = dfc[col].astype(str)
            ser = ser.replace(['nan', 'NaN', 'None'], '')
            ser = ser.replace({'^\s+$': ''}, regex=True)
            dfc[col] = ser
    dfc = drop_vim(dfc)
    dfc = drop_trailing_footer_if_any(dfc)
    return dfc

# =========================
# ZIP 탐색 & 체인 처리 (임시파일 없음)
# =========================
def _is_trash_member(name: str) -> bool:
    lower = name.lower()
    if lower.startswith('__macosx/'):
        return True
    base = Path(name).name
    if base.startswith('._'):
        return True
    return False

def list_members_in_chain(zip_path: str, chain: List[str]) -> List[str]:
    """
    chain으로 지정된 ZIP 내부로 내려간 뒤, 그 ZIP의 1-depth 멤버 목록 반환.
    chain = []  => 최상위 ZIP
    chain = ["A.zip"] => 최상위의 A.zip 안의 멤버들
    """
    curr: Any
    curr = zipfile.ZipFile(zip_path, 'r')
    try:
        for i, part in enumerate(chain):
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
            lower = name.lower()
            if lower.endswith('.csv') or lower.endswith('.zip'):
                out.append(name)
        return out
    finally:
        try:
            curr.close()
        except Exception:
            pass

def walk_all_csv_chains(root_zip: str, include_nested: bool = True) -> List[Tuple[List[str], str]]:
    """
    최상위 ZIP(root_zip)에서 시작해, CSV 멤버까지의 entry_chain을 반환.
    반환: [(entry_chain, final_member_name), ...]
    - entry_chain은 최상위 기준 경로 리스트(중첩 zip 경로들 + 마지막 csv 파일명 포함 X)
    - final_member_name은 마지막 단계의 파일명(확장자 .csv)
    """
    result: List[Tuple[List[str], str]] = []
    stack: List[List[str]] = [[]]  # 각 요소는 zip 경로의 체인
    while stack:
        chain = stack.pop()
        members = list_members_in_chain(root_zip, chain)
        for m in members:
            if m.lower().endswith('.csv'):
                result.append((chain + [m], m))
            elif include_nested and m.lower().endswith('.zip'):
                stack.append(chain + [m])
    # 체인 마지막 요소가 csv인 항목만 남기되, 반환 형태는 (체인전체, csv파일명)
    return result

def iter_csv_chunks_from_chain(root_zip: str, entry_chain: List[str], **read_kwargs):
    """
    최상위 ZIP(root_zip)에서 entry_chain을 따라 BytesIO로만 내려가 최종 CSV를 청크로 read.
    """
    encodings = [read_kwargs.pop('encoding', 'utf-8-sig'), 'cp949', 'euc-kr', 'latin1']
    chunksize = read_kwargs.pop('chunksize', DEFAULT_CHUNKSIZE)
    # 마지막 요소는 csv 이름
    assert entry_chain and entry_chain[-1].lower().endswith('.csv')
    last_csv = entry_chain[-1]

    # 최상위 zip 열기
    with zipfile.ZipFile(root_zip, 'r') as zf:
        # 중간 zip들을 순차적으로 BytesIO로 열기
        current = zf
        opened_stack = []  # zipfile 객체 스택(닫기 용)
        try:
            # chain을 따라가되 마지막 csv는 열지 않고 이름만 보관
            for part in entry_chain[:-1]:
                with current.open(part) as f:
                    data = f.read()
                if current is not zf:
                    opened_stack.append(current)
                current = zipfile.ZipFile(io.BytesIO(data), 'r')
            # 이제 current는 csv를 포함한 zip
            for enc in encodings:
                try:
                    with current.open(last_csv, 'r') as raw:
                        text = io.TextIOWrapper(raw, encoding=enc, newline='')
                        reader = pd.read_csv(text, chunksize=chunksize, **read_kwargs)
                        for chunk in reader:
                            yield chunk
                    return
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    # 다른 에러면 다음 인코딩으로 시도, 최종 실패 시 raise
                    last_err = e
                    continue
            raise last_err  # type: ignore
        finally:
            try:
                if current is not zf:
                    current.close()
            except Exception:
                pass
            for z in opened_stack:
                try:
                    z.close()
                except Exception:
                    pass

# =========================
# 출력 경로 / 체크포인트 (anchor=최상위 ZIP)
# =========================
def _safe_part(s: str) -> str:
    return re.sub(r'[:\\/*?"<>|\r\n]+', '_', s)

def safe_out_path(out_dir: str, anchor_zip: str, member_path_in_zip: str) -> str:
    """
    member_path_in_zip: zip 내부 전체 경로 문자열 (체인 기준 /로 join)
    """
    out_dir = Path(out_dir)
    zip_path = Path(anchor_zip)
    mem_parts = [_safe_part(p) for p in Path(member_path_in_zip).parts]

    mode = OUTPUT_NAMING
    if mode == 'zip_stem':
        base = _safe_part(zip_path.stem)
        out_path = out_dir / base / Path(*mem_parts)
    elif mode == 'rel_zip':
        try:
            in_root_abs = Path(os.environ.get("IN_ROOT", DEFAULT_IN_ROOT)).resolve()
            rel = zip_path.resolve().relative_to(in_root_abs)
        except Exception:
            rel = Path(_safe_part(zip_path.stem))
        rel_parts = [_safe_part(p) for p in rel.parts]
        out_path = out_dir.joinpath(*rel_parts) / Path(*mem_parts)
    elif mode == 'member_only':
        out_path = out_dir / Path(*mem_parts)
    elif mode == 'flat_dunder':
        flat = _safe_part(zip_path.stem) + '__' + '__'.join(mem_parts)
        out_path = out_dir / flat
    else:
        base = _safe_part(zip_path.stem)
        out_path = out_dir / base / Path(*mem_parts)

    out_path = out_path.with_suffix('.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return str(out_path)

def _chk_root(out_dir: str) -> Path:
    p = Path(out_dir) / ".checkpoint"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _member_key(anchor_zip: str, member_path_in_zip: str) -> str:
    h = hashlib.sha1((str(Path(anchor_zip)) + "::" + member_path_in_zip).encode('utf-8')).hexdigest()
    return h

def chk_paths(out_dir: str, anchor_zip: str, member_path_in_zip: str) -> Dict[str, Path]:
    root = _chk_root(out_dir)
    key = _member_key(anchor_zip, member_path_in_zip)
    dirp = root / key
    dirp.mkdir(parents=True, exist_ok=True)
    return {
        "dir": dirp,
        "started": dirp / "STARTED",
        "done": dirp / "DONE",
        "meta": dirp / "meta.txt",
    }

def mark_started(paths: Dict[str, Path], meta: Dict[str, Any]):
    if not paths["started"].exists():
        paths["started"].write_text("started", encoding="utf-8")
    try:
        lines = [f"{k}={v}" for k, v in meta.items()]
        paths["meta"].write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass

def mark_done(paths: Dict[str, Path]):
    paths["done"].write_text("done", encoding="utf-8")

def is_done(paths: Dict[str, Path]) -> bool:
    return paths["done"].exists()

# =========================
# 멤버 처리 (원격 태스크) — 임시파일 없이 체인으로 읽기
# =========================
@ray.remote(num_cpus=1)
def process_member_remote(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    spec = {"anchor_zip":(최상위zip), "entry_chain":[...csv], "out_dir":..., "chunksize": int}
    return dict(member_path_in_zip, output_path, status, reason)
    """
    anchor_zip = spec["anchor_zip"]
    entry_chain: List[str] = list(spec["entry_chain"])
    out_dir = spec["out_dir"]
    chunksize = int(spec.get("chunksize", DEFAULT_CHUNKSIZE))

    # zip 내부 경로 문자열 (체인 join)
    member_path_in_zip = "/".join(entry_chain)

    cps = chk_paths(out_dir, anchor_zip, member_path_in_zip)
    out_path = safe_out_path(out_dir, anchor_zip, member_path_in_zip)

    try:
        if is_done(cps):
            return {"member": member_path_in_zip, "output_path": out_path if Path(out_path).exists() else None,
                    "status": "ok", "reason": "already_done"}

        # 재시작 시 일관성 위해 기존 출력 제거
        try:
            if Path(out_path).exists():
                Path(out_path).unlink()
        except Exception:
            pass

        mark_started(cps, {"anchor_zip": anchor_zip, "member": member_path_in_zip})

        last_name = entry_chain[-1]
        cat = detect_file_category(last_name)
        if cat == 'skip':
            return {"member": member_path_in_zip, "output_path": None, "status": "skip", "reason": "category_skip"}

        if cat == 'bms':
            # 1패스: 최대 길이 탐지
            max_cv = 0; max_mt = 0
            for chunk in iter_csv_chunks_from_chain(anchor_zip, entry_chain, sep='|', header=0, dtype=str, chunksize=chunksize):
                cols_lower = [str(c).lower() for c in chunk.columns]
                if any('cell_volt_list' in c for c in cols_lower):
                    ser = chunk[[c for c in chunk.columns if 'cell_volt_list' in str(c).lower()][0]].fillna('').astype(str)
                    cv = (ser.str.count(',').fillna(0).astype(int).max() + 1) if len(ser) else 0
                    max_cv = max(max_cv, int(cv))
                if any('mod_temp_list' in c for c in cols_lower):
                    ser = chunk[[c for c in chunk.columns if 'mod_temp_list' in str(c).lower()][0]].fillna('').astype(str)
                    mt = (ser.str.count(',').fillna(0).astype(int).max() + 1) if len(ser) else 0
                    max_mt = max(max_mt, int(mt))
                del chunk
            header_written = False
            wrote_any = False
            for chunk in iter_csv_chunks_from_chain(anchor_zip, entry_chain, sep='|', header=0, dtype=str, chunksize=chunksize):
                if len(chunk) > 0 and chunk.index.min() == 0 and chunk.iloc[0].astype(str).str.contains('-').any():
                    chunk = chunk.drop(0)
                chunk.columns = [str(c).strip() for c in chunk.columns]
                if any('cell_volt_list' in str(c).lower() for c in chunk.columns):
                    chunk = expand_list_inplace(chunk, 'cell_volt_list', 'cell_volt', fixed_len=max_cv)
                if any('mod_temp_list' in str(c).lower() for c in chunk.columns):
                    chunk = expand_list_inplace(chunk, 'mod_temp_list', 'mod_temp', fixed_len=max_mt)
                chunk = drop_trailing_footer_if_any(chunk)
                proc = preprocess_bms_chunk_keep_order(chunk)
                if proc.empty or len(proc) < MIN_ROWS_TO_SAVE:
                    del chunk, proc; gc.collect(); continue
                df_clean = clean_data_for_csv(proc)
                if df_clean.empty or len(df_clean) < MIN_ROWS_TO_SAVE:
                    del chunk, proc, df_clean; gc.collect(); continue
                df_clean.to_csv(out_path, mode='a', index=False, header=not header_written, encoding='utf-8-sig')
                header_written = True; wrote_any = True
                del chunk, proc, df_clean; gc.collect()
            if not wrote_any:
                return {"member": member_path_in_zip, "output_path": None, "status": "empty", "reason": "no_valid_rows"}
            mark_done(cps)
            return {"member": member_path_in_zip, "output_path": out_path, "status": "ok", "reason": None}

        elif cat == 'gps':
            header_written = False
            wrote_any = False
            for raw in iter_csv_chunks_from_chain(anchor_zip, entry_chain, sep='\t', header=None, dtype=str, chunksize=chunksize):
                raw = drop_trailing_footer_if_any(raw)
                if len(raw) < MIN_ROWS_TO_SAVE:
                    del raw; gc.collect(); continue
                final_df = preprocess_gps_df_keep_order(raw)
                del raw
                if not final_df.empty and has_zero_latlng(final_df):
                    del final_df; gc.collect()
                    return {"member": member_path_in_zip, "output_path": None, "status": "skip", "reason": "lat/lng_zero"}
                if final_df.empty or len(final_df) < MIN_ROWS_TO_SAVE:
                    del final_df; gc.collect(); continue
                df_clean = clean_data_for_csv(final_df)
                del final_df
                if df_clean.empty or len(df_clean) < MIN_ROWS_TO_SAVE:
                    del df_clean; gc.collect(); continue
                df_clean.to_csv(out_path, mode='a', index=False, header=not header_written, encoding='utf-8-sig')
                header_written = True; wrote_any = True
                del df_clean; gc.collect()
            if not wrote_any:
                return {"member": member_path_in_zip, "output_path": None, "status": "empty", "reason": "no_valid_rows"}
            mark_done(cps)
            return {"member": member_path_in_zip, "output_path": out_path, "status": "ok", "reason": None}

        else:
            return {"member": member_path_in_zip, "output_path": None, "status": "skip", "reason": "category_skip"}

    except Exception as e:
        return {"member": member_path_in_zip, "output_path": None, "status": "error", "reason": f"{e}"}

# =========================
# 메인
# =========================
def main():
    in_root_env = os.environ.get("IN_ROOT", DEFAULT_IN_ROOT)
    out_dir = os.environ.get("OUT_DIR", DEFAULT_OUT_DIR)
    cpus = int(os.environ.get("CPUS", DEFAULT_CPUS))
    chunksize = int(os.environ.get("CHUNKSIZE", DEFAULT_CHUNKSIZE))
    include_nested = os.environ.get("INCLUDE_NESTED", "1") not in ("0","false","False")

    # 절대경로 + 드라이브 가드(E:)
    root = Path(in_root_env).resolve()
    if not root.exists():
        raise FileNotFoundError(f"[ERR] IN_ROOT not found: {root}")
    if root.drive.upper() != "E:":
        raise RuntimeError(f"[ERR] IN_ROOT must be on E: drive, got {root.drive}")
    in_root = str(root)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=cpus)

    # E:\origin 안의 ZIP만 수집
    zip_list = []
    for p in Path(in_root).rglob("*.zip"):
        p = p.resolve()
        if p.drive.upper() != "E:":
            continue
        zip_list.append(str(p))
    if not zip_list:
        print(f"[WARN] ZIP을 찾지 못했습니다: {in_root}")
        ray.shutdown(); return

    print(f"=== 설정 ===")
    print(f"in_root        : {in_root}")
    print(f"out_dir        : {out_dir}")
    print(f"cpus           : {cpus}")
    print(f"chunksize      : {chunksize}")
    print(f"include_nested : {include_nested}")
    print(f"ZIP found      : {len(zip_list)}")
    print(f"OUTPUT_NAMING  : {OUTPUT_NAMING}")
    print("================")

    t_all = time.time()
    for i, z in enumerate(zip_list, 1):
        print(f"\n=== [{i}/{len(zip_list)}] {z} ===")
        t0 = time.time()

        # 이 ZIP의 모든 CSV entry_chain 수집
        chains = walk_all_csv_chains(z, include_nested=include_nested)
        if not chains:
            print("[INFO] CSV 멤버 없음"); continue

        # 멤버 경로 문자열(체인 join) 생성
        specs = [{"anchor_zip": z, "entry_chain": chain, "out_dir": out_dir, "chunksize": chunksize}
                 for (chain, _csvname) in chains]

        futures = [process_member_remote.remote(s) for s in specs]

        results = []
        pending = set(futures)
        collected = 0
        total = len(futures)

        while pending:
            done, pending = ray.wait(list(pending), num_returns=1, timeout=1.0)
            if not done: continue
            (ref,) = done
            res = ray.get(ref)
            results.append(res)
            collected += 1
            if collected % max(1, total//10) == 0 or collected == total:
                print(f"[PROGRESS] {collected}/{total}")

        # 매 ZIP별 매니페스트
        base = Path(out_dir) / (f"{_safe_part(Path(z).relative_to(in_root).__str__())}_manifest")
        with open(base.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        try:
            with open(base.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=["member","output_path","status","reason"])
                w.writeheader(); w.writerows(results)
        except Exception as e:
            print(f"[WARN] manifest CSV 저장 실패: {e}")

        print(f"[DONE] {z} :: members={len(chains)} :: elapsed={time.time()-t0:.1f}s")

    print(f"\n=== ALL DONE in {time.time()-t_all:.1f}s ===")
    ray.shutdown()

if __name__ == "__main__":
    main()
