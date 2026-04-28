from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import logging
import os
import io
import re
from datetime import datetime
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SPEC Overview - With Scan Info Table", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BrowseRequest(BaseModel):
    path: Optional[str] = None

class PlotRequest(BaseModel):
    x_column: str
    y_columns: List[str]
    scans: List[int]
    plot_type: str = "line"
    normalize: bool = False
    log_scale: bool = False

class ExportRequest(BaseModel):
    export_type: str = "all"
    scans: Optional[List[int]] = None
    columns: Optional[List[str]] = None

class FitRequest(BaseModel):
    x_column: str
    y_column: str
    scan: int
    fit_type: str = "gaussian"  # "gaussian" or "lorentzian"

data_store = {
    "combined_data": None,
    "available_columns": [],
    "metadata": {},
    "scan_info": {},  # Store detailed scan information
    "root_path": "/nfs/chess/id4b/",
    "last_plot_data": None
}

# Second file store for comparison
data_store_2 = {
    "combined_data": None,
    "available_columns": [],
    "metadata": {},
    "scan_info": {},
}

def is_likely_spec_file(file_path, file_size: int = -1) -> bool:
    """
    Fast SPEC-file detector.  Reads only the first 512 bytes of the file —
    one NFS round-trip — and counts SPEC header markers.  Accepts an
    optional pre-fetched file_size so the caller can avoid a redundant stat().
    """
    try:
        if file_size < 0:
            file_size = os.path.getsize(file_path)
        if file_size == 0 or file_size > 200 * 1024 * 1024:
            return False
        with open(file_path, 'rb') as f:
            head = f.read(512)
        text = head.decode('utf-8', errors='ignore')
        marker_count = sum(
            1 for line in text.splitlines()
            if line.startswith(('#F', '#E', '#D', '#S', '#L', '#C'))
        )
        return marker_count >= 2
    except (OSError, PermissionError):
        return False

class SPECDataProcessor:
    @staticmethod
    def parse_spec_data(spec_text: str):
        """Parse SPEC data from text with detailed scan information"""
        logger.info("Starting SPEC data parsing...")

        lines = spec_text.strip().split('\n')
        all_data = []
        metadata = {}
        scan_info = {}

        current_scan = None
        current_scan_info = {}
        columns = None

        # ── Collect global motor names (#O) and mnemonics (#o) ───────────────
        # These appear once at the top of the file (before any #S scan header)
        global_motor_names = []      # full names from #O lines, in order
        global_motor_mnemonics = []  # short names from #o lines, in order

        for line in lines:
            line_s = line.strip()
            # Stop collecting once we hit the first scan
            if line_s.startswith('#S '):
                break
            if line_s.startswith('#O'):
                # SPEC uses 2+ spaces as delimiter between motor names so that
                # multi-word names like "Motor 75" or "HPVX2 FOCUS" stay intact.
                rest = line_s.split(None, 1)
                if len(rest) > 1:
                    names = re.split(r'  +', rest[1])
                    global_motor_names.extend(
                        [n.strip() for n in names if n.strip()]
                    )
            elif line_s.startswith('#o'):
                # Mnemonics are always single tokens — plain split is fine
                rest = line_s.split(None, 1)
                if len(rest) > 1:
                    global_motor_mnemonics.extend(rest[1].split())

        metadata['motor_names']     = global_motor_names
        metadata['motor_mnemonics'] = global_motor_mnemonics

        # Extract global metadata
        for line in lines:
            line = line.strip()
            if line.startswith('#F'):
                parts = line.split()
                metadata['filename'] = parts[1] if len(parts) > 1 else 'unknown'
            elif line.startswith('#E'):
                parts = line.split()
                metadata['epoch'] = parts[1] if len(parts) > 1 else ''
                if len(parts) > 1:
                    try:
                        epoch_time = int(parts[1])
                        metadata['start_time'] = datetime.fromtimestamp(epoch_time).strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass
            elif line.startswith('#D'):
                metadata['date'] = line[3:].strip()
            elif line.startswith('#C'):
                metadata['comment'] = line[3:].strip()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith('#S'):
                # Save previous scan info if exists
                if current_scan is not None and current_scan_info:
                    scan_info[current_scan] = current_scan_info.copy()

                # Start new scan
                parts = line.split()
                if len(parts) > 1:
                    try:
                        current_scan = int(parts[1])
                        current_scan_info = {
                            'scan_number': current_scan,
                            'command': ' '.join(parts[2:]) if len(parts) > 2 else 'unknown',
                            'full_command': line,
                            'comments': [],
                            'timestamps': [],
                            'motors': {},
                            'counters': {},
                            'temperature': None,
                            'other_info': {}
                        }
                        logger.info(f"Found scan {current_scan}: {current_scan_info['command']}")
                    except ValueError:
                        current_scan = None

            elif line.startswith('#D') and current_scan is not None:
                timestamp = line[3:].strip()
                current_scan_info['timestamps'].append(timestamp)

            elif line.startswith('#C') and current_scan is not None:
                comment = line[3:].strip()
                current_scan_info['comments'].append(comment)

                # Extract temperature only from "Temperature Setpoint at <value>"
                temp_setpoint_match = re.search(
                    r'[Tt]emperature\s+[Ss]etpoint\s+[Aa]t\s+(\d+\.?\d*)', comment
                )
                if temp_setpoint_match:
                    current_scan_info['temperature'] = temp_setpoint_match.group(1)

            elif line.startswith('#T') and current_scan is not None:
                parts = line.split()
                if len(parts) > 1:
                    current_scan_info['count_time'] = parts[1]
                    if len(parts) > 2:
                        current_scan_info['count_time_desc'] = ' '.join(parts[2:])

            elif line.startswith('#G') and current_scan is not None:
                parts = line.split()
                if len(parts) > 1:
                    current_scan_info['geometry'] = ' '.join(parts[1:])

            elif line.startswith('#P') and current_scan is not None:
                parts = line.split()
                motor_line = parts[0][2:]
                if len(parts) > 1:
                    current_scan_info[f'motors_P{motor_line}'] = ' '.join(parts[1:])
                    # Also accumulate into flat motor_positions list
                    if 'motor_positions' not in current_scan_info:
                        current_scan_info['motor_positions'] = []
                    current_scan_info['motor_positions'].extend(
                        [float(v) if v not in ('?', '-') else None
                         for v in parts[1:]]
                    )

            elif line.startswith('#O') and current_scan is not None:
                parts = line.split()
                motor_line = parts[0][2:]
                if len(parts) > 1:
                    current_scan_info[f'motor_names_O{motor_line}'] = ' '.join(parts[1:])

            elif line.startswith('#J') and current_scan is not None:
                parts = line.split()
                counter_line = parts[0][2:]
                if len(parts) > 1:
                    current_scan_info[f'counter_names_J{counter_line}'] = ' '.join(parts[1:])

            elif line.startswith('#L'):
                column_text = line[3:].strip()
                columns = column_text.split()

                fixed_columns = []
                skip_next = False

                for j, col in enumerate(columns):
                    if skip_next:
                        skip_next = False
                        continue

                    if col == "VBPM" and j+1 < len(columns):
                        next_col = columns[j+1]
                        if next_col in ["VER", "HOR"]:
                            fixed_columns.append(f"VBPM_{next_col}")
                            skip_next = True
                        else:
                            fixed_columns.append(col)
                    else:
                        fixed_columns.append(col)

                columns = fixed_columns
                if current_scan is not None:
                    current_scan_info['data_columns'] = columns.copy()
                logger.info(f"Found {len(columns)} columns")

            elif not line.startswith('#') and line and columns and current_scan is not None:
                try:
                    values = []
                    parts = line.split()

                    for part in parts:
                        try:
                            values.append(float(part))
                        except ValueError:
                            values.append(0.0)

                    if len(values) == len(columns):
                        data_row = dict(zip(columns, values))
                        data_row['scan_number'] = current_scan
                        data_row['source_file'] = metadata.get('filename', 'unknown')
                        all_data.append(data_row)
                    elif len(values) > 0:
                        min_len = min(len(values), len(columns))
                        data_row = dict(zip(columns[:min_len], values[:min_len]))
                        data_row['scan_number'] = current_scan
                        data_row['source_file'] = metadata.get('filename', 'unknown')
                        all_data.append(data_row)

                except Exception as e:
                    continue

            i += 1

        if current_scan is not None and current_scan_info:
            scan_info[current_scan] = current_scan_info.copy()

        logger.info(f"Parsing complete: {len(all_data)} data rows, {len(scan_info)} scans with info")

        if all_data:
            df = pd.DataFrame(all_data)
            available_columns = [col for col in df.columns
                               if col not in ['scan_number', 'source_file']]
            return df, available_columns, metadata, scan_info
        else:
            return pd.DataFrame(), [], metadata, scan_info

    @staticmethod
    def create_plot(data: pd.DataFrame, x_column: str, y_columns: List[str],
                   scans: List[int], plot_type: str = "line",
                   normalize: bool = False, log_scale: bool = False):
        """Create plot and return JSON representation"""
        if data.empty:
            raise ValueError("No data available")

        if not y_columns:
            raise ValueError("Please select at least one Y-axis column")

        missing_cols = [col for col in [x_column] + y_columns if col not in data.columns]
        if missing_cols:
            available = list(data.columns)
            raise ValueError(f"Columns not found: {missing_cols}. Available: {available}")

        plot_data = data[data['scan_number'].isin(scans)]

        if plot_data.empty:
            raise ValueError("No data available for selected scans")

        data_store["last_plot_data"] = {
            "data": plot_data,
            "x_column": x_column,
            "y_columns": y_columns,
            "scans": scans,
            "normalize": normalize
        }

        fig = go.Figure()
        colors = px.colors.qualitative.Set1

        for i, y_col in enumerate(y_columns):
            for j, scan in enumerate(scans):
                scan_data = plot_data[plot_data['scan_number'] == scan].copy()

                if scan_data.empty:
                    continue

                scan_data = scan_data.sort_values(x_column)
                x_data = scan_data[x_column]
                y_data = scan_data[y_col]

                mask = ~(pd.isna(x_data) | pd.isna(y_data))
                x_data = x_data[mask]
                y_data = y_data[mask]

                if len(x_data) == 0:
                    continue

                if normalize and len(y_data) > 1:
                    y_min, y_max = y_data.min(), y_data.max()
                    if y_max != y_min:
                        y_data = (y_data - y_min) / (y_max - y_min)

                name = f"Scan {scan} - {y_col}"
                color = colors[(i * len(scans) + j) % len(colors)]

                if plot_type == 'line':
                    trace = go.Scatter(
                        x=x_data.tolist(), y=y_data.tolist(),
                        mode='lines+markers',
                        name=name,
                        line=dict(color=color),
                        marker=dict(size=4)
                    )
                elif plot_type == 'scatter':
                    trace = go.Scatter(
                        x=x_data.tolist(), y=y_data.tolist(),
                        mode='markers',
                        name=name,
                        marker=dict(color=color, size=6)
                    )
                elif plot_type == 'bar':
                    trace = go.Bar(
                        x=x_data.tolist(), y=y_data.tolist(),
                        name=name,
                        marker=dict(color=color)
                    )

                fig.add_trace(trace)

        y_title = ', '.join(y_columns) if len(y_columns) <= 3 else "Values"
        if normalize:
            y_title += " (Normalized)"

        fig.update_layout(
            title=f"{y_title} vs {x_column}",
            xaxis_title=x_column,
            yaxis_title=y_title,
            width=900,
            height=600,
            hovermode='x unified'
        )

        if log_scale:
            fig.update_yaxes(type="log")

        return json.loads(fig.to_json())


# ── /set_root: update browse root path dynamically ───────────────────────────
@app.post("/set_root")
async def set_root_path(request: BrowseRequest):
    """Update the root browse path dynamically from the path input bar"""
    try:
        if not request.path:
            raise HTTPException(status_code=400, detail="No path provided")
        path = os.path.normpath(os.path.abspath(request.path))
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")
        if not os.path.isdir(path):
            path = os.path.dirname(path)
        data_store["root_path"] = path
        return {"root_path": path}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/browse")
async def browse_directory(request: BrowseRequest):
    """Browse directory with improved SPEC file detection"""
    try:
        root_path = data_store["root_path"]

        if request.path is None or request.path == "":
            current_path = root_path
        else:
            requested_path = os.path.abspath(os.path.join(root_path, request.path))
            if not requested_path.startswith(root_path):
                raise HTTPException(status_code=400, detail="Access denied")
            current_path = requested_path

        if not os.path.exists(current_path):
            raise HTTPException(status_code=404, detail=f"Directory not found: {current_path}")

        if not os.path.isdir(current_path):
            raise HTTPException(status_code=400, detail="Not a directory")

        items = []

        if current_path != root_path:
            parent_path = os.path.dirname(current_path)
            relative_parent = os.path.relpath(parent_path, root_path)
            items.append({
                "name": "..",
                "type": "directory",
                "path": relative_parent if relative_parent != "." else "",
                "is_parent": True
            })

        _SKIP_EXTS = {'.cbf','.tif','.tiff','.h5','.hdf5','.edf','.mar',
                      '.img','.sfrm','.mccd','.nxs','.nx','.png','.jpg',
                      '.jpeg','.gif','.pdf','.zip','.gz','.tar','.bz2',
                      '.py','.pyc','.so','.o','.c','.cpp','.f','.f90'}
        _SPEC_EXTS  = {'.dat','.txt','.spec','.scan','.log','.fio'}
        _SPEC_NAMES = {'align','spec','scan','week','day','run','test'}

        dirs_list  = []
        specs_list = []

        try:
            # os.scandir() gives DirEntry objects with a CACHED stat — far
            # fewer NFS round-trips than listdir() + separate stat calls.
            with os.scandir(current_path) as it:
                entries = sorted(it, key=lambda e: e.name)

            for entry in entries:
                name = entry.name
                if name.startswith('.') or name.lower().endswith('.mac'):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=True):
                        relative_path = os.path.relpath(entry.path, root_path)
                        dirs_list.append({
                            "name": name,
                            "type": "directory",
                            "path": relative_path,
                            "is_parent": False
                        })
                    elif entry.is_file(follow_symlinks=True):
                        st        = entry.stat()
                        file_size = st.st_size
                        mtime     = st.st_mtime
                        ext       = os.path.splitext(name)[1].lower()

                        # Fast-reject non-spec file types (no file open needed)
                        if ext in _SKIP_EXTS:
                            continue

                        is_spec_ext  = ext in _SPEC_EXTS
                        is_spec_name = any(p in name.lower() for p in _SPEC_NAMES)
                        is_no_ext    = (ext == '')

                        # Content check only when extension/name give no signal
                        # and the file is small enough to open cheaply on NFS
                        is_spec_content = False
                        if not is_spec_ext and not is_spec_name:
                            if is_no_ext and file_size < 50 * 1024 * 1024:
                                is_spec_content = is_likely_spec_file(
                                    entry.path, file_size)

                        if not (is_spec_ext or is_spec_name or is_spec_content):
                            continue

                        relative_path = os.path.relpath(entry.path, root_path)
                        file_type = ("file" if is_spec_ext and not is_spec_name
                                     else "spec_file")
                        specs_list.append({
                            "name": name,
                            "type": file_type,
                            "path": relative_path,
                            "size": file_size,
                            "is_parent": False,
                            "is_spec": True,
                            "_mtime": mtime
                        })
                except OSError:
                    continue

            # Directories alphabetically first; spec files newest → oldest
            specs_list.sort(key=lambda x: x["_mtime"], reverse=True)
            for s in specs_list:
                del s["_mtime"]
            items.extend(dirs_list)
            items.extend(specs_list)

        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")

        result = {
            "current_path": os.path.relpath(current_path, root_path) if current_path != root_path else "",
            "items": items,
            "root_path": root_path,
            "total_items": len(items),
            "spec_files": len([i for i in items if i.get("is_spec", False)])
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_file")
async def load_spec_file(request: BrowseRequest):
    """Load a SPEC file from the file system"""
    try:
        if not request.path:
            raise HTTPException(status_code=400, detail="No file path provided")

        root_path = data_store["root_path"]
        file_path = os.path.join(root_path, request.path)

        if not file_path.startswith(root_path):
            raise HTTPException(status_code=400, detail="Access denied")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                spec_text = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    spec_text = f.read()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

        processor = SPECDataProcessor()
        df, columns, metadata, scan_info = processor.parse_spec_data(spec_text)

        if df.empty:
            raise HTTPException(status_code=400, detail="No valid SPEC data found in file")

        data_store["combined_data"] = df
        data_store["available_columns"] = columns
        data_store["metadata"] = metadata
        data_store["scan_info"] = scan_info
        data_store["metadata"]["filename"] = os.path.basename(file_path)
        data_store["metadata"]["full_path"] = file_path

        return {
            "message": f"File {os.path.basename(file_path)} loaded successfully",
            "total_scans": len(df['scan_number'].unique()),
            "available_columns": columns,
            "total_points": len(df),
            "file_path": request.path,
            "scan_numbers": sorted(df['scan_number'].unique().tolist()),
            "scan_info_available": len(scan_info) > 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading file: {str(e)}")

@app.post("/load_sample_data")
async def load_sample_data():
    try:
        spec_data = """#F LuNb6Sn6
#E 1740686986
#D Thu Feb 27 15:09:46 2025
#C fourc  User = chess_id4b

#S 1  flyscan phi 0 365 3650 0.1
#D Thu Jan 29 13:44:24 2026
#C Temperature Setpoint at 300
#T 0.1  (Seconds)
#L Time  Epoch  ic1  ic2  diode  load  pilwide  cesr  pilroi  sampleT  VBPM VER  VBPM HOR  flow  pilroi6  pilroi6w  p10roi  Seconds
2.28882e-05 172.464 34363 19886 16229 9 0 14412 0 300.495 0 0 0.41 0 0 550324 0.1

#S 2  ascan th 10 20 10 0.1
#D Thu Jan 29 14:15:30 2026
#C Temperature Setpoint at 350
#T 0.1  (Seconds)
#L Time  Epoch  ic1  ic2  diode  load  pilwide  cesr  pilroi  sampleT  VBPM VER  VBPM HOR  flow  pilroi6  pilroi6w  p10roi  Seconds
1.78814e-05 216.456 34342 19846 16194 9 0 14393 0 350.495 0 0 0.41 0 0 511810 0.1

#S 3  dscan phi -5 5 20 0.5
#D Thu Jan 29 15:22:15 2026
#C Temperature Setpoint at 300
#T 0.5  (Seconds)
#L Time  Epoch  ic1  ic2  diode  load  pilwide  cesr  pilroi  sampleT  VBPM VER  VBPM HOR  flow  pilroi6  pilroi6w  p10roi  Seconds
7.86781e-06 244.882 34267 19891 16262 9 0 14380 0 300.496 0 0 0.41 0 0 456935 0.5

#S 4  mesh th 5 15 20 phi 0 10 10 0.2
#D Thu Jan 29 16:45:50 2026
#C Temperature Setpoint at 400
#T 0.2  (Seconds)
#L Time  Epoch  ic1  ic2  diode  load  pilwide  cesr  pilroi  sampleT  VBPM VER  VBPM HOR  flow  pilroi6  pilroi6w  p10roi  Seconds
1.90735e-05 280.463 34225 19836 16220 9 0 14366 0 400.497 0 0 0.41 0 0 408949 0.2"""

        processor = SPECDataProcessor()
        df, columns, metadata, scan_info = processor.parse_spec_data(spec_data)

        data_store["combined_data"] = df
        data_store["available_columns"] = columns
        data_store["metadata"] = metadata
        data_store["scan_info"] = scan_info

        return {
            "message": "Sample data loaded successfully",
            "total_scans": len(df['scan_number'].unique()),
            "available_columns": columns,
            "total_points": len(df),
            "scan_info_available": len(scan_info) > 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data_info")
async def get_data_info():
    try:
        if data_store["combined_data"] is None:
            raise HTTPException(status_code=404, detail="No data loaded")

        df = data_store["combined_data"]
        return {
            "total_scans": len(df['scan_number'].unique()),
            "scan_numbers": sorted(df['scan_number'].unique().tolist()),
            "available_columns": data_store["available_columns"],
            "total_points": len(df),
            "sample_name": data_store["metadata"].get("filename", "unknown"),
            "scan_info_available": len(data_store["scan_info"]) > 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scan_info")
async def get_scan_info():
    """Get detailed scan information for display in table"""
    try:
        if not data_store["scan_info"]:
            raise HTTPException(status_code=404, detail="No scan information available")

        scan_table_data = []

        for scan_num, info in data_store["scan_info"].items():
            row = {
                "scan_number": scan_num,
                "command": info.get('command', 'unknown'),
                "timestamp": info.get('timestamps', [''])[0] if info.get('timestamps') else '',
                "temperature": info.get('temperature', ''),
                "count_time": info.get('count_time', ''),
                "comments": '; '.join(info.get('comments', [])),
                "data_points": len(data_store["combined_data"][data_store["combined_data"]['scan_number'] == scan_num]) if data_store["combined_data"] is not None else 0
            }

            motor_info = []
            for key, value in info.items():
                if key.startswith('motors_P') or key.startswith('geometry'):
                    motor_info.append(f"{key}: {value}")
            row["motor_info"] = '; '.join(motor_info) if motor_info else ''

            scan_table_data.append(row)

        scan_table_data.sort(key=lambda x: x['scan_number'])

        return {
            "scan_table": scan_table_data,
            "total_scans": len(scan_table_data)
        }

    except Exception as e:
        logger.error(f"Error getting scan info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/motor_positions")
async def get_motor_positions():
    """Return motor names, mnemonics, and per-scan positions."""
    try:
        if not data_store["scan_info"]:
            raise HTTPException(status_code=404, detail="No scan information available")

        motor_names     = data_store["metadata"].get("motor_names", [])
        motor_mnemonics = data_store["metadata"].get("motor_mnemonics", [])

        # Build a lookup: mnemonic → full name (and vice versa)
        n_motors = max(len(motor_names), len(motor_mnemonics))
        motors_meta = []
        for i in range(n_motors):
            motors_meta.append({
                "index":    i,
                "name":     motor_names[i]     if i < len(motor_names)     else f"Motor {i}",
                "mnemonic": motor_mnemonics[i] if i < len(motor_mnemonics) else f"m{i}",
            })

        scans_data = []
        for scan_num in sorted(data_store["scan_info"].keys()):
            info = data_store["scan_info"][scan_num]
            positions = info.get("motor_positions", [])
            scans_data.append({
                "scan_number": scan_num,
                "command":     info.get("command", ""),
                "positions":   positions,
            })

        return {
            "motors":      motors_meta,
            "scans":       scans_data,
            "n_motors":    n_motors,
            "n_scans":     len(scans_data),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fit")
async def fit_data(request: FitRequest):
    """Fit a single Y column for one scan with Gaussian or Lorentzian"""
    try:
        if data_store["combined_data"] is None:
            raise HTTPException(status_code=404, detail="No data loaded")

        df = data_store["combined_data"]
        scan_data = df[df['scan_number'] == request.scan].copy().sort_values(request.x_column)

        if scan_data.empty:
            raise HTTPException(status_code=404, detail="No data for this scan")

        x = scan_data[request.x_column].values.astype(float)
        y = scan_data[request.y_column].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        if len(x) < 4:
            raise HTTPException(status_code=400, detail="Not enough data points to fit")

        stats = {
            "mean":   float(np.mean(y)),
            "max":    float(np.max(y)),
            "min":    float(np.min(y)),
            "std":    float(np.std(y)),
            "delta":  float(np.max(y) - np.min(y)),
            "peak_x": float(x[np.argmax(y)]),
        }

        x_fit = np.linspace(x[0], x[-1], 400)
        amp0   = float(np.max(y) - np.min(y))
        cen0   = float(x[np.argmax(y)])
        wid0   = float((x[-1] - x[0]) / 4)
        off0   = float(np.min(y))

        try:
            if request.fit_type == "gaussian":
                def model(x, amp, cen, sigma, offset):
                    return amp * np.exp(-(x - cen)**2 / (2 * sigma**2)) + offset

                popt, _ = curve_fit(model, x, y, p0=[amp0, cen0, wid0, off0], maxfev=8000)
                amp, cen, sigma, offset = popt
                fwhm = 2.3548 * abs(sigma)
                stats.update({
                    "fit_type":       "Gaussian",
                    "peak_position":  float(cen),
                    "fwhm":           float(fwhm),
                    "amplitude":      float(amp),
                    "sigma":          float(abs(sigma)),
                    "offset":         float(offset),
                })

            elif request.fit_type == "lorentzian":
                def model(x, amp, cen, gamma, offset):
                    return amp * gamma**2 / ((x - cen)**2 + gamma**2) + offset

                popt, _ = curve_fit(model, x, y, p0=[amp0, cen0, wid0, off0], maxfev=8000)
                amp, cen, gamma, offset = popt
                fwhm = 2 * abs(gamma)
                stats.update({
                    "fit_type":       "Lorentzian",
                    "peak_position":  float(cen),
                    "fwhm":           float(fwhm),
                    "amplitude":      float(amp),
                    "gamma":          float(abs(gamma)),
                    "offset":         float(offset),
                })

            y_fit = model(x_fit, *popt)
            return {"stats": stats, "fit_curve": {"x": x_fit.tolist(), "y": y_fit.tolist()}, "success": True}

        except Exception as fit_err:
            return {"stats": stats, "fit_curve": None, "success": False, "error": str(fit_err)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plot")
async def create_plot(request: PlotRequest):
    try:
        if data_store["combined_data"] is None:
            raise HTTPException(status_code=404, detail="No data loaded")

        processor = SPECDataProcessor()
        plot_json = processor.create_plot(
            data_store["combined_data"],
            request.x_column,
            request.y_columns,
            request.scans,
            request.plot_type,
            request.normalize,
            request.log_scale
        )
        return {"plot": plot_json}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/export_csv")
async def export_csv(request: ExportRequest):
    """Export data as CSV"""
    try:
        if data_store["combined_data"] is None:
            raise HTTPException(status_code=404, detail="No data loaded")

        df = data_store["combined_data"]

        if request.export_type == "all":
            export_df = df.copy()
            filename = f"{data_store['metadata'].get('filename', 'spec_data')}_all.csv"

        elif request.export_type == "plotted" and data_store["last_plot_data"]:
            plot_info = data_store["last_plot_data"]
            export_df = plot_info["data"].copy()
            columns_to_export = ['scan_number'] + [plot_info["x_column"]] + plot_info["y_columns"]
            columns_to_export = [col for col in columns_to_export if col in export_df.columns]
            export_df = export_df[columns_to_export]
            filename = f"{data_store['metadata'].get('filename', 'spec_data')}_plotted.csv"

        elif request.export_type == "selected_scans" and request.scans:
            export_df = df[df['scan_number'].isin(request.scans)].copy()
            if request.columns:
                columns_to_export = ['scan_number'] + [col for col in request.columns if col in export_df.columns]
                export_df = export_df[columns_to_export]
            scans_str = "_".join(map(str, request.scans))
            filename = f"{data_store['metadata'].get('filename', 'spec_data')}_scans_{scans_str}.csv"

        else:
            raise HTTPException(status_code=400, detail="Invalid export configuration")

        if export_df.empty:
            raise HTTPException(status_code=400, detail="No data to export")

        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        return StreamingResponse(
            io.BytesIO(csv_content.encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/export_info")
async def get_export_info():
    """Get information about available export options"""
    try:
        if data_store["combined_data"] is None:
            raise HTTPException(status_code=404, detail="No data loaded")

        df = data_store["combined_data"]

        export_info = {
            "total_rows": len(df),
            "total_scans": len(df['scan_number'].unique()),
            "available_scans": sorted(df['scan_number'].unique().tolist()),
            "available_columns": data_store["available_columns"],
            "has_plotted_data": data_store["last_plot_data"] is not None,
            "filename": data_store["metadata"].get("filename", "spec_data")
        }

        if data_store["last_plot_data"]:
            plot_info = data_store["last_plot_data"]
            export_info["plotted_data_info"] = {
                "x_column": plot_info["x_column"],
                "y_columns": plot_info["y_columns"],
                "scans": plot_info["scans"],
                "rows": len(plot_info["data"])
            }

        return export_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/file_status")
async def file_status():
    """Return modification time and size of the currently loaded file."""
    path = data_store["metadata"].get("full_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No file loaded or file not found")
    stat = os.stat(path)
    return {"mtime": stat.st_mtime, "size": stat.st_size, "path": path}

@app.post("/reload_file")
async def reload_file():
    """Re-parse the currently loaded file from disk (called when file has grown)."""
    path = data_store["metadata"].get("full_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No file loaded or file not found")

    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                spec_text = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise HTTPException(status_code=400, detail="Could not decode file")

    df, columns, metadata, scan_info = SPECDataProcessor.parse_spec_data(spec_text)
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid SPEC data found")

    metadata["full_path"] = path
    metadata["filename"] = os.path.basename(path)
    data_store["combined_data"]     = df
    data_store["available_columns"] = columns
    data_store["metadata"]          = metadata
    data_store["scan_info"]         = scan_info

    return {
        "total_scans":      int(df["scan_number"].nunique()),
        "total_points":     len(df),
        "scan_numbers":     sorted(df["scan_number"].unique().tolist()),
        "available_columns": columns,
    }


# ── Absolute-path browser (used by compare panel; never changes root_path) ────
class AbsBrowseRequest(BaseModel):
    abs_path: str

@app.post("/browse_abs")
async def browse_abs_directory(request: AbsBrowseRequest):
    """Browse a directory by absolute path without changing root_path."""
    try:
        abs_path = os.path.normpath(os.path.abspath(request.abs_path))
        if not os.path.exists(abs_path):
            raise HTTPException(status_code=404, detail=f"Path not found: {abs_path}")
        if not os.path.isdir(abs_path):
            abs_path = os.path.dirname(abs_path)

        items = []
        parent = os.path.dirname(abs_path)
        if parent != abs_path:
            items.append({"name": "..", "type": "directory",
                          "abs_path": parent, "is_parent": True})

        # Extension sets used for classification
        IMAGE_EXTS  = {'.cbf', '.tif', '.tiff', '.h5', '.hdf5', '.edf',
                       '.mar', '.img', '.sfrm', '.mccd', '.nxs', '.nx'}
        SPEC_EXTS   = {'.dat', '.txt', '.spec', '.scan', '.log', '.fio'}
        SPEC_NAMES  = {'align','spec','scan','week','day','run','test'}
        SKIP_EXTS   = IMAGE_EXTS | {'.png','.jpg','.jpeg','.gif','.pdf',
                                    '.zip','.gz','.tar','.bz2',
                                    '.py','.pyc','.so','.o','.c','.cpp'}

        dirs_abs   = []
        specs_abs  = []
        others_abs = []

        # os.scandir() — one cached stat per entry vs. multiple NFS calls
        with os.scandir(abs_path) as it:
            entries = sorted(it, key=lambda e: e.name)

        for entry in entries:
            name = entry.name
            if name.startswith('.') or name.lower().endswith('.mac'):
                continue
            try:
                if entry.is_dir(follow_symlinks=True):
                    dirs_abs.append({"name": name, "type": "directory",
                                     "abs_path": entry.path, "is_parent": False,
                                     "is_spec": False, "file_kind": "dir"})
                elif entry.is_file(follow_symlinks=True):
                    st        = entry.stat()
                    file_size = st.st_size
                    mtime     = st.st_mtime
                    ext       = os.path.splitext(name)[1].lower()
                    no_ext    = (ext == '')
                    is_image  = ext in IMAGE_EXTS

                    # Fast-reject obvious non-spec types before opening file
                    if ext in SKIP_EXTS and not is_image:
                        continue

                    is_ext   = ext in SPEC_EXTS
                    is_named = any(p in name.lower() for p in SPEC_NAMES)
                    # Content sniff only when needed and file is reasonably small
                    is_spec  = is_ext or (is_named and not is_image)
                    if not is_spec and not is_image and no_ext and file_size < 50*1024*1024:
                        is_spec = is_likely_spec_file(entry.path, file_size)

                    file_kind = "image" if is_image else ("spec" if is_spec else "other")
                    rec = {"name": name, "type": "file",
                           "abs_path": entry.path, "size": file_size,
                           "is_spec": is_spec, "is_parent": False,
                           "file_kind": file_kind, "_mtime": mtime}
                    if is_spec:
                        specs_abs.append(rec)
                    else:
                        others_abs.append(rec)
            except OSError:
                continue

        # Dirs alphabetically, spec files newest→oldest, then other files
        specs_abs.sort(key=lambda x: x["_mtime"], reverse=True)
        for e in specs_abs + others_abs:
            e.pop("_mtime", None)
        items.extend(dirs_abs)
        items.extend(specs_abs)
        items.extend(others_abs)

        return {"abs_path": abs_path, "items": items,
                "total_items": len(items),
                "spec_files": sum(1 for i in items if i.get("is_spec"))}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Folder Timeline ───────────────────────────────────────────────────────────
_TIMELINE_IMAGE_EXTS = {'.cbf','.tif','.tiff','.h5','.hdf5','.edf',
                        '.mar','.img','.sfrm','.mccd','.nxs','.nx'}
_TIMELINE_SPEC_EXTS  = {'.dat','.txt','.spec','.scan','.log','.fio'}
_TIMELINE_SPEC_NAMES = {'align','spec','scan','week','day','run','test'}

def _parse_spec_for_timeline(file_path: str, filename: str) -> list:
    """
    Lightweight SPEC header-only parser.  Reads only # lines plus data-line
    counts.  Returns a list of per-scan dicts for the folder timeline table.
    """
    rows        = []
    current     = None
    data_count  = 0

    _ts_fmts = [
        '%a %b %d %H:%M:%S %Y',   # Thu Jan 29 13:44:24 2026
        '%a %b  %d %H:%M:%S %Y',  # single-digit day with extra space
    ]

    def _parse_ts(ts_str):
        for fmt in _ts_fmts:
            try:
                return datetime.strptime(ts_str, fmt).timestamp()
            except ValueError:
                pass
        return 0.0

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                line = raw.rstrip('\n')
                if line.startswith('#S '):
                    if current is not None:
                        current['data_points'] = data_count
                        rows.append(current)
                    parts = line.split()
                    try:
                        snum = int(parts[1]) if len(parts) > 1 else 0
                    except ValueError:
                        snum = 0
                    current    = {
                        'spec_file':       filename,
                        'scan_number':     snum,
                        'command':         ' '.join(parts[2:]) if len(parts) > 2 else '',
                        'timestamp':       '',
                        'timestamp_epoch': 0.0,
                        'temperature':     None,
                        'count_time':      None,
                        'data_points':     0,
                        'comments':        '',
                    }
                    data_count = 0
                elif current is not None:
                    if line.startswith('#D '):
                        ts = line[3:].strip()
                        current['timestamp']       = ts
                        current['timestamp_epoch'] = _parse_ts(ts)
                    elif line.startswith('#T '):
                        parts = line.split()
                        if len(parts) > 1:
                            unit = parts[2] if len(parts) > 2 else ''
                            current['count_time'] = parts[1] + (' ' + unit if unit else '')
                    elif line.startswith('#C '):
                        comment = line[3:].strip()
                        m = re.search(
                            r'[Tt]emperature\s+[Ss]etpoint\s+[Aa]t\s+(\d+\.?\d*)',
                            comment)
                        if m:
                            current['temperature'] = m.group(1)
                        if current['comments']:
                            current['comments'] += '; ' + comment
                        else:
                            current['comments'] = comment
                    elif not line.startswith('#') and line.strip():
                        data_count += 1
        if current is not None:
            current['data_points'] = data_count
            rows.append(current)
    except (OSError, UnicodeDecodeError):
        pass

    return rows


class FolderTimelineRequest(BaseModel):
    abs_path: str

@app.post("/folder_timeline")
async def folder_timeline(request: FolderTimelineRequest):
    """
    Parse every SPEC file in a folder and return all scans sorted by
    scan-timestamp so the user can track the full experiment chronology.
    """
    abs_path = os.path.normpath(os.path.abspath(request.abs_path))
    if not os.path.isdir(abs_path):
        raise HTTPException(status_code=400, detail="Not a directory")

    all_rows: list = []

    try:
        for item in sorted(os.listdir(abs_path)):
            if item.startswith('.') or item.lower().endswith('.mac'):
                continue
            item_path = os.path.join(abs_path, item)
            if not os.path.isfile(item_path):
                continue
            ext     = os.path.splitext(item)[1].lower()
            no_ext  = (ext == '')
            if ext in _TIMELINE_IMAGE_EXTS:
                continue
            # Only consider likely-spec files
            is_spec_ext  = ext in _TIMELINE_SPEC_EXTS
            is_spec_name = any(p in item.lower() for p in _TIMELINE_SPEC_NAMES)
            if not (is_spec_ext or no_ext or is_spec_name):
                continue
            try:
                if not is_likely_spec_file(item_path):
                    continue
            except OSError:
                continue
            rows = _parse_spec_for_timeline(item_path, item)
            all_rows.extend(rows)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Sort chronologically by per-scan #D timestamp
    # Newest scan first (Mon Feb 16 at top, Wed Feb 11 at bottom)
    all_rows.sort(key=lambda r: r.get('timestamp_epoch', 0.0), reverse=True)

    return {"folder": abs_path, "rows": all_rows, "total": len(all_rows)}


# ── Scan data folder finder ───────────────────────────────────────────────────
class FindScanRequest(BaseModel):
    scan_number: int
    # Optional overrides used by the Folder Timeline table (multi-file context)
    spec_file:   Optional[str] = None   # e.g. "NiTaSe2"  (no extension needed)
    spec_parent: Optional[str] = None   # absolute folder that contains the spec file

def _walk_find(start: str, target: str, max_depth: int = 4) -> Optional[str]:
    """
    Fast bounded search using os.walk.  Prunes depth and skips hidden /
    irrelevant directories.  Never touches root_path — always starts from
    the spec file's own parent directory.
    """
    start = os.path.normpath(start)
    base_depth = start.count(os.sep)
    skip_dirs = {'.', '..', '__pycache__', 'lost+found', '.git', '.svn'}

    for dirpath, dirnames, _ in os.walk(start, topdown=True):
        depth = dirpath.count(os.sep) - base_depth
        # Check immediate children for the target name
        if target in dirnames:
            return os.path.join(dirpath, target)
        # Stop descending if we've hit the depth limit
        if depth >= max_depth:
            dirnames.clear()
            continue
        # Prune hidden and irrelevant dirs to keep walk fast
        dirnames[:] = [d for d in dirnames
                       if d not in skip_dirs and not d.startswith('.')]
    return None

@app.post("/find_scan_data")
async def find_scan_data(request: FindScanRequest):
    """
    Fast scan-folder lookup.  Checks likely locations directly (O(1))
    before falling back to a bounded os.walk that starts from the spec
    file's parent — NOT from root_path — keeping NFS calls to a minimum.

    Priority:
      1. Direct path checks for known layouts (raw6M, tiffs, rawpil, data)
      2. Bounded os.walk from spec-file parent (depth ≤ 4)
      3. tiffs/ directory fallback
    """
    scan_num = request.scan_number

    # Timeline rows supply spec_file + spec_parent directly; fall back to data_store
    if request.spec_file and request.spec_parent:
        spec_base   = os.path.splitext(request.spec_file)[0]
        spec_parent = os.path.normpath(request.spec_parent)
    else:
        spec_file   = data_store["metadata"].get("filename", "")
        spec_base   = os.path.splitext(spec_file)[0] if spec_file else ""
        spec_full   = data_store["metadata"].get("full_path", "")
        spec_parent = (os.path.dirname(spec_full) if spec_full
                       else data_store.get("root_path", "/nfs/chess/id4b/"))

    folder_name = f"{spec_base}_{scan_num:03d}" if spec_base else f"scan_{scan_num:03d}"

    # ── Pass 1: direct checks for common known sub-layouts (instant) ──────────
    common_subdirs = ["raw6M", "tiffs", "rawpil", "data", "raw", "images",
                      os.path.join("raw6M", spec_base),
                      os.path.join("raw6M", spec_base, "standard"),
                      os.path.join("tiffs", spec_base)]
    for sub in common_subdirs:
        candidate = os.path.join(spec_parent, sub, folder_name)
        if os.path.isdir(candidate):
            return {"found": True, "path": candidate,
                    "folder": folder_name, "strategy": "direct"}

    # Also check one level up (experiment root may be one dir above spec file)
    exp_root = os.path.dirname(spec_parent)
    for sub in common_subdirs:
        candidate = os.path.join(exp_root, sub, folder_name)
        if os.path.isdir(candidate):
            return {"found": True, "path": candidate,
                    "folder": folder_name, "strategy": "direct_up"}

    # ── Pass 2: bounded os.walk from spec parent (depth ≤ 4) ─────────────────
    found = _walk_find(spec_parent, folder_name, max_depth=4)
    if found:
        return {"found": True, "path": found,
                "folder": folder_name, "strategy": "walk"}

    # ── Pass 3: tiffs/ directory as fallback ──────────────────────────────────
    for base in (spec_parent, exp_root):
        tiffs_dir = os.path.join(base, "tiffs")
        if os.path.isdir(tiffs_dir):
            return {"found": False, "path": tiffs_dir,
                    "folder": folder_name, "strategy": "tiffs_dir"}

    # Nothing useful found
    return {"found": False, "path": spec_parent,
            "folder": folder_name, "strategy": "none"}


# ── File-2 endpoints (compare mode) ──────────────────────────────────────────
@app.post("/load_file2")
async def load_spec_file2(request: BrowseRequest):
    """Load a second SPEC file for comparison"""
    try:
        if not request.path:
            raise HTTPException(status_code=400, detail="No file path provided")

        root_path = data_store["root_path"]
        file_path = request.path if os.path.isabs(request.path) else os.path.join(root_path, request.path)
        file_path = os.path.normpath(file_path)
        if not file_path.startswith(root_path):
            raise HTTPException(status_code=400, detail="Access denied")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        for enc in ("utf-8", "latin-1"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    spec_text = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise HTTPException(status_code=400, detail="Could not decode file")

        processor = SPECDataProcessor()
        df, columns, metadata, scan_info = processor.parse_spec_data(spec_text)
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid SPEC data found in file 2")

        metadata["full_path"] = file_path
        metadata["filename"] = os.path.basename(file_path)
        data_store_2["combined_data"]     = df
        data_store_2["available_columns"] = columns
        data_store_2["metadata"]          = metadata
        data_store_2["scan_info"]         = scan_info

        return {
            "message": f"Compare file {os.path.basename(file_path)} loaded successfully",
            "total_scans": int(df["scan_number"].nunique()),
            "total_points": len(df),
            "available_columns": columns,
            "scan_numbers": sorted(df["scan_number"].unique().tolist()),
            "filename": os.path.basename(file_path),
            "scan_info_available": len(scan_info) > 0
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading file 2: {str(e)}")


@app.get("/data_info2")
async def get_data_info2():
    if data_store_2["combined_data"] is None:
        raise HTTPException(status_code=404, detail="No comparison file loaded")
    df = data_store_2["combined_data"]
    return {
        "total_scans":       int(df["scan_number"].nunique()),
        "scan_numbers":      sorted(df["scan_number"].unique().tolist()),
        "available_columns": data_store_2["available_columns"],
        "total_points":      len(df),
        "sample_name":       data_store_2["metadata"].get("filename", "unknown"),
        "scan_info_available": len(data_store_2["scan_info"]) > 0
    }


@app.post("/plot2")
async def create_plot2(request: PlotRequest):
    """Plot data from the comparison (second) file"""
    try:
        if data_store_2["combined_data"] is None:
            raise HTTPException(status_code=404, detail="No comparison file loaded")
        processor = SPECDataProcessor()
        plot_json = processor.create_plot(
            data_store_2["combined_data"],
            request.x_column,
            request.y_columns,
            request.scans,
            request.plot_type,
            request.normalize,
            request.log_scale
        )
        return {"plot": plot_json}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/scan_info2")
async def get_scan_info2():
    try:
        if not data_store_2["scan_info"]:
            raise HTTPException(status_code=404, detail="No scan information available for file 2")
        scan_table_data = []
        for scan_num, info in data_store_2["scan_info"].items():
            row = {
                "scan_number": scan_num,
                "command":     info.get("command", "unknown"),
                "timestamp":   info.get("timestamps", [""])[0] if info.get("timestamps") else "",
                "temperature": info.get("temperature", ""),
                "count_time":  info.get("count_time", ""),
                "comments":    "; ".join(info.get("comments", [])),
                "data_points": len(data_store_2["combined_data"][
                    data_store_2["combined_data"]["scan_number"] == scan_num
                ]) if data_store_2["combined_data"] is not None else 0
            }
            scan_table_data.append(row)
        scan_table_data.sort(key=lambda x: x["scan_number"])
        return {"scan_table": scan_table_data, "total_scans": len(scan_table_data)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Web Interface ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <title>SPEC Overview &mdash; CHESS ID4B Beamline</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="SPEC Overview: interactive browser-based tool for loading, visualising, and exporting SPEC data files collected at the CHESS ID4B beamline at Cornell University.">
    <meta name="author" content="CHESS ID4B Beamline Team">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --pri:#1e40af; --pri-l:#3b82f6; --pri-d:#1e3a8a;
            --teal:#0d9488; --amber:#d97706; --green:#059669; --red:#dc2626;
            --bg:#f1f5f9; --surf:#ffffff; --surf2:#f8fafc;
            --bdr:#e2e8f0; --bdr2:#cbd5e1;
            --txt:#0f172a; --txt2:#475569; --txt3:#94a3b8;
            --r:10px; --rs:6px;
            --sh:0 1px 3px rgba(0,0,0,.10),0 1px 2px rgba(0,0,0,.06);
            --shm:0 4px 6px rgba(0,0,0,.07),0 2px 4px rgba(0,0,0,.06);
            --shl:0 10px 25px rgba(0,0,0,.10),0 4px 6px rgba(0,0,0,.05);
        }
        *{box-sizing:border-box;margin:0;padding:0}
        body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
             background:var(--bg);color:var(--txt);min-height:100vh;font-size:14px}

        /* ── Top navigation ──────────────────────────────── */
        .topnav{
            background:linear-gradient(135deg,#0c1445 0%,#1e3a8a 100%);
            padding:0 20px; display:flex; align-items:center; gap:14px;
            height:54px; box-shadow:0 2px 12px rgba(0,0,0,.35);
            position:sticky; top:0; z-index:200;
        }
        .nav-logo{display:flex;align-items:center;gap:10px;color:#fff;font-weight:700;font-size:17px;letter-spacing:-.3px;white-space:nowrap}
        .nav-logo-icon{width:30px;height:30px;background:var(--pri-l);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px}
        .nav-divider{width:1px;height:28px;background:rgba(255,255,255,.15);flex-shrink:0}
        .nav-file{flex:1;display:flex;align-items:center;gap:8px;min-width:0}
        .nav-filename{
            background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.15);
            color:rgba(255,255,255,.9);padding:3px 11px;border-radius:20px;
            font-size:12px;font-family:monospace;max-width:320px;
            overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
        }
        .nav-badge{font-size:10px;padding:2px 7px;border-radius:10px;font-weight:700;display:none}
        .nav-badge.loaded{display:inline-block;background:#22c55e;color:#052e16}
        .nav-controls{display:flex;align-items:center;gap:8px;flex-shrink:0}
        #refreshStatus{color:rgba(255,255,255,.75);font-size:11px;white-space:nowrap;display:none}
        .nav-btn{
            background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);
            color:#fff;padding:5px 11px;border-radius:var(--rs);cursor:pointer;
            font-size:12px;font-weight:600;transition:all .15s;white-space:nowrap
        }
        .nav-btn:hover{background:rgba(255,255,255,.22)}
        .nav-btn.stop{background:#dc2626;border-color:#b91c1c}
        .nav-select{
            background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);
            color:#fff;padding:4px 8px;border-radius:var(--rs);font-size:12px;
            cursor:pointer;
        }
        .nav-select option{background:#1e3a8a;color:#fff}

        /* ── App body ────────────────────────────────────── */
        .app-body{max-width:1600px;margin:0 auto;padding:18px 20px}

        /* ── Toolbar ─────────────────────────────────────── */
        .toolbar{
            background:var(--surf);border-radius:var(--r);box-shadow:var(--sh);
            border:1px solid var(--bdr);padding:10px 14px;margin-bottom:14px;
            display:flex;flex-wrap:wrap;gap:6px;align-items:center;
        }
        .toolbar-group{display:flex;align-items:center;gap:6px}
        .tlabel{font-size:10px;font-weight:700;color:var(--txt3);text-transform:uppercase;letter-spacing:.6px;margin-right:2px}
        .tsep{width:1px;height:26px;background:var(--bdr);margin:0 6px}

        /* ── Buttons ─────────────────────────────────────── */
        button{
            display:inline-flex;align-items:center;gap:5px;
            padding:6px 13px;border:1px solid transparent;border-radius:var(--rs);
            cursor:pointer;font-size:13px;font-weight:600;
            transition:all .15s ease;line-height:1.2;font-family:inherit
        }
        .btn-pri{background:var(--pri);color:#fff;border-color:var(--pri-d)}
        .btn-pri:hover{background:var(--pri-l);box-shadow:var(--shm);transform:translateY(-1px)}
        .btn-sec{background:var(--surf);color:var(--txt2);border-color:var(--bdr2)}
        .btn-sec:hover{background:var(--bg);box-shadow:var(--sh);color:var(--txt)}
        .btn-teal{background:var(--teal);color:#fff;border-color:#0f766e}
        .btn-teal:hover{background:#14b8a6;transform:translateY(-1px)}
        .btn-amber{background:var(--amber);color:#fff;border-color:#b45309}
        .btn-amber:hover{background:#f59e0b;transform:translateY(-1px)}
        .btn-green{background:var(--green);color:#fff;border-color:#047857}
        .btn-green:hover{background:#10b981;transform:translateY(-1px)}
        .btn-red{background:var(--red);color:#fff;border-color:#b91c1c}
        .btn-red:hover{background:#ef4444}
        .btn-ghost{background:transparent;color:var(--txt3);border-color:transparent;padding:4px 6px}
        .btn-ghost:hover{background:var(--bg);color:var(--txt)}
        .btn-active{
            background:#eff6ff !important;color:#1d4ed8 !important;
            border-color:#bfdbfe !important;
            box-shadow:0 0 0 3px rgba(59,130,246,.15) !important;
        }
        .btn-action{
            width:100%;padding:13px;font-size:15px;justify-content:center;
            background:linear-gradient(135deg,#1e40af,#3b82f6);
            color:#fff;border:none;border-radius:var(--r);
            box-shadow:var(--shm);
        }
        .btn-action:hover{transform:translateY(-2px);box-shadow:var(--shl);background:linear-gradient(135deg,#1e3a8a,#2563eb)}
        .big-button{grid-column:1/-1}

        /* ── Panels ──────────────────────────────────────── */
        .panel{
            background:var(--surf);border-radius:var(--r);box-shadow:var(--shm);
            border:1px solid var(--bdr);padding:22px;margin-bottom:14px;
            animation:fadeSlide .2s ease;
        }
        @keyframes fadeSlide{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:none}}
        .panel-header{
            display:flex;align-items:center;justify-content:space-between;
            padding-bottom:14px;margin-bottom:16px;
            border-bottom:1px solid var(--bdr);
        }
        .panel-title{font-size:15px;font-weight:700;color:var(--txt);display:flex;align-items:center;gap:8px}
        .panel-sub{font-size:12px;color:var(--txt3);font-weight:400;margin-top:2px}
        .pill{display:inline-flex;align-items:center;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:600}
        .pill-blue{background:#eff6ff;color:#1d4ed8}
        .pill-green{background:#f0fdf4;color:#166534}

        /* ── File browser ────────────────────────────────── */
        .file-browser{
            border:1px solid var(--bdr);border-radius:var(--rs);
            max-height:320px;overflow-y:auto;background:var(--surf2);
        }
        .file-browser::-webkit-scrollbar{width:6px}
        .file-browser::-webkit-scrollbar-thumb{background:var(--bdr2);border-radius:3px}
        .breadcrumb{
            display:flex;align-items:center;gap:4px;
            background:var(--surf2);border:1px solid var(--bdr);
            border-radius:var(--rs);padding:7px 12px;
            font-family:monospace;font-size:11px;color:var(--txt2);margin-bottom:6px;
        }
        .file-item{
            display:flex;align-items:center;gap:10px;padding:9px 14px;
            border-bottom:1px solid var(--bdr);cursor:pointer;transition:background .12s;
            font-size:13px;
        }
        .file-item:last-child{border-bottom:none}
        .file-item:hover{background:#eff6ff}
        .file-item.spec-file{background:#fefce8}
        .file-item.spec-file:hover{background:#fef9c3}
        .icon{font-size:14px;flex-shrink:0;width:18px;text-align:center}
        .name{flex:1;font-weight:500;color:var(--txt);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
        .size{font-size:11px;color:var(--txt3);flex-shrink:0}
        .spec-badge{background:var(--amber);color:#fff;font-size:9px;font-weight:700;padding:2px 6px;border-radius:10px;letter-spacing:.3px}

        /* ── Path/text inputs ────────────────────────────── */
        input[type="text"],input[type="number"]{
            border:1px solid var(--bdr2);border-radius:var(--rs);
            padding:7px 11px;font-size:13px;color:var(--txt);
            background:var(--surf);transition:border-color .15s;width:100%;
            font-family:inherit;
        }
        input:focus{outline:none;border-color:var(--pri-l);box-shadow:0 0 0 3px rgba(59,130,246,.15)}
        #pathInput,#compare2PathInput{font-family:monospace;font-size:12px}

        /* ── Select ──────────────────────────────────────── */
        select{
            width:100%;padding:7px 10px;border:1px solid var(--bdr2);
            border-radius:var(--rs);background:var(--surf);color:var(--txt);
            font-size:13px;cursor:pointer;transition:border-color .15s;font-family:inherit;
        }
        select:focus{outline:none;border-color:var(--pri-l);box-shadow:0 0 0 3px rgba(59,130,246,.15)}
        .multi-select{height:130px}

        /* ── Control grid ────────────────────────────────── */
        .control-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
        .control-group{
            background:var(--surf2);border-radius:var(--rs);
            border:1px solid var(--bdr);padding:13px;transition:border-color .15s;
        }
        .control-group:hover{border-color:#bfdbfe}
        .control-group label{
            display:block;font-size:10px;font-weight:700;color:var(--txt3);
            text-transform:uppercase;letter-spacing:.5px;margin-bottom:7px;
        }
        .checkbox-group{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
        .checkbox-group div{display:flex;align-items:center;gap:6px}
        .checkbox-group input{width:14px;height:14px;cursor:pointer;accent-color:var(--pri)}
        .checkbox-group label{font-size:13px;color:var(--txt2);text-transform:none;letter-spacing:0;margin:0;font-weight:500}

        /* ── Plot area ───────────────────────────────────── */
        .plot-area{
            background:var(--surf);border-radius:var(--r);
            box-shadow:var(--shm);border:1px solid var(--bdr);
            margin-bottom:14px;overflow:hidden;min-height:520px;
        }
        #plot{min-height:520px}

        /* ── Scan table ──────────────────────────────────── */
        .scan-table{width:100%;border-collapse:collapse;font-size:12px}
        .scan-table thead{position:sticky;top:0;z-index:2}
        .scan-table th{
            background:#1e3a8a;color:#fff;padding:9px 10px;
            text-align:left;font-weight:600;font-size:11px;
            text-transform:uppercase;letter-spacing:.3px;
        }
        .scan-table td{padding:8px 10px;border-bottom:1px solid var(--bdr);vertical-align:top}
        .scan-table tr:nth-child(even){background:#f8fafc}
        .scan-table tr:hover{background:#eff6ff}
        .scan-number{font-weight:700;color:var(--pri);text-align:center}
        .scan-link{
            background:none;border:1px solid var(--pri);border-radius:4px;
            color:var(--pri);font-weight:700;padding:2px 8px;cursor:pointer;
            font-size:inherit;font-family:inherit;transition:background .15s,color .15s;
        }
        .scan-link:hover{background:var(--pri);color:#fff;}
        body.dark .scan-link{border-color:var(--pri-l);color:var(--pri-l)}
        body.dark .scan-link:hover{background:var(--pri-l);color:#fff}
        .command{font-family:'Courier New',monospace;font-size:11px;background:var(--bg);padding:2px 6px;border-radius:3px;color:#1e40af}
        .temperature{color:#dc2626;font-weight:600}
        .timestamp{color:var(--txt3);font-size:10px}

        /* ── Export options ──────────────────────────────── */
        .export-option{
            padding:13px 15px;margin:7px 0;
            border:1px solid var(--bdr);border-radius:var(--rs);
            background:var(--surf2);cursor:pointer;transition:all .15s;
        }
        .export-option:hover{border-color:var(--pri-l);background:#eff6ff}
        .export-option input[type="radio"]{accent-color:var(--pri);margin-right:8px}
        .export-option label{font-weight:600;color:var(--txt);cursor:pointer}
        .export-option .description{font-size:12px;color:var(--txt2);margin-top:4px;padding-left:22px}

        /* ── Status toasts ───────────────────────────────── */
        #messages{position:fixed;top:62px;right:16px;z-index:300;display:flex;flex-direction:column;gap:7px;max-width:370px;pointer-events:none}
        .status{
            padding:11px 15px;border-radius:var(--rs);font-size:13px;font-weight:500;
            box-shadow:var(--shm);border-left:4px solid transparent;pointer-events:auto;
            animation:toastIn .2s ease;
        }
        @keyframes toastIn{from{opacity:0;transform:translateX(16px)}to{opacity:1;transform:none}}
        .success{background:#f0fdf4;color:#166534;border-left-color:#22c55e}
        .error{background:#fef2f2;color:#991b1b;border-left-color:#ef4444}
        .info{background:#eff6ff;color:#1e40af;border-left-color:#3b82f6}

        /* ── Notebook ────────────────────────────────────── */
        .notebook-panel{background:#fffbeb;border-color:#fde68a}
        .notebook-entry{background:#fff;border:1px solid #fde68a;border-radius:var(--rs);padding:11px;margin:8px 0}
        .note-time{font-size:10px;color:var(--txt3);margin-bottom:4px}
        #noteInput{width:100%;min-height:80px;resize:vertical;border:1px solid #fde68a;border-radius:var(--rs);padding:10px;font-size:13px;background:#fff;font-family:inherit}
        #noteInput:focus{outline:none;border-color:#f59e0b;box-shadow:0 0 0 3px rgba(245,158,11,.15)}

        /* ── Compare panel ───────────────────────────────── */
        .compare-panel{background:#f0fdfa;border-color:#99f6e4}

        /* ── Fit results ─────────────────────────────────── */
        .fit-panel{background:#fafaff;border-color:#c7d2fe}

        /* ── Motor positions table ───────────────────────── */
        .motor-panel{background:#f0fdf4;border-color:#86efac}
        .motor-table{width:100%;border-collapse:collapse;font-size:12px}
        .motor-table thead{position:sticky;top:0;z-index:2}
        .motor-table th{
            background:#065f46;color:#fff;padding:7px 10px;
            text-align:left;font-weight:600;font-size:11px;
            text-transform:uppercase;letter-spacing:.3px;white-space:nowrap;
        }
        .motor-table th.scan-hdr{background:#047857;text-align:right;min-width:90px}
        .motor-table td{
            padding:6px 10px;border-bottom:1px solid var(--bdr);white-space:nowrap;font-size:12px;
        }
        .motor-table td.mtr-name{
            font-weight:600;color:var(--txt);background:var(--surf2);
            position:sticky;left:0;z-index:1;border-right:2px solid var(--bdr2);min-width:130px;
        }
        .motor-table td.mtr-mnem{
            font-family:'Courier New',monospace;color:#059669;
            background:var(--surf2);min-width:80px;border-right:1px solid var(--bdr);
        }
        .motor-table tr:nth-child(even) td{background:#f0fdf4}
        .motor-table tr:nth-child(even) td.mtr-name,
        .motor-table tr:nth-child(even) td.mtr-mnem{background:#dcfce7}
        .motor-table tr:hover td{background:#bbf7d0!important}
        .motor-table td.pos-val{text-align:right;font-family:'Courier New',monospace;color:#1e40af}
        .motor-table td.pos-zero{text-align:right;font-family:'Courier New',monospace;color:var(--txt3)}
        .motor-table td.pos-na{text-align:center;color:var(--txt3);font-style:italic}
        body.dark .motor-panel{background:#022c22;border-color:#14532d}
        body.dark .motor-table th{background:#064e3b}
        body.dark .motor-table th.scan-hdr{background:#065f46}
        body.dark .motor-table td.mtr-name,
        body.dark .motor-table td.mtr-mnem{background:#0f172a}
        body.dark .motor-table tr:nth-child(even) td{background:#022c22}
        body.dark .motor-table tr:nth-child(even) td.mtr-name,
        body.dark .motor-table tr:nth-child(even) td.mtr-mnem{background:#052e16}
        body.dark .motor-table tr:hover td{background:#14532d!important}

        /* ── Misc ────────────────────────────────────────── */
        .hidden{display:none!important}
        #refreshInterval{width:auto;padding:4px 8px;font-size:12px;border-color:var(--bdr2)}
        .scan-table-scroll{max-height:460px;overflow-y:auto;border-radius:var(--rs);border:1px solid var(--bdr)}
        .timeline-table-scroll{max-height:600px;overflow-y:auto;overflow-x:auto;border-radius:var(--rs);border:1px solid var(--bdr)}
        .tl-file-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;
            background:#e0f2fe;color:#0369a1;white-space:nowrap}
        .tl-ts{font-size:11px;color:var(--txt2);white-space:nowrap}
        .placeholder-msg{display:flex;flex-direction:column;align-items:center;justify-content:center;
            min-height:460px;color:var(--txt3);gap:10px;font-size:14px}
        .placeholder-msg .big-icon{font-size:48px;margin-bottom:8px}

        /* ── Dark mode ───────────────────────────────────── */
        body.dark{
            --bg:#0f172a;--surf:#1e293b;--surf2:#0f172a;
            --bdr:#334155;--bdr2:#475569;
            --txt:#f1f5f9;--txt2:#94a3b8;--txt3:#64748b;
        }
        body.dark .topnav{background:linear-gradient(135deg,#020617,#0f172a)}
        body.dark .toolbar{background:var(--surf);border-color:var(--bdr)}
        body.dark .panel{background:var(--surf);border-color:var(--bdr)}
        body.dark .plot-area{background:var(--surf);border-color:var(--bdr)}
        body.dark .control-group{background:var(--bg);border-color:var(--bdr)}
        body.dark .control-group:hover{border-color:#4f46e5}
        body.dark select{background:var(--bg);color:var(--txt);border-color:var(--bdr)}
        body.dark input[type="text"],body.dark input[type="number"]{background:var(--bg);color:var(--txt);border-color:var(--bdr)}
        body.dark .file-browser{background:var(--bg);border-color:var(--bdr)}
        body.dark .file-item{border-bottom-color:var(--bdr);color:var(--txt)}
        body.dark .file-item:hover{background:rgba(59,130,246,.1)}
        body.dark .file-item.spec-file{background:rgba(217,119,6,.1)}
        body.dark .file-item.spec-file:hover{background:rgba(217,119,6,.18)}
        body.dark .breadcrumb{background:var(--bg);border-color:var(--bdr);color:var(--txt2)}
        body.dark .scan-table td{border-bottom-color:var(--bdr);color:var(--txt)}
        body.dark .scan-table tr:nth-child(even){background:var(--bg)}
        body.dark .scan-table tr:hover{background:rgba(59,130,246,.1)}
        body.dark .command{background:var(--bg);color:#818cf8}
        body.dark .success{background:#052e16;color:#86efac}
        body.dark .error{background:#2d0a0a;color:#fca5a5}
        body.dark .info{background:#0c1f3d;color:#93c5fd}
        body.dark .export-option{background:var(--bg);border-color:var(--bdr)}
        body.dark .export-option:hover{background:rgba(59,130,246,.08)}
        body.dark .export-option label{color:var(--txt)}
        body.dark .notebook-panel{background:#1c1a05;border-color:#78350f}
        body.dark #noteInput{background:var(--bg);color:var(--txt);border-color:#78350f}
        body.dark .notebook-entry{background:var(--bg);border-color:#78350f;color:var(--txt)}
        body.dark .btn-sec{background:var(--bg);color:var(--txt);border-color:var(--bdr)}
        body.dark .btn-sec:hover{background:var(--bdr)}
        body.dark .compare-panel{background:#042f2e;border-color:#134e4a}
        body.dark .fit-panel{background:#0f0f2a;border-color:#312e81}
        body.dark .placeholder-msg{color:var(--txt3)}

        /* ── Help modal tabs ─────────────────────────── */
        .htab{
            background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);
            color:rgba(255,255,255,.75);padding:6px 14px;border-radius:6px 6px 0 0;
            cursor:pointer;font-size:12px;font-weight:600;transition:all .15s;
            border-bottom:none;margin-bottom:-1px;
        }
        .htab:hover{background:rgba(255,255,255,.22);color:#fff}
        .htab-active{background:#fff!important;color:#1e3a8a!important;border-color:#fff!important}
    </style>
</head>
<body>

    <!-- ── Top navigation bar ─────────────────────────── -->
    <nav class="topnav">
        <div class="nav-logo">
            <div class="nav-logo-icon">&#128300;</div>
            <div>
                <div style="font-size:16px;font-weight:700;line-height:1.1;">SPEC Overview</div>
                <div style="font-size:10px;font-weight:400;color:rgba(255,255,255,.55);letter-spacing:.4px;">CHESS &bull; ID4B Beamline</div>
            </div>
        </div>
        <div class="nav-divider"></div>
        <button onclick="showHomePage()" id="homeBtn" class="nav-btn" title="Back to Home / Documentation" style="font-size:13px;">&#127968; Home</button>
        <div class="nav-divider"></div>
        <div class="nav-file">
            <span class="nav-filename" id="navFilename">No file loaded</span>
            <span class="nav-badge" id="navBadge">LOADED</span>
        </div>
        <div class="nav-controls">
            <span id="refreshStatus"></span>
            <select id="refreshInterval" class="nav-select">
                <option value="5">5 s</option>
                <option value="10" selected>10 s</option>
                <option value="30">30 s</option>
                <option value="60">60 s</option>
            </select>
            <button onclick="toggleAutoRefresh()" id="refreshBtn" class="nav-btn">&#9654; Auto-Refresh</button>
            <button onclick="toggleDark()" id="darkBtn" class="nav-btn">&#127769; Dark</button>
        </div>
    </nav>

    <!-- Toast notifications (fixed) -->
    <div id="messages"></div>

    <div class="app-body">

        <!-- ── Main toolbar ───────────────────────────── -->
        <div class="toolbar">
            <div class="toolbar-group">
                <span class="tlabel">Files</span>
                <button onclick="toggleFileBrowser()" class="btn-sec" id="browseBtn">&#128193; Browse</button>
            </div>
            <div class="tsep"></div>
            <div class="toolbar-group">
                <span class="tlabel">View</span>
                <button onclick="showDataInfo()" class="btn-sec">&#8505;&#65039; Data Info</button>
                <button onclick="showPlotControls()" class="btn-sec">&#127912; Plot Controls</button>
                <button onclick="showScanTable()" class="btn-sec">&#128203; Scan Table</button>
                <button onclick="showExportControls()" class="btn-sec">&#128190; Export CSV</button>
            </div>
            <div class="tsep"></div>
            <div class="toolbar-group">
                <span class="tlabel">Tools</span>
                <button onclick="toggleCompareMode()" id="compareBtn" class="btn-teal">&#9878;&#65038; Compare</button>
                <button onclick="showNotebook()" class="btn-amber">&#128211; Notebook</button>
                <button onclick="showMotorPositions()" class="btn-sec" id="motorBtn">&#9881;&#65038; Motors</button>
                <button onclick="showFolderTimeline()" class="btn-sec" id="timelineBtn">&#128197; Timeline</button>
            </div>
            <div style="margin-left:auto;">
                <button onclick="showHelp()" class="btn-ghost" title="Help &amp; Documentation" style="font-size:15px;padding:4px 10px;">&#10067; Help</button>
            </div>
        </div>

        <!-- ════════════════════════════════════════════════════
             HOMEPAGE  (hidden once a file is loaded)
             ════════════════════════════════════════════════════ -->
        <div id="homePage">

            <!-- Hero banner -->
            <div style="background:linear-gradient(135deg,#0c1445 0%,#1e3a8a 55%,#0d9488 100%);
                 border-radius:var(--r) var(--r) 0 0;padding:36px 36px 0 36px;">

                <!-- Title row -->
                <div style="display:flex;align-items:flex-start;gap:20px;flex-wrap:wrap;margin-bottom:24px;">
                    <div style="width:56px;height:56px;background:rgba(255,255,255,.15);border-radius:14px;
                         display:flex;align-items:center;justify-content:center;font-size:30px;flex-shrink:0;">&#128300;</div>
                    <div style="flex:1;min-width:220px;">
                        <div style="color:rgba(255,255,255,.55);font-size:11px;font-weight:700;
                             text-transform:uppercase;letter-spacing:.9px;margin-bottom:6px;">
                            Cornell High Energy Synchrotron Source &bull; ID4B Beamline
                        </div>
                        <div style="color:#fff;font-size:26px;font-weight:700;line-height:1.2;margin-bottom:10px;">
                            SPEC Data Analysis Insights @ CHESS
                        </div>
                        <div style="color:rgba(255,255,255,.75);font-size:13px;line-height:1.8;max-width:600px;">
                            A browser-based tool for loading, visualising, and exporting SPEC data files collected at CHESS ID4B.
                            Browse the beamline NFS filesystem, plot any scan interactively, compare two experiments,
                            fit peaks, and monitor live data during an active run &mdash; no local software required.
                        </div>
                        <div style="margin-top:18px;display:flex;gap:10px;flex-wrap:wrap;">
                            <button onclick="startBrowsing()"
                                style="background:#3b82f6;border:none;color:#fff;padding:9px 20px;
                                       border-radius:7px;cursor:pointer;font-size:13px;font-weight:700;">
                                &#128193; Browse Files to Start
                            </button>
                        </div>
                    </div>
                    <!-- Stats sidebar -->
                    <div style="display:flex;flex-direction:column;gap:8px;min-width:170px;flex-shrink:0;">
                        <div style="background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.15);
                             border-radius:8px;padding:12px 15px;color:#fff;">
                            <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;
                                 color:rgba(255,255,255,.5);margin-bottom:4px;">Default data path</div>
                            <div style="font-family:monospace;font-size:12px;">/nfs/chess/id4b/</div>
                        </div>
                        <div style="background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.15);
                             border-radius:8px;padding:12px 15px;color:rgba(255,255,255,.85);">
                            <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;
                                 color:rgba(255,255,255,.5);margin-bottom:6px;">Capabilities</div>
                            <div style="font-size:12px;line-height:1.9;">
                                &#128202; Interactive plots<br>
                                &#9878;&#65038; Two-file comparison<br>
                                &#128208; Gaussian / Lorentzian fit<br>
                                &#9654; Live auto-refresh<br>
                                &#128190; CSV &amp; report export
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Tab strip -->
                <div style="display:flex;gap:2px;">
                    <button onclick="switchHomePage('about')"      id="hptab-about"      class="htab htab-active">&#127968; About</button>
                    <button onclick="switchHomePage('quickstart')" id="hptab-quickstart" class="htab">&#9889; Quick Start</button>
                    <button onclick="switchHomePage('features')"   id="hptab-features"   class="htab">&#128295; Features</button>
                    <button onclick="switchHomePage('format')"     id="hptab-format"     class="htab">&#128196; SPEC Format</button>
                    <button onclick="switchHomePage('shortcuts')"  id="hptab-shortcuts"  class="htab">&#9000; Shortcuts</button>
                </div>
            </div><!-- /hero -->

            <!-- Tab content area -->
            <div class="panel" style="border-radius:0 0 var(--r) var(--r);border-top:none;margin-bottom:14px;">

                <!-- ══ ABOUT ══ -->
                <div id="hppane-about">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;">
                        <div style="background:linear-gradient(135deg,#eff6ff,#f0fdfa);border:1px solid #bfdbfe;
                             border-radius:var(--rs);padding:18px;">
                            <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;
                                 color:#1e40af;margin-bottom:8px;">&#127979; About CHESS &amp; ID4B</div>
                            <div style="font-size:13px;color:var(--txt);line-height:1.8;">
                                <strong>CHESS</strong> (Cornell High Energy Synchrotron Source) is a national user facility at
                                Cornell University, Ithaca, NY, providing high-intensity hard X-ray beams for research in physics,
                                chemistry, biology, and materials science.
                                The <strong>ID4B beamline</strong> supports powder diffraction, total scattering, and pair
                                distribution function (PDF) experiments. Data acquisition is controlled by
                                <strong>SPEC</strong>, which writes scan data as plain-text ASCII files to the NFS server at
                                <code style="background:rgba(30,64,175,.1);padding:1px 6px;border-radius:3px;font-family:monospace;">/nfs/chess/id4b/</code>.
                            </div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:18px;">
                            <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;
                                 color:var(--txt3);margin-bottom:8px;">&#128202; About This Overview</div>
                            <div style="font-size:13px;color:var(--txt);line-height:1.8;">
                                The <strong>SPEC Overview</strong> is a lightweight, browser-based analysis tool built for
                                researchers at CHESS ID4B. It reads SPEC files directly from the NFS filesystem, parses every
                                scan and its metadata, and creates interactive Plotly charts in seconds.
                                It works both during an active experiment (live auto-refresh) and for post-experiment analysis.
                                No Python, MATLAB, or local software installation is needed.
                            </div>
                        </div>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px;">
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:14px;text-align:center;">
                            <div style="font-size:26px;margin-bottom:6px;">&#128202;</div>
                            <div style="font-weight:700;font-size:13px;margin-bottom:3px;">Interactive Plots</div>
                            <div style="font-size:12px;color:var(--txt2);">Zoom, pan, tooltips and drawing tools via Plotly</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:14px;text-align:center;">
                            <div style="font-size:26px;margin-bottom:6px;">&#128203;</div>
                            <div style="font-weight:700;font-size:13px;margin-bottom:3px;">Scan Metadata</div>
                            <div style="font-size:12px;color:var(--txt2);">Command, temperature, timestamp and comments per scan</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:14px;text-align:center;">
                            <div style="font-size:26px;margin-bottom:6px;">&#9878;&#65038;</div>
                            <div style="font-weight:700;font-size:13px;margin-bottom:3px;">File Comparison</div>
                            <div style="font-size:12px;color:var(--txt2);">Overlay scans from two SPEC files on one plot</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:14px;text-align:center;">
                            <div style="font-size:26px;margin-bottom:6px;">&#128208;</div>
                            <div style="font-weight:700;font-size:13px;margin-bottom:3px;">Curve Fitting</div>
                            <div style="font-size:12px;color:var(--txt2);">Gaussian &amp; Lorentzian fits with FWHM and peak position</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:14px;text-align:center;">
                            <div style="font-size:26px;margin-bottom:6px;">&#9654;</div>
                            <div style="font-weight:700;font-size:13px;margin-bottom:3px;">Live Auto-Refresh</div>
                            <div style="font-size:12px;color:var(--txt2);">Monitors active scans and auto-plots each new scan</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:14px;text-align:center;">
                            <div style="font-size:26px;margin-bottom:6px;">&#128190;</div>
                            <div style="font-weight:700;font-size:13px;margin-bottom:3px;">CSV Export</div>
                            <div style="font-size:12px;color:var(--txt2);">Download all data or a custom selection of scans &amp; columns</div>
                        </div>
                    </div>
                    <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);
                         padding:12px 16px;font-size:12px;color:var(--txt2);">
                        <strong style="color:var(--txt);">&#128295; Built with:</strong>
                        &nbsp;Python 3 &bull; FastAPI &bull; Pandas &bull; SciPy (curve fitting) &bull; Plotly.js &bull; Inter (Google Fonts)
                    </div>
                </div>

                <!-- ══ QUICK START ══ -->
                <div id="hppane-quickstart" class="hidden">
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;">
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:16px;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                                <div style="width:26px;height:26px;background:#1e3a8a;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;">1</div>
                                <div style="font-weight:700;font-size:13px;">Load a SPEC file</div>
                            </div>
                            <div style="font-size:12px;color:var(--txt2);line-height:1.7;">Click <strong>Browse Files to Start</strong> above or <strong>Browse</strong> in the toolbar. The browser opens at <code style="background:var(--bg);padding:1px 4px;border-radius:3px;font-family:monospace;font-size:11px;">/nfs/chess/id4b/</code>. Navigate to your experiment folder and click any file highlighted in <span style="background:#fef9c3;padding:0 4px;border-radius:2px;">yellow</span> — those are SPEC files. The filename appears in the top bar when loaded.</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:16px;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                                <div style="width:26px;height:26px;background:#1e3a8a;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;">2</div>
                                <div style="font-weight:700;font-size:13px;">Create a plot</div>
                            </div>
                            <div style="font-size:12px;color:var(--txt2);line-height:1.7;">Click <strong>Plot Controls</strong>. The last scan is pre-selected and the X-axis is auto-set from the scan command motor (e.g. <em>ascan mond &hellip;</em> &rarr; X&nbsp;=&nbsp;<em>mond</em>). Select Y columns (hold <kbd style="background:#1e3a8a;color:#fff;padding:1px 4px;border-radius:3px;font-size:10px;">Ctrl</kbd> for multiple), then click <strong>Create Plot</strong>.</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:16px;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                                <div style="width:26px;height:26px;background:#1e3a8a;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;">3</div>
                                <div style="font-weight:700;font-size:13px;">Review scan metadata</div>
                            </div>
                            <div style="font-size:12px;color:var(--txt2);line-height:1.7;">Click <strong>Scan Table</strong> to see every scan's command, timestamp, temperature, count time and comments. Download the full table as CSV for your records.</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:16px;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                                <div style="width:26px;height:26px;background:#0d9488;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;">4</div>
                                <div style="font-weight:700;font-size:13px;">Live monitoring</div>
                            </div>
                            <div style="font-size:12px;color:var(--txt2);line-height:1.7;">Enable <strong>Auto-Refresh</strong> in the top bar. Choose a polling interval (5&ndash;60&nbsp;s). When a new scan completes the plot updates automatically showing the latest scan only — ideal for active beamtime.</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:16px;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                                <div style="width:26px;height:26px;background:#0d9488;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;">5</div>
                                <div style="font-weight:700;font-size:13px;">Compare two files</div>
                            </div>
                            <div style="font-size:12px;color:var(--txt2);line-height:1.7;">Click <strong>Compare</strong> and browse to a second SPEC file. Select scans and Y columns from File 2 and click <strong>Overlay Comparison Plot</strong>. File 2 traces appear as dashed lines with <em>[F2]</em> labels.</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:16px;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                                <div style="width:26px;height:26px;background:#d97706;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;">6</div>
                                <div style="font-weight:700;font-size:13px;">Export &amp; annotate</div>
                            </div>
                            <div style="font-size:12px;color:var(--txt2);line-height:1.7;">Use <strong>Export CSV</strong> to download data for offline analysis. Use <strong>Notebook</strong> to add timestamped notes, then <em>Download Plot + Notes</em> to save an HTML report combining the plot image with all your observations.</div>
                        </div>
                    </div>
                </div>

                <!-- ══ FEATURES ══ -->
                <div id="hppane-features" class="hidden">
                    <table style="width:100%;border-collapse:collapse;font-size:13px;">
                        <thead>
                            <tr style="background:#1e3a8a;color:#fff;">
                                <th style="padding:10px 12px;text-align:left;font-weight:600;width:18%;">Feature</th>
                                <th style="padding:10px 12px;text-align:left;font-weight:600;width:24%;">How to access</th>
                                <th style="padding:10px 12px;text-align:left;font-weight:600;">Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#128193; Browse Files</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Toolbar &rarr; Files &rarr; Browse<br><small style="color:var(--txt3);">Ctrl+O</small></td>
                                <td style="padding:10px 12px;color:var(--txt2);">Browse the CHESS NFS filesystem. SPEC files auto-detected and highlighted in yellow with a SPEC badge. Enter any absolute path and press Go to navigate directly.</td>
                            </tr>
                            <tr style="background:var(--surf2);border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#8505;&#65039; Data Info</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Toolbar &rarr; View &rarr; Data Info<br><small style="color:var(--txt3);">Ctrl+I</small></td>
                                <td style="padding:10px 12px;color:var(--txt2);">Centred panel showing file name, total scan count, total data points, all available column names, and the full list of scan numbers.</td>
                            </tr>
                            <tr style="border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#127912; Plot Controls</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Toolbar &rarr; View &rarr; Plot Controls<br><small style="color:var(--txt3);">Ctrl+P</small></td>
                                <td style="padding:10px 12px;color:var(--txt2);">Choose X-axis (auto-set from scan motor), Y columns (multi-select with Ctrl), scans to overlay, plot type (Line / Scatter / Bar), Normalise, Log-scale Y, and optional curve fitting.</td>
                            </tr>
                            <tr style="background:var(--surf2);border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#128203; Scan Table</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Toolbar &rarr; View &rarr; Scan Table<br><small style="color:var(--txt3);">Ctrl+T</small></td>
                                <td style="padding:10px 12px;color:var(--txt2);">Scrollable table of every scan: number, command, timestamp, temperature, count time, data points, and inline comments. Downloadable as CSV.</td>
                            </tr>
                            <tr style="border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#128190; Export CSV</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Toolbar &rarr; View &rarr; Export CSV<br><small style="color:var(--txt3);">Ctrl+E</small></td>
                                <td style="padding:10px 12px;color:var(--txt2);">Three modes: <strong>All Data</strong> (entire file), <strong>Last Plotted</strong> (current plot only), or <strong>Custom</strong> (choose specific scans and columns).</td>
                            </tr>
                            <tr style="background:var(--surf2);border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#9878;&#65038; Compare Files</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Toolbar &rarr; Tools &rarr; Compare</td>
                                <td style="padding:10px 12px;color:var(--txt2);">Load a second SPEC file independently (no effect on main browser). Overlay File 2 scans as dashed lines with [F2] labels on the same plot as File 1.</td>
                            </tr>
                            <tr style="border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#128208; Curve Fitting</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Plot Controls &rarr; Curve Fitting</td>
                                <td style="padding:10px 12px;color:var(--txt2);">Gaussian or Lorentzian fit on any Y column / scan. Results: peak position, FWHM, amplitude, mean, max, min, std dev, &Delta;(max&minus;min). Fit curve overlaid as a dashed red line.</td>
                            </tr>
                            <tr style="background:var(--surf2);border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#128211; Lab Notebook</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Toolbar &rarr; Tools &rarr; Notebook</td>
                                <td style="padding:10px 12px;color:var(--txt2);">Add free-text notes with automatic timestamps. Download as plain text or export an HTML report combining the plot image with all notes.</td>
                            </tr>
                            <tr style="border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#9654; Auto-Refresh</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Top navigation bar</td>
                                <td style="padding:10px 12px;color:var(--txt2);">Watches the file for changes (size + mtime). Intervals: 5, 10, 30, 60 s. On new scan detection, reloads the file and re-plots the latest scan. Ideal for live monitoring at the beamline.</td>
                            </tr>
                            <tr style="background:var(--surf2);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#127769; Dark Mode</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Top navigation bar &rarr; Dark</td>
                                <td style="padding:10px 12px;color:var(--txt2);">Full dark colour scheme across all panels and plot area. Useful in low-light hutch environments.</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- ══ SPEC FORMAT ══ -->
                <div id="hppane-format" class="hidden">
                    <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:16px;margin-bottom:14px;font-size:13px;color:var(--txt);line-height:1.8;">
                        SPEC data files are plain-text ASCII files written by the SPEC diffractometer control software.
                        They typically have <strong>no file extension</strong> and names like
                        <code style="background:var(--bg);padding:1px 5px;border-radius:3px;font-family:monospace;">align_week1</code> or
                        <code style="background:var(--bg);padding:1px 5px;border-radius:3px;font-family:monospace;">aTaCo2</code>.
                        The dashboard auto-detects them by checking the first 10 lines for at least two SPEC header markers.
                    </div>
                    <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--txt3);margin-bottom:10px;">Header markers</div>
                    <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:16px;">
                        <thead>
                            <tr style="background:#1e3a8a;color:#fff;">
                                <th style="padding:9px 12px;text-align:left;font-weight:600;width:10%;">Marker</th>
                                <th style="padding:9px 12px;text-align:left;font-weight:600;width:36%;">Meaning</th>
                                <th style="padding:9px 12px;text-align:left;font-weight:600;">Example</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom:1px solid var(--bdr);"><td style="padding:8px 12px;font-family:monospace;font-weight:700;color:#1e40af;">#F</td><td style="padding:8px 12px;color:var(--txt2);">File name header</td><td style="padding:8px 12px;font-family:monospace;font-size:12px;color:var(--txt3);">#F align_week1</td></tr>
                            <tr style="background:var(--surf2);border-bottom:1px solid var(--bdr);"><td style="padding:8px 12px;font-family:monospace;font-weight:700;color:#1e40af;">#E</td><td style="padding:8px 12px;color:var(--txt2);">Epoch (Unix timestamp of file creation)</td><td style="padding:8px 12px;font-family:monospace;font-size:12px;color:var(--txt3);">#E 1704067200</td></tr>
                            <tr style="border-bottom:1px solid var(--bdr);"><td style="padding:8px 12px;font-family:monospace;font-weight:700;color:#1e40af;">#D</td><td style="padding:8px 12px;color:var(--txt2);">Date/time string</td><td style="padding:8px 12px;font-family:monospace;font-size:12px;color:var(--txt3);">#D Mon Jan  1 00:00:00 2024</td></tr>
                            <tr style="background:var(--surf2);border-bottom:1px solid var(--bdr);"><td style="padding:8px 12px;font-family:monospace;font-weight:700;color:#1e40af;">#S</td><td style="padding:8px 12px;color:var(--txt2);">Scan start — number + full command</td><td style="padding:8px 12px;font-family:monospace;font-size:12px;color:var(--txt3);">#S 14 ascan mond 6.8 6.9 160 0.1</td></tr>
                            <tr style="border-bottom:1px solid var(--bdr);"><td style="padding:8px 12px;font-family:monospace;font-weight:700;color:#1e40af;">#L</td><td style="padding:8px 12px;color:var(--txt2);">Column labels (space-separated)</td><td style="padding:8px 12px;font-family:monospace;font-size:12px;color:var(--txt3);">#L mond  Epoch  I0  Det  Monitor</td></tr>
                            <tr style="background:var(--surf2);"><td style="padding:8px 12px;font-family:monospace;font-weight:700;color:#1e40af;">#C</td><td style="padding:8px 12px;color:var(--txt2);">Comment (sample info, conditions)</td><td style="padding:8px 12px;font-family:monospace;font-size:12px;color:var(--txt3);">#C Sample: TaCoO at 300K</td></tr>
                        </tbody>
                    </table>
                    <div style="background:#fefce8;border:1px solid #fde68a;border-radius:var(--rs);padding:14px;font-size:13px;color:#78350f;line-height:1.7;">
                        <strong>&#128161; X-axis auto-selection:</strong> When you select a scan, the dashboard reads its
                        <code style="font-family:monospace;">#S</code> command and extracts the <em>second token</em> as the scanning motor.
                        For example <code style="font-family:monospace;">ascan <strong>mond</strong> 6.8 6.9 160 0.1</code> sets X&nbsp;=&nbsp;<strong>mond</strong>.
                        This works for <code style="font-family:monospace;">ascan</code>, <code style="font-family:monospace;">dscan</code>, and <code style="font-family:monospace;">flyscan</code>.
                    </div>
                </div>

                <!-- ══ SHORTCUTS ══ -->
                <div id="hppane-shortcuts" class="hidden">
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;">
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:13px;display:flex;align-items:center;gap:12px;">
                            <kbd style="background:#1e3a8a;color:#fff;padding:4px 10px;border-radius:5px;font-size:12px;font-family:monospace;white-space:nowrap;flex-shrink:0;">Ctrl+O</kbd>
                            <div><div style="font-weight:600;font-size:13px;">Browse Files</div><div style="font-size:11px;color:var(--txt3);">Open / close the file browser</div></div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:13px;display:flex;align-items:center;gap:12px;">
                            <kbd style="background:#1e3a8a;color:#fff;padding:4px 10px;border-radius:5px;font-size:12px;font-family:monospace;white-space:nowrap;flex-shrink:0;">Ctrl+I</kbd>
                            <div><div style="font-weight:600;font-size:13px;">Data Info</div><div style="font-size:11px;color:var(--txt3);">Show file &amp; column summary</div></div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:13px;display:flex;align-items:center;gap:12px;">
                            <kbd style="background:#1e3a8a;color:#fff;padding:4px 10px;border-radius:5px;font-size:12px;font-family:monospace;white-space:nowrap;flex-shrink:0;">Ctrl+P</kbd>
                            <div><div style="font-weight:600;font-size:13px;">Plot Controls</div><div style="font-size:11px;color:var(--txt3);">Open / close plot settings</div></div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:13px;display:flex;align-items:center;gap:12px;">
                            <kbd style="background:#1e3a8a;color:#fff;padding:4px 10px;border-radius:5px;font-size:12px;font-family:monospace;white-space:nowrap;flex-shrink:0;">Ctrl+T</kbd>
                            <div><div style="font-weight:600;font-size:13px;">Scan Table</div><div style="font-size:11px;color:var(--txt3);">Open / close scan metadata table</div></div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:13px;display:flex;align-items:center;gap:12px;">
                            <kbd style="background:#1e3a8a;color:#fff;padding:4px 10px;border-radius:5px;font-size:12px;font-family:monospace;white-space:nowrap;flex-shrink:0;">Ctrl+E</kbd>
                            <div><div style="font-weight:600;font-size:13px;">Export CSV</div><div style="font-size:11px;color:var(--txt3);">Open / close export panel</div></div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:13px;display:flex;align-items:center;gap:12px;">
                            <kbd style="background:#475569;color:#fff;padding:4px 10px;border-radius:5px;font-size:12px;font-family:monospace;white-space:nowrap;flex-shrink:0;">Escape</kbd>
                            <div><div style="font-weight:600;font-size:13px;">Close modal</div><div style="font-size:11px;color:var(--txt3);">Close Data Info, Help, or any open modal</div></div>
                        </div>
                    </div>
                </div>

            </div><!-- /tab content panel -->
        </div><!-- /#homePage -->

        <!-- ── File browser (collapsible) ────────────── -->
        <div id="fileBrowser" class="panel hidden">
            <div class="panel-header">
                <div>
                    <div class="panel-title">&#128193; File Browser</div>
                    <div class="panel-sub">Navigate to your SPEC data files</div>
                </div>
                <div style="display:flex;gap:6px;align-items:center;">
                    <button onclick="showFolderTimeline()" class="btn-teal" style="white-space:nowrap;font-size:12px;padding:4px 10px;" title="Show chronological timeline of all scans in current folder">&#128197; Folder Timeline</button>
                    <button onclick="toggleFileBrowser()" class="btn-ghost">&#10005;</button>
                </div>
            </div>
            <div style="display:flex;gap:8px;margin-bottom:8px;align-items:center;">
                <input id="pathInput" type="text" value="/nfs/chess/id4b/"
                       placeholder="Enter full directory path..."
                       style="flex:1;"
                       onkeydown="if(event.key==='Enter') gotoPath()">
                <button onclick="gotoPath()" class="btn-pri" style="white-space:nowrap;">&#128269; Go</button>
            </div>
            <div id="breadcrumb" class="breadcrumb">/nfs/chess/id4b/</div>
            <div id="fileList" class="file-browser">
                <div style="text-align:center;padding:20px;color:var(--txt3);">Enter a path above and press Go, or click Browse to start</div>
            </div>
        </div>

        <div id="folderTimelinePanel" class="panel hidden">
            <div class="panel-header">
                <div>
                    <div class="panel-title">&#128197; Folder Timeline</div>
                    <div class="panel-sub" id="folderTimelineSub">Chronological scan history across all SPEC files in the folder</div>
                </div>
                <div style="display:flex;gap:6px;align-items:center;">
                    <button onclick="downloadTimelineCSV()" class="btn-teal" style="white-space:nowrap;font-size:12px;">&#8681; CSV</button>
                    <button onclick="document.getElementById('folderTimelinePanel').classList.add('hidden')" class="btn-ghost">&#10005;</button>
                </div>
            </div>
            <!-- Scan data folder result — shown when a Scan # is clicked -->
            <div id="timelineScanDataPath" style="display:none;margin-bottom:10px;padding:10px 14px;
                 border-radius:var(--rs);border:1px solid var(--bdr);background:var(--surf2);
                 font-size:13px;line-height:1.6;"></div>
            <div id="folderTimelineContent">
                <div class="placeholder-msg">
                    <div class="big-icon">&#128197;</div>
                    <div>Open a folder in the File Browser and click <strong>Folder Timeline</strong></div>
                </div>
            </div>
        </div>

        <div id="plotControls" class="panel hidden">
            <div class="panel-header">
                <div>
                    <div class="panel-title">&#127912; Plot Controls</div>
                    <div class="panel-sub">Select axes, scans, and plot options</div>
                </div>
                <button onclick="document.getElementById('plotControls').classList.add('hidden')" class="btn-ghost">&#10005;</button>
            </div>

            <div class="control-grid">
                <div class="control-group">
                    <label>&#128202; X-Axis Column:</label>
                    <select id="xAxis"><option value="">Select X column...</option></select>
                </div>

                <div class="control-group">
                    <label>&#128200; Plot Type:</label>
                    <select id="plotType">
                        <option value="line">&#128200; Line Plot</option>
                        <option value="scatter">&#9899; Scatter Plot</option>
                        <option value="bar">&#128202; Bar Chart</option>
                    </select>
                </div>

                <div class="control-group">
                    <label>&#128203; Y-Axis Columns (hold Ctrl for multiple):</label>
                    <select id="yAxis" multiple class="multi-select"></select>
                </div>

                <div class="control-group">
                    <label>&#128290; Scans to Plot (hold Ctrl for multiple):</label>
                    <select id="scans" multiple class="multi-select" onchange="onScanSelectionChange()"></select>
                </div>

                <div class="control-group">
                    <label>&#9881;&#65039; Plot Options:</label>
                    <div class="checkbox-group">
                        <div><input type="checkbox" id="normalize"><label for="normalize">Normalize Data</label></div>
                        <div><input type="checkbox" id="logScale"><label for="logScale">Log Scale Y</label></div>
                    </div>
                </div>

                <div class="control-group">
                    <label>&#128208; Curve Fitting:</label>
                    <select id="fitType" onchange="toggleFitOptions()">
                        <option value="none">None</option>
                        <option value="gaussian">Gaussian</option>
                        <option value="lorentzian">Lorentzian</option>
                    </select>
                    <div id="fitOptions" style="display:none;margin-top:10px;">
                        <label style="font-size:12px;color:#6c757d;display:block;margin-bottom:4px;">Fit Y column:</label>
                        <select id="fitYColumn"></select>
                        <label style="font-size:12px;color:#6c757d;display:block;margin-top:8px;margin-bottom:4px;">Fit Scan:</label>
                        <select id="fitScan"></select>
                    </div>
                </div>

                <button onclick="createCustomPlot()" class="btn-action big-button">
                    &#128640; CREATE PLOT
                </button>
            </div>
        </div>

        <div id="scanTableContainer" class="panel hidden">
            <div class="panel-header">
                <div>
                    <div class="panel-title">&#128203; Scan Information Table</div>
                    <div class="panel-sub">Detailed information about all scans in the loaded file</div>
                </div>
                <div style="display:flex;gap:8px;align-items:center;">
                    <button onclick="downloadScanTableCSV()" class="btn-teal" style="white-space:nowrap;">
                        &#8681; Download CSV
                    </button>
                    <button onclick="document.getElementById('scanTableContainer').classList.add('hidden')" class="btn-ghost">&#10005;</button>
                </div>
            </div>
            <!-- Data path result — shown when a Scan # is clicked -->
            <div id="scanDataPath" style="display:none;margin-bottom:10px;padding:10px 14px;
                 border-radius:var(--rs);border:1px solid var(--bdr);background:var(--surf2);
                 font-size:13px;line-height:1.6;">
            </div>
            <div id="scanTableContent">
                <div class="placeholder-msg">
                    <div class="big-icon">&#128269;</div>
                    <div>Load a SPEC file to view scan information</div>
                </div>
            </div>
        </div>

        <div id="exportControls" class="panel hidden">
            <div class="panel-header">
                <div>
                    <div class="panel-title">&#128190; Export Data</div>
                    <div class="panel-sub">Download scan data as CSV files</div>
                </div>
                <button onclick="document.getElementById('exportControls').classList.add('hidden')" class="btn-ghost">&#10005;</button>
            </div>

            <div class="export-option">
                <input type="radio" id="exportAll" name="exportType" value="all" checked>
                <label for="exportAll">&#128202; Export All Data</label>
                <div class="description">Export complete dataset with all scans and columns</div>
            </div>

            <div class="export-option">
                <input type="radio" id="exportPlotted" name="exportType" value="plotted">
                <label for="exportPlotted">&#128200; Export Last Plotted Data</label>
                <div class="description">Export only the data from your last plot</div>
                <div id="plottedDataInfo" class="description" style="margin-top:10px;color:#007bff;"></div>
            </div>

            <div class="export-option">
                <input type="radio" id="exportSelected" name="exportType" value="selected">
                <label for="exportSelected">&#127919; Export Selected Scans/Columns</label>
                <div class="description">Choose specific scans and columns to export</div>
                <div id="customExportOptions" style="margin-top:15px;display:none;">
                    <div class="control-grid">
                        <div class="control-group">
                            <label>Select Scans:</label>
                            <select id="exportScans" multiple class="multi-select"></select>
                        </div>
                        <div class="control-group">
                            <label>Select Columns:</label>
                            <select id="exportColumns" multiple class="multi-select"></select>
                        </div>
                    </div>
                </div>
            </div>

            <div style="margin-top:20px;">
                <button onclick="exportCSV()" class="btn-action">
                    &#128190; DOWNLOAD CSV FILE
                </button>
            </div>
        </div>

        <!-- ── Plot area ───────────────────────────────── -->
        <div class="plot-area">
            <div id="plotPlaceholder" class="placeholder-msg">
                <div class="big-icon">&#127919;</div>
                <div style="font-size:16px;font-weight:600;color:var(--txt2);">Ready for Analysis</div>
                <div style="font-size:13px;text-align:center;max-width:320px;">
                    Load a SPEC file &rarr; open Plot Controls &rarr; select axes &rarr; Create Plot
                </div>
            </div>
            <div id="plot" style="display:none;min-height:520px;"></div>
        </div>

        <!-- ── Data Info modal (centered overlay) ────── -->
        <div id="dataInfoModal" class="hidden" style="
            position:fixed;inset:0;z-index:400;
            display:flex;align-items:center;justify-content:center;
            background:rgba(15,23,42,.45);backdrop-filter:blur(3px);
        " onclick="if(event.target===this)closeDataInfo()">
            <div class="panel" style="width:min(540px,92vw);max-height:80vh;overflow-y:auto;margin:0;animation:fadeSlide .2s ease;">
                <div class="panel-header">
                    <div>
                        <div class="panel-title">&#8505;&#65039; Data Information</div>
                        <div class="panel-sub" id="dataInfoSubtitle"></div>
                    </div>
                    <button onclick="closeDataInfo()" class="btn-ghost">&#10005;</button>
                </div>
                <div id="dataInfoContent" style="display:grid;gap:10px;"></div>
            </div>
        </div>

        <!-- ── Help / Documentation modal ───────────────── -->
        <div id="helpModal" class="hidden" style="
            position:fixed;inset:0;z-index:400;
            display:flex;align-items:center;justify-content:center;
            background:rgba(15,23,42,.5);backdrop-filter:blur(4px);
        " onclick="if(event.target===this)closeHelp()">
            <div class="panel" style="width:min(780px,96vw);max-height:90vh;overflow-y:auto;margin:0;animation:fadeSlide .2s ease;padding:0;">
                <!-- Banner -->
                <div style="background:linear-gradient(135deg,#0c1445 0%,#1e3a8a 60%,#0d9488 100%);
                     padding:24px 28px 0 28px;border-radius:var(--r) var(--r) 0 0;position:relative;">
                    <button onclick="closeHelp()" style="position:absolute;top:14px;right:14px;
                        background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);
                        color:#fff;border-radius:6px;padding:4px 9px;cursor:pointer;font-size:13px;">&#10005;</button>
                    <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
                        <div style="width:48px;height:48px;background:rgba(255,255,255,.15);border-radius:12px;
                             display:flex;align-items:center;justify-content:center;font-size:26px;flex-shrink:0;">&#128300;</div>
                        <div>
                            <div style="color:#fff;font-size:20px;font-weight:700;line-height:1.2;">CHESS SPEC Overview</div>
                            <div style="color:rgba(255,255,255,.65);font-size:12px;margin-top:3px;">
                                Cornell High Energy Synchrotron Source &bull; Qunatum Materials Beamline (QM2) &bull; v1.0
                            </div>
                        </div>
                    </div>
                    <!-- Tab strip -->
                    <div style="display:flex;gap:2px;" id="helpTabs">
                        <button onclick="switchHelpTab('about')"     id="htab-about"     class="htab htab-active">&#127968; About</button>
                        <button onclick="switchHelpTab('quickstart')" id="htab-quickstart" class="htab">&#9889; Quick Start</button>
                        <button onclick="switchHelpTab('features')"  id="htab-features"  class="htab">&#128295; Features</button>
                        <button onclick="switchHelpTab('format')"    id="htab-format"    class="htab">&#128196; SPEC Format</button>
                        <button onclick="switchHelpTab('shortcuts')" id="htab-shortcuts" class="htab">&#9000; Shortcuts</button>
                    </div>
                </div>

                <div style="padding:24px 28px;">

                <!-- ══ ABOUT TAB ══════════════════════════════════════════ -->
                <div id="hpane-about">
                    <div style="background:linear-gradient(135deg,#eff6ff,#f0fdfa);border:1px solid #bfdbfe;
                         border-radius:var(--rs);padding:18px;margin-bottom:14px;">
                        <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;
                             color:#1e40af;margin-bottom:8px;">&#127979; Cornell High Energy Synchrotron Source (CHESS)</div>
                        <div style="font-size:13px;color:var(--txt);line-height:1.8;">
                            <strong>CHESS</strong> is a national user facility at Cornell University (Ithaca, NY) that provides
                            high-intensity hard X-ray beams for research in physics, chemistry, biology, and materials science.
                            The <strong>ID4B beamline</strong> supports powder diffraction, total scattering, and pair distribution
                            function (PDF) experiments. Data acquisition is controlled by <strong>SPEC</strong>, which writes
                            scan data as ASCII files to the beamline NFS server at
                            <code style="background:rgba(30,64,175,.1);padding:1px 6px;border-radius:3px;font-family:monospace;">/nfs/chess/id4b/</code>.
                        </div>
                    </div>
                    <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:18px;margin-bottom:14px;">
                        <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;
                             color:var(--txt3);margin-bottom:8px;">&#128202; About This Overview</div>
                        <div style="font-size:13px;color:var(--txt);line-height:1.8;">
                            The <strong>SPEC Overview</strong> is a lightweight, browser-based analysis tool built for
                            researchers at CHESS ID4B. It reads SPEC files directly from the NFS filesystem, parses every
                            scan and its metadata, and lets you create interactive Plotly charts in seconds &mdash; no Python,
                            MATLAB, or local software installation required. It is designed to work both during an active
                            experiment (live auto-refresh) and for post-experiment analysis.
                        </div>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:14px;">
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:14px;text-align:center;">
                            <div style="font-size:26px;margin-bottom:6px;">&#128202;</div>
                            <div style="font-weight:700;font-size:13px;margin-bottom:4px;">Interactive Plots</div>
                            <div style="font-size:12px;color:var(--txt2);">Zoom, pan, tooltips and drawing tools via Plotly</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:14px;text-align:center;">
                            <div style="font-size:26px;margin-bottom:6px;">&#128203;</div>
                            <div style="font-weight:700;font-size:13px;margin-bottom:4px;">Scan Metadata</div>
                            <div style="font-size:12px;color:var(--txt2);">Command, temperature, timestamp, comments per scan</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:14px;text-align:center;">
                            <div style="font-size:26px;margin-bottom:6px;">&#9878;&#65038;</div>
                            <div style="font-weight:700;font-size:13px;margin-bottom:4px;">File Comparison</div>
                            <div style="font-size:12px;color:var(--txt2);">Overlay scans from two SPEC files on one plot</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:14px;text-align:center;">
                            <div style="font-size:26px;margin-bottom:6px;">&#128208;</div>
                            <div style="font-weight:700;font-size:13px;margin-bottom:4px;">Curve Fitting</div>
                            <div style="font-size:12px;color:var(--txt2);">Gaussian &amp; Lorentzian fits with FWHM and peak position</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:14px;text-align:center;">
                            <div style="font-size:26px;margin-bottom:6px;">&#9654;</div>
                            <div style="font-weight:700;font-size:13px;margin-bottom:4px;">Live Auto-Refresh</div>
                            <div style="font-size:12px;color:var(--txt2);">Monitors active scans and auto-plots each new scan</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:14px;text-align:center;">
                            <div style="font-size:26px;margin-bottom:6px;">&#128190;</div>
                            <div style="font-weight:700;font-size:13px;margin-bottom:4px;">CSV Export</div>
                            <div style="font-size:12px;color:var(--txt2);">Download all data or a custom selection of scans &amp; columns</div>
                        </div>
                    </div>
                    <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);
                         padding:12px 16px;font-size:12px;color:var(--txt2);">
                        <strong style="color:var(--txt);">&#128295; Built with:</strong>
                        &nbsp;Python 3 &bull; FastAPI &bull; Pandas &bull; SciPy (curve fitting) &bull; Plotly.js &bull; Inter (Google Fonts)
                    </div>
                </div>

                <!-- ══ QUICK-START TAB ════════════════════════════════════ -->
                <div id="hpane-quickstart" class="hidden">
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;">
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:16px;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                                <div style="width:26px;height:26px;background:#1e3a8a;color:#fff;border-radius:50%;
                                     display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;">1</div>
                                <div style="font-weight:700;font-size:13px;">Load a SPEC file</div>
                            </div>
                            <div style="font-size:12px;color:var(--txt2);line-height:1.7;">Click <strong>Browse</strong> in the toolbar. The file browser starts at <code style="background:var(--bg);padding:1px 4px;border-radius:3px;font-family:monospace;font-size:11px;">/nfs/chess/id4b/</code>. Navigate to your experiment folder and click any file highlighted in <span style="background:#fef9c3;padding:0 4px;border-radius:2px;">yellow</span> &mdash; those are SPEC files. The filename appears in the top bar when loaded.</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:16px;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                                <div style="width:26px;height:26px;background:#1e3a8a;color:#fff;border-radius:50%;
                                     display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;">2</div>
                                <div style="font-weight:700;font-size:13px;">Create a plot</div>
                            </div>
                            <div style="font-size:12px;color:var(--txt2);line-height:1.7;">Click <strong>Plot Controls</strong>. The last scan is pre-selected and the X-axis is auto-set from the scan command motor (e.g. <em>ascan mond &hellip;</em> &rarr; X = <em>mond</em>). Select one or more Y columns (hold <kbd style="background:#1e3a8a;color:#fff;padding:1px 4px;border-radius:3px;font-size:10px;">Ctrl</kbd> for multiple), then click <strong>Create Plot</strong>.</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:16px;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                                <div style="width:26px;height:26px;background:#1e3a8a;color:#fff;border-radius:50%;
                                     display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;">3</div>
                                <div style="font-weight:700;font-size:13px;">Review scan metadata</div>
                            </div>
                            <div style="font-size:12px;color:var(--txt2);line-height:1.7;">Click <strong>Scan Table</strong> to view every scan&rsquo;s command, timestamp, temperature, count time and comments. Download the full table as CSV for your lab records.</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:16px;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                                <div style="width:26px;height:26px;background:#0d9488;color:#fff;border-radius:50%;
                                     display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;">4</div>
                                <div style="font-weight:700;font-size:13px;">Live monitoring</div>
                            </div>
                            <div style="font-size:12px;color:var(--txt2);line-height:1.7;">During an active experiment enable <strong>Auto-Refresh</strong> in the top bar. Choose a polling interval (5&ndash;60 s). When a new scan completes the plot updates automatically showing the latest scan only.</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:16px;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                                <div style="width:26px;height:26px;background:#0d9488;color:#fff;border-radius:50%;
                                     display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;">5</div>
                                <div style="font-weight:700;font-size:13px;">Compare two files</div>
                            </div>
                            <div style="font-size:12px;color:var(--txt2);line-height:1.7;">Click <strong>Compare</strong> and browse to a second SPEC file. Select scans and Y columns from File 2 and click <strong>Overlay Comparison Plot</strong>. File 2 traces appear as dashed lines with <em>[F2]</em> labels.</div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:16px;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                                <div style="width:26px;height:26px;background:#d97706;color:#fff;border-radius:50%;
                                     display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0;">6</div>
                                <div style="font-weight:700;font-size:13px;">Export &amp; annotate</div>
                            </div>
                            <div style="font-size:12px;color:var(--txt2);line-height:1.7;">Use <strong>Export CSV</strong> to download data for offline analysis. Use <strong>Notebook</strong> to add timestamped notes, then <em>Download Plot&nbsp;+&nbsp;Notes</em> to save an HTML report combining the plot image with your observations.</div>
                        </div>
                    </div>
                </div>

                <!-- ══ FEATURES TAB ═══════════════════════════════════════ -->
                <div id="hpane-features" class="hidden">
                    <table style="width:100%;border-collapse:collapse;font-size:13px;">
                        <thead>
                            <tr style="background:#1e3a8a;color:#fff;">
                                <th style="padding:10px 12px;text-align:left;font-weight:600;width:20%;">Feature</th>
                                <th style="padding:10px 12px;text-align:left;font-weight:600;width:26%;">How to access</th>
                                <th style="padding:10px 12px;text-align:left;font-weight:600;">Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#128193; Browse Files</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Toolbar &rarr; Files &rarr; Browse<br><small>Ctrl+O</small></td>
                                <td style="padding:10px 12px;color:var(--txt2);">Browse the CHESS NFS filesystem. SPEC files auto-detected and highlighted in yellow. Enter any absolute path and press Go to jump directly.</td>
                            </tr>
                            <tr style="background:var(--surf2);border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#8505;&#65039; Data Info</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Toolbar &rarr; View &rarr; Data Info<br><small>Ctrl+I</small></td>
                                <td style="padding:10px 12px;color:var(--txt2);">Centred panel showing file name, total scan count, total data points, all available column names, and the full scan number list.</td>
                            </tr>
                            <tr style="border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#127912; Plot Controls</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Toolbar &rarr; View &rarr; Plot Controls<br><small>Ctrl+P</small></td>
                                <td style="padding:10px 12px;color:var(--txt2);">Choose X-axis (auto-set from scan motor), Y columns (multi-select), scans to overlay, plot type (Line/Scatter/Bar), Normalise, Log-scale Y, and optional Gaussian or Lorentzian curve fitting.</td>
                            </tr>
                            <tr style="background:var(--surf2);border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#128203; Scan Table</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Toolbar &rarr; View &rarr; Scan Table<br><small>Ctrl+T</small></td>
                                <td style="padding:10px 12px;color:var(--txt2);">Scrollable table of every scan: number, command, timestamp, temperature, count time, data points, comments. Downloadable as CSV.</td>
                            </tr>
                            <tr style="border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#128190; Export CSV</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Toolbar &rarr; View &rarr; Export CSV<br><small>Ctrl+E</small></td>
                                <td style="padding:10px 12px;color:var(--txt2);">Three modes: <strong>All Data</strong>, <strong>Last Plotted</strong>, or <strong>Custom</strong> (choose scans and columns).</td>
                            </tr>
                            <tr style="background:var(--surf2);border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#9878;&#65038; Compare Files</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Toolbar &rarr; Tools &rarr; Compare</td>
                                <td style="padding:10px 12px;color:var(--txt2);">Load a second SPEC file independently (does not affect the main file browser). Overlay File 2 scans as dashed lines with <em>[F2]</em> labels on the same plot as File 1.</td>
                            </tr>
                            <tr style="border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#128208; Curve Fitting</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Plot Controls &rarr; Curve Fitting</td>
                                <td style="padding:10px 12px;color:var(--txt2);">Gaussian or Lorentzian fit on any Y column / scan. Results: peak position, FWHM, amplitude, mean, max, min, std dev, &Delta;(max&minus;min). Fit curve overlaid as dashed red line.</td>
                            </tr>
                            <tr style="background:var(--surf2);border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#128211; Lab Notebook</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Toolbar &rarr; Tools &rarr; Notebook</td>
                                <td style="padding:10px 12px;color:var(--txt2);">Add free-text notes with automatic timestamps. Download as plain text or export as HTML report combining the plot image with all notes.</td>
                            </tr>
                            <tr style="border-bottom:1px solid var(--bdr);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#9654; Auto-Refresh</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Top navigation bar</td>
                                <td style="padding:10px 12px;color:var(--txt2);">Watches the file for changes (size + mtime). Intervals: 5, 10, 30, 60 s. When a new scan is detected, reloads the file and re-plots the latest scan automatically. Ideal for live monitoring at the beamline.</td>
                            </tr>
                            <tr style="background:var(--surf2);">
                                <td style="padding:10px 12px;font-weight:600;vertical-align:top;">&#127769; Dark Mode</td>
                                <td style="padding:10px 12px;color:var(--txt2);vertical-align:top;">Top navigation bar &rarr; Dark</td>
                                <td style="padding:10px 12px;color:var(--txt2);">Full dark colour scheme. Useful in low-light hutch environments.</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- ══ SPEC FORMAT TAB ════════════════════════════════════ -->
                <div id="hpane-format" class="hidden">
                    <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:16px;margin-bottom:14px;font-size:13px;color:var(--txt);line-height:1.8;">
                        SPEC data files are plain-text ASCII files written by the SPEC control software with <strong>no file extension</strong>
                        (e.g. <code style="background:var(--bg);padding:1px 5px;border-radius:3px;font-family:monospace;">align_week1</code>,
                        <code style="background:var(--bg);padding:1px 5px;border-radius:3px;font-family:monospace;">aTaCo2</code>).
                        The dashboard auto-detects them by checking the first 10 lines for at least two SPEC header markers.
                    </div>
                    <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--txt3);margin-bottom:10px;">Header markers</div>
                    <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:16px;">
                        <thead>
                            <tr style="background:#1e3a8a;color:#fff;">
                                <th style="padding:8px 12px;text-align:left;font-weight:600;width:10%;">Marker</th>
                                <th style="padding:8px 12px;text-align:left;font-weight:600;width:35%;">Meaning</th>
                                <th style="padding:8px 12px;text-align:left;font-weight:600;">Example</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom:1px solid var(--bdr);">
                                <td style="padding:8px 12px;font-family:monospace;font-weight:700;color:#1e40af;">#F</td>
                                <td style="padding:8px 12px;color:var(--txt2);">File name header</td>
                                <td style="padding:8px 12px;font-family:monospace;font-size:12px;color:var(--txt3);">#F align_week1</td>
                            </tr>
                            <tr style="background:var(--surf2);border-bottom:1px solid var(--bdr);">
                                <td style="padding:8px 12px;font-family:monospace;font-weight:700;color:#1e40af;">#E</td>
                                <td style="padding:8px 12px;color:var(--txt2);">Epoch (Unix timestamp of file creation)</td>
                                <td style="padding:8px 12px;font-family:monospace;font-size:12px;color:var(--txt3);">#E 1704067200</td>
                            </tr>
                            <tr style="border-bottom:1px solid var(--bdr);">
                                <td style="padding:8px 12px;font-family:monospace;font-weight:700;color:#1e40af;">#D</td>
                                <td style="padding:8px 12px;color:var(--txt2);">Date/time string</td>
                                <td style="padding:8px 12px;font-family:monospace;font-size:12px;color:var(--txt3);">#D Mon Jan  1 00:00:00 2024</td>
                            </tr>
                            <tr style="background:var(--surf2);border-bottom:1px solid var(--bdr);">
                                <td style="padding:8px 12px;font-family:monospace;font-weight:700;color:#1e40af;">#S</td>
                                <td style="padding:8px 12px;color:var(--txt2);">Scan start — number + full command</td>
                                <td style="padding:8px 12px;font-family:monospace;font-size:12px;color:var(--txt3);">#S 14 ascan mond 6.8 6.9 160 0.1</td>
                            </tr>
                            <tr style="border-bottom:1px solid var(--bdr);">
                                <td style="padding:8px 12px;font-family:monospace;font-weight:700;color:#1e40af;">#L</td>
                                <td style="padding:8px 12px;color:var(--txt2);">Column labels (space-separated)</td>
                                <td style="padding:8px 12px;font-family:monospace;font-size:12px;color:var(--txt3);">#L mond  Epoch  I0  Det  Monitor</td>
                            </tr>
                            <tr style="background:var(--surf2);">
                                <td style="padding:8px 12px;font-family:monospace;font-weight:700;color:#1e40af;">#C</td>
                                <td style="padding:8px 12px;color:var(--txt2);">Comment (sample info, conditions)</td>
                                <td style="padding:8px 12px;font-family:monospace;font-size:12px;color:var(--txt3);">#C Sample: TaCoO at 300K</td>
                            </tr>
                        </tbody>
                    </table>
                    <div style="background:#fefce8;border:1px solid #fde68a;border-radius:var(--rs);padding:14px;font-size:13px;color:#78350f;line-height:1.7;">
                        <strong>&#128161; X-axis auto-selection:</strong> When you select a scan the dashboard reads its
                        <code style="font-family:monospace;">#S</code> command and extracts the <em>second token</em> as the scanning motor.
                        For example, <code style="font-family:monospace;">ascan <strong>mond</strong> 6.8 6.9 160 0.1</code> sets X&nbsp;=&nbsp;<strong>mond</strong>.
                        This works for <code style="font-family:monospace;">ascan</code>, <code style="font-family:monospace;">dscan</code>, and <code style="font-family:monospace;">flyscan</code>.
                    </div>
                </div>

                <!-- ══ SHORTCUTS TAB ══════════════════════════════════════ -->
                <div id="hpane-shortcuts" class="hidden">
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;">
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:13px;display:flex;align-items:center;gap:12px;">
                            <kbd style="background:#1e3a8a;color:#fff;padding:4px 10px;border-radius:5px;font-size:12px;font-family:monospace;white-space:nowrap;flex-shrink:0;">Ctrl+O</kbd>
                            <div><div style="font-weight:600;font-size:13px;">Browse Files</div><div style="font-size:11px;color:var(--txt3);">Open / close the file browser</div></div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:13px;display:flex;align-items:center;gap:12px;">
                            <kbd style="background:#1e3a8a;color:#fff;padding:4px 10px;border-radius:5px;font-size:12px;font-family:monospace;white-space:nowrap;flex-shrink:0;">Ctrl+I</kbd>
                            <div><div style="font-weight:600;font-size:13px;">Data Info</div><div style="font-size:11px;color:var(--txt3);">Show file &amp; column summary</div></div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:13px;display:flex;align-items:center;gap:12px;">
                            <kbd style="background:#1e3a8a;color:#fff;padding:4px 10px;border-radius:5px;font-size:12px;font-family:monospace;white-space:nowrap;flex-shrink:0;">Ctrl+P</kbd>
                            <div><div style="font-weight:600;font-size:13px;">Plot Controls</div><div style="font-size:11px;color:var(--txt3);">Open / close plot settings</div></div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:13px;display:flex;align-items:center;gap:12px;">
                            <kbd style="background:#1e3a8a;color:#fff;padding:4px 10px;border-radius:5px;font-size:12px;font-family:monospace;white-space:nowrap;flex-shrink:0;">Ctrl+T</kbd>
                            <div><div style="font-weight:600;font-size:13px;">Scan Table</div><div style="font-size:11px;color:var(--txt3);">Open / close scan metadata table</div></div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:13px;display:flex;align-items:center;gap:12px;">
                            <kbd style="background:#1e3a8a;color:#fff;padding:4px 10px;border-radius:5px;font-size:12px;font-family:monospace;white-space:nowrap;flex-shrink:0;">Ctrl+E</kbd>
                            <div><div style="font-weight:600;font-size:13px;">Export CSV</div><div style="font-size:11px;color:var(--txt3);">Open / close export panel</div></div>
                        </div>
                        <div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:13px;display:flex;align-items:center;gap:12px;">
                            <kbd style="background:#475569;color:#fff;padding:4px 10px;border-radius:5px;font-size:12px;font-family:monospace;white-space:nowrap;flex-shrink:0;">Escape</kbd>
                            <div><div style="font-weight:600;font-size:13px;">Close modal</div><div style="font-size:11px;color:var(--txt3);">Close Data Info, Help, or any open modal</div></div>
                        </div>
                    </div>
                </div>

                </div><!-- /padding wrapper -->
            </div>
        </div>

        <!-- ── Notebook panel ─────────────────────────── -->
        <div id="notebookPanel" class="panel notebook-panel hidden">
            <div class="panel-header">
                <div class="panel-title">&#128211; Lab Notebook</div>
                <button onclick="document.getElementById('notebookPanel').classList.add('hidden')" class="btn-ghost">&#10005;</button>
            </div>
            <textarea id="noteInput" placeholder="Type your note here... (e.g., sample observations, scan conditions, reminders)"></textarea>
            <div style="display:flex;gap:8px;margin:10px 0;flex-wrap:wrap;">
                <button onclick="addNote()" class="btn-pri">&#10133; Add Note</button>
                <button onclick="downloadNotes()" class="btn-sec">&#8681; Download Notes</button>
                <button onclick="downloadPlotWithNotes()" class="btn-teal">&#128247; Download Plot + Notes</button>
                <button onclick="clearNotes()" class="btn-red">&#128465; Clear All</button>
            </div>
            <div id="notebookEntries"></div>
        </div>

        <!-- ── Compare Files panel ─────────────────────── -->
        <div id="comparePanel" class="panel compare-panel hidden">
            <div class="panel-header">
                <div>
                    <div class="panel-title">&#9878;&#65038; Compare Files</div>
                    <div class="panel-sub">Overlay data from two SPEC files on the same plot</div>
                </div>
                <button onclick="document.getElementById('comparePanel').classList.add('hidden')" class="btn-ghost">&#10005;</button>
            </div>

            <div style="display:flex;gap:24px;margin-bottom:12px;flex-wrap:wrap;">
                <div style="min-width:180px;">
                    <div style="font-size:10px;font-weight:700;color:var(--txt3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">File 1 (current)</div>
                    <span id="compareFile1Name" style="font-family:monospace;font-size:12px;color:var(--txt2);">Not loaded</span>
                </div>
                <div style="min-width:180px;">
                    <div style="font-size:10px;font-weight:700;color:var(--txt3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">File 2 (compare)</div>
                    <span id="compareFile2Name" style="font-family:monospace;font-size:12px;color:var(--txt2);">Not loaded</span>
                </div>
            </div>

            <div id="compareFileBrowser" style="margin-bottom:16px;">
                <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;">
                    <input id="compare2PathInput" type="text" value="/nfs/chess/id4b/"
                           placeholder="Enter path or browse..."
                           style="flex:1;"
                           onkeydown="if(event.key==='Enter') gotoCompare2Path()">
                    <button onclick="gotoCompare2Path()" class="btn-teal" style="white-space:nowrap;">&#128269; Go</button>
                </div>
                <div id="compare2FileList" class="file-browser" style="max-height:220px;">
                    <div style="color:var(--txt3);font-size:13px;text-align:center;padding:20px;">Enter a path above and press Go to browse files</div>
                </div>
            </div>

            <div id="compareControls" class="hidden" style="margin-top:14px;padding-top:14px;border-top:1px solid var(--bdr);">
                <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end;margin-bottom:12px;">
                    <div class="control-group" style="flex:1;min-width:160px;">
                        <label>Scans from File 2:</label>
                        <select id="compareScans" multiple class="multi-select" style="max-height:120px;"></select>
                    </div>
                    <div class="control-group" style="flex:1;min-width:160px;">
                        <label>Y Columns from File 2:</label>
                        <select id="compareYCols" multiple class="multi-select" style="max-height:120px;"></select>
                    </div>
                </div>
                <div style="display:flex;gap:8px;">
                    <button onclick="createComparePlot()" class="btn-teal">&#128202; Overlay Comparison Plot</button>
                    <button onclick="clearComparePlot()" class="btn-red">&#10005; Clear Overlay</button>
                </div>
            </div>
        </div>

        <!-- ── Fit results panel ──────────────────────── -->
        <div id="fitResultsPanel" class="panel fit-panel hidden">
            <div class="panel-header">
                <div class="panel-title">&#128208; Fit Results</div>
                <button onclick="document.getElementById('fitResultsPanel').classList.add('hidden')" class="btn-ghost">&#10005;</button>
            </div>
            <div id="fitResultsContent"></div>
        </div>

        <!-- ── Motor positions panel ─────────────────── -->
        <div id="motorPanel" class="panel motor-panel hidden">
            <div class="panel-header">
                <div>
                    <div class="panel-title">&#9881;&#65038; Motor Positions</div>
                    <div class="panel-sub">Motor positions at the start of each scan (#P lines)</div>
                </div>
                <div style="display:flex;gap:8px;align-items:center;">
                    <button onclick="downloadMotorCSV()" class="btn-teal" style="white-space:nowrap;">&#8681; CSV</button>
                    <button onclick="document.getElementById('motorPanel').classList.add('hidden')" class="btn-ghost">&#10005;</button>
                </div>
            </div>
            <!-- Filter bar -->
            <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap;">
                <input type="text" id="motorFilter" placeholder="Filter motors by name or mnemonic..."
                       oninput="filterMotorTable()" style="max-width:300px;">
                <label style="font-size:12px;color:var(--txt2);display:flex;align-items:center;gap:5px;">
                    <input type="checkbox" id="motorHideZero" onchange="filterMotorTable()">
                    Hide all-zero rows
                </label>
                <span id="motorCountLabel" style="font-size:12px;color:var(--txt3);margin-left:4px;"></span>
            </div>
            <div style="overflow:auto;max-height:500px;border:1px solid var(--bdr);border-radius:var(--rs);">
                <div id="motorTableContent">
                    <div class="placeholder-msg" style="min-height:120px;">
                        <div class="big-icon" style="font-size:32px;">&#9881;&#65038;</div>
                        <div>Load a SPEC file to view motor positions</div>
                    </div>
                </div>
            </div>
        </div>

    </div><!-- /.app-body -->

    <script>
        let currentPath = "";
        let dataLoaded = false;
        let dataLoaded2 = false;
        let exportInfo = null;
        let _comparePlotsActive = false;

        function showMessage(message, type = 'info') {
            const div = document.createElement('div');
            div.className = 'status ' + type;
            div.innerHTML = message;
            document.getElementById('messages').appendChild(div);
            setTimeout(() => div.remove(), 10000);
        }

        // ── Path input / goto ─────────────────────────────────────────────────
        async function gotoPath() {
            const input = document.getElementById('pathInput');
            const path = (input ? input.value : '').trim();
            if (!path) return;
            try {
                const r = await fetch('/set_root', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({path: path})
                });
                if (!r.ok) {
                    const err = await r.json();
                    showMessage(err.detail, 'error');
                    return;
                }
                await browseDirectory('');
            } catch(e) {
                showMessage('Could not navigate: ' + e.message, 'error');
            }
        }

        async function toggleFileBrowser() {
            const browser = document.getElementById('fileBrowser');
            if (browser.classList.contains('hidden')) {
                browser.classList.remove('hidden');
                // Use whatever path is in the input box (defaults to /nfs/chess/id4b/)
                await gotoPath();
            } else {
                browser.classList.add('hidden');
                // If no file has been loaded yet, restore the homepage
                if (!dataLoaded) {
                    var hp = document.getElementById('homePage');
                    if (hp) hp.style.display = '';
                }
            }
        }

        // Called from the "Browse Files to Start" button on the homepage
        function startBrowsing() {
            var hp = document.getElementById('homePage');
            if (hp) hp.style.display = 'none';
            var browser = document.getElementById('fileBrowser');
            if (browser) {
                browser.classList.remove('hidden');
                browser.scrollIntoView({behavior:'smooth', block:'start'});
                gotoPath();
            }
        }

        // Called from the Home nav button — restores the documentation homepage
        function showHomePage() {
            // Close every panel so the home page is clean
            var allPanels = [
                'fileBrowser', 'plotControls', 'scanTableContainer',
                'exportControls', 'folderTimelinePanel',
                'notebookPanel', 'comparePanel', 'fitResultsPanel', 'motorPanel'
            ];
            allPanels.forEach(function(id) {
                var el = document.getElementById(id);
                if (el) el.classList.add('hidden');
            });
            // Also hide modals if open
            ['helpModal','dataInfoModal'].forEach(function(id) {
                var el = document.getElementById(id);
                if (el) el.classList.add('hidden');
            });
            // Clear the plot so it doesn't bleed through onto the home page
            var plotDiv = document.getElementById('plot');
            if (plotDiv) {
                try { Plotly.purge('plot'); } catch(e) {}
                plotDiv.style.display = 'none';
            }
            var ph = document.getElementById('plotPlaceholder');
            if (ph) ph.style.display = '';
            // Reset the file browser path back to the NFS root so the next
            // Browse opens at /nfs/chess/id4b/ instead of a deep sub-folder
            var defaultRoot = '/nfs/chess/id4b/';
            var pi = document.getElementById('pathInput');
            if (pi) pi.value = defaultRoot;
            var bc = document.getElementById('breadcrumb');
            if (bc) bc.textContent = defaultRoot;
            var fl = document.getElementById('fileList');
            if (fl) fl.innerHTML =
                '<div style="text-align:center;padding:20px;color:var(--txt3);">Enter a path above and press Go, or click Browse to start</div>';
            var hp = document.getElementById('homePage');
            if (hp) {
                hp.style.display = '';
                hp.scrollIntoView({behavior:'smooth', block:'start'});
            }
        }

        async function browseDirectory(path) {
            try {
                showMessage('Browsing ' + (path || 'root directory') + '...', 'info');

                const response = await fetch('/browse', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: path })
                });

                const result = await response.json();

                if (response.ok) {
                    currentPath = result.current_path;
                    displayFiles(result);
                    showMessage('Found ' + result.total_items + ' items (' + result.spec_files + ' SPEC files)', 'success');
                } else {
                    showMessage('Error: ' + result.detail, 'error');
                }
            } catch (error) {
                showMessage('Browse failed: ' + error.message, 'error');
            }
        }

        function displayFiles(data) {
            const breadcrumb = document.getElementById('breadcrumb');
            const fileList = document.getElementById('fileList');

            const fullPath = data.root_path + (data.current_path ? '/' + data.current_path : '');
            breadcrumb.textContent = fullPath;
            const pathInput = document.getElementById('pathInput');
            if (pathInput) pathInput.value = fullPath;

            fileList.innerHTML = '';

            if (data.items.length === 0) {
                fileList.innerHTML = '<div style="padding:20px;text-align:center;">No items found</div>';
                return;
            }

            data.items.forEach(function(item) {
                const div = document.createElement('div');
                div.className = 'file-item' + (item.is_spec ? ' spec-file' : '');

                let icon = '&#128196;';
                if (item.type === 'directory') {
                    icon = item.is_parent ? '&#128281;' : '&#128193;';
                } else if (item.is_spec) {
                    icon = '&#128202;';
                }

                const sizeInfo = item.size ? formatBytes(item.size) : '';
                const specBadge = item.is_spec ? '<span class="spec-badge">SPEC</span>' : '';

                div.innerHTML = '<span class="icon">' + icon + '</span>' +
                                '<span class="name">' + item.name + '</span>' +
                                '<span class="size">' + sizeInfo + '</span>' +
                                specBadge;

                div.onclick = function() {
                    if (item.type === 'directory') {
                        browseDirectory(item.path);
                    } else {
                        loadFile(item.path);
                    }
                };

                fileList.appendChild(div);
            });
        }

        async function loadFile(filePath) {
            try {
                const fileName = filePath.split('/').pop();
                showMessage('Loading ' + fileName + '...', 'info');

                const response = await fetch('/load_file', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: filePath })
                });

                const result = await response.json();

                if (response.ok) {
                    showMessage(result.message + '<br>' + result.total_scans + ' scans, ' + result.total_points + ' points<br>Columns: ' + result.available_columns.length + '<br>Scan info: ' + (result.scan_info_available ? 'Available' : 'Not available'), 'success');
                    dataLoaded = true;
                    updatePlotControls();
                    updateExportInfo();
                    document.getElementById('fileBrowser').classList.add('hidden');
                    // Hide welcome banner and placeholder once data is loaded
                    var ab = document.getElementById('homePage');
                    if (ab) ab.style.display = 'none';
                    var ph = document.getElementById('plotPlaceholder');
                    if (ph) ph.style.display = 'none';
                    // Update topnav filename display
                    var nf = document.getElementById('navFilename');
                    var nb = document.getElementById('navBadge');
                    if (nf) nf.textContent = fileName;
                    if (nb) nb.className = 'nav-badge loaded';
                    // Update compare panel File 1 label
                    var cf1 = document.getElementById('compareFile1Name');
                    if (cf1) cf1.textContent = fileName;
                    // ── Reset stale-data state so the new file is used everywhere ──
                    _lastScanData = [];
                    // Clear scan-folder path result from the previous file
                    var sdp = document.getElementById('scanDataPath');
                    if (sdp) { sdp.style.display = 'none'; sdp.innerHTML = ''; }
                    // If the scan table panel is already open, reload it immediately
                    var stc = document.getElementById('scanTableContainer');
                    if (stc && !stc.classList.contains('hidden')) {
                        loadScanTable();
                    }
                } else {
                    showMessage('Error loading ' + fileName + ': ' + result.detail, 'error');
                }
            } catch (error) {
                showMessage('Failed to load file: ' + error.message, 'error');
            }
        }

        // ── Compare-file functions ────────────────────────────────────────────
        // The compare browser tracks its OWN absolute path and uses /browse_abs.
        // It NEVER calls /set_root, so the main file browser is unaffected.
        let _compare2AbsPath = '/nfs/chess/id4b/';

        function toggleCompareMode() {
            const panel = document.getElementById('comparePanel');
            if (panel.classList.contains('hidden')) {
                panel.classList.remove('hidden');
                panel.scrollIntoView({behavior:'smooth'});
                const f1name = exportInfo ? (exportInfo.filename || 'File 1') : 'File 1';
                document.getElementById('compareFile1Name').textContent = f1name;
                // DO NOT call gotoCompare2Path here to avoid changing root_path
            } else {
                panel.classList.add('hidden');
            }
        }

        async function gotoCompare2Path() {
            const input = document.getElementById('compare2PathInput');
            const path = (input ? input.value : '').trim() || '/nfs/chess/id4b/';
            await browseCompare2Abs(path);
        }

        async function browseCompare2Abs(absPath) {
            // Navigate the compare file browser using absolute paths (/browse_abs)
            // This never touches data_store["root_path"]
            const listDiv = document.getElementById('compare2FileList');
            listDiv.innerHTML = '<div style="color:#6b7280;text-align:center;padding:16px;">Loading...</div>';
            try {
                const r = await fetch('/browse_abs', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({abs_path: absPath})
                });
                if (!r.ok) {
                    const err = await r.json();
                    listDiv.innerHTML = '<div style="color:red;padding:10px;">' + (err.detail || 'Error') + '</div>';
                    return;
                }
                const result = await r.json();
                _compare2AbsPath = result.abs_path;
                const input = document.getElementById('compare2PathInput');
                if (input) input.value = _compare2AbsPath;
                renderCompare2FileList(result);
            } catch(e) {
                listDiv.innerHTML = '<div style="color:red;padding:10px;">' + e.message + '</div>';
            }
        }

        function renderCompare2FileList(result) {
            const listDiv = document.getElementById('compare2FileList');
            let html = '';
            if (!result.items || result.items.length === 0) {
                listDiv.innerHTML = '<div style="color:#6b7280;text-align:center;padding:16px;">No SPEC files or directories found</div>';
                return;
            }
            result.items.forEach(function(item) {
                // Use data attributes to avoid inline-string escaping issues
                const absPathSafe = encodeURIComponent(item.abs_path);
                const nameSafe    = item.name.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
                if (item.type === 'directory') {
                    html += '<div class="_cmp2dir" data-abs="' + absPathSafe + '" ' +
                        'style="padding:6px 10px;cursor:pointer;border-radius:6px;color:#0d9488;font-size:13px;">' +
                        '&#128193; ' + nameSafe + '</div>';
                } else if (item.is_spec) {
                    html += '<div class="_cmp2file" data-abs="' + absPathSafe + '" data-name="' + nameSafe + '" ' +
                        'style="padding:6px 10px;cursor:pointer;border-radius:6px;color:#0891b2;font-size:13px;font-weight:600;">' +
                        '&#128202; ' + nameSafe + '</div>';
                }
            });
            listDiv.innerHTML = html || '<div style="color:#6b7280;text-align:center;padding:16px;">No SPEC files found</div>';

            // Attach click handlers after rendering (no inline onclick)
            listDiv.querySelectorAll('._cmp2dir').forEach(function(el) {
                el.addEventListener('mouseover', function() { this.style.background='#f0fdfa'; });
                el.addEventListener('mouseout',  function() { this.style.background=''; });
                el.addEventListener('click', function() {
                    browseCompare2Abs(decodeURIComponent(this.dataset.abs));
                });
            });
            listDiv.querySelectorAll('._cmp2file').forEach(function(el) {
                el.addEventListener('mouseover', function() { this.style.background='#e0f2fe'; });
                el.addEventListener('mouseout',  function() { this.style.background=''; });
                el.addEventListener('click', function() {
                    loadCompareFile(decodeURIComponent(this.dataset.abs), this.dataset.name);
                });
            });
        }

        async function loadCompareFile(filePath, fileName) {
            showMessage('Loading compare file: ' + fileName + '...', 'info');
            try {
                const r = await fetch('/load_file2', {
                    method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({path: filePath})
                });
                const result = await r.json();
                if (!r.ok) { showMessage('Error loading file 2: ' + result.detail, 'error'); return; }

                dataLoaded2 = true;
                document.getElementById('compareFile2Name').textContent = fileName;
                showMessage('Compare file loaded: ' + result.total_scans + ' scans, ' + result.total_points + ' pts', 'success');

                // Populate compare controls
                const compareScans = document.getElementById('compareScans');
                compareScans.innerHTML = '';
                const lastScan2 = result.scan_numbers.length > 0 ? result.scan_numbers[result.scan_numbers.length - 1] : null;
                result.scan_numbers.forEach(function(s) {
                    const o = document.createElement('option');
                    o.value = s; o.textContent = 'Scan ' + s;
                    o.selected = (s === lastScan2);
                    compareScans.appendChild(o);
                });

                const compareYCols = document.getElementById('compareYCols');
                compareYCols.innerHTML = '';
                var cmpDefaultPicked = false;
                var cmpDefaultPriority = ['ic1','diode','I0','I1'];
                result.available_columns.forEach(function(col) {
                    const o = document.createElement('option');
                    o.value = col; o.textContent = col;
                    if (!cmpDefaultPicked && cmpDefaultPriority.indexOf(col) >= 0) {
                        o.selected = true; cmpDefaultPicked = true;
                    }
                    compareYCols.appendChild(o);
                });

                document.getElementById('compareControls').classList.remove('hidden');
            } catch(e) {
                showMessage('Failed to load compare file: ' + e.message, 'error');
            }
        }

        async function createComparePlot() {
            if (!dataLoaded) { showMessage('Load the primary file first.', 'error'); return; }
            if (!dataLoaded2) { showMessage('Load a compare file first.', 'error'); return; }

            const xCol = document.getElementById('xAxis') ? document.getElementById('xAxis').value : null;
            if (!xCol) { showMessage('Select an X-axis column first.', 'error'); return; }

            const file1Scans = Array.from(document.getElementById('scans').selectedOptions).map(function(o) { return parseInt(o.value); });
            const file1YCols = Array.from(document.getElementById('yAxis').selectedOptions).map(function(o) { return o.value; });
            const file2Scans = Array.from(document.getElementById('compareScans').selectedOptions).map(function(o) { return parseInt(o.value); });
            const file2YCols = Array.from(document.getElementById('compareYCols').selectedOptions).map(function(o) { return o.value; });
            const plotType   = document.getElementById('plotType') ? document.getElementById('plotType').value : 'line';

            if (!file1Scans.length || !file1YCols.length) { showMessage('Select scans and Y columns for file 1.', 'error'); return; }
            if (!file2Scans.length || !file2YCols.length) { showMessage('Select scans and Y columns for file 2.', 'error'); return; }

            showMessage('Creating comparison plot...', 'info');
            try {
                // Plot file 1
                const r1 = await fetch('/plot', {method:'POST',headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({x_column:xCol, y_columns:file1YCols, scans:file1Scans, plot_type:plotType, normalize:false, log_scale:false})});
                if (!r1.ok) { showMessage('Error plotting file 1.', 'error'); return; }
                const res1 = await r1.json();

                // Plot file 2
                const r2 = await fetch('/plot2', {method:'POST',headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({x_column:xCol, y_columns:file2YCols, scans:file2Scans, plot_type:plotType, normalize:false, log_scale:false})});
                if (!r2.ok) { showMessage('Error plotting file 2.', 'error'); return; }
                const res2 = await r2.json();

                // Merge traces: file2 traces get dashed lines and "[F2]" label suffix
                var file1Traces = res1.plot.data;
                var file2Traces = res2.plot.data.map(function(trace) {
                    var t = Object.assign({}, trace);
                    t.name = (t.name || '') + ' [F2]';
                    t.line = Object.assign({}, t.line || {}, {dash: 'dash'});
                    t.marker = Object.assign({}, t.marker || {}, {symbol: 'x'});
                    return t;
                });

                var layout = Object.assign({}, res1.plot.layout, {
                    title: (res1.plot.layout.title || '') + ' vs Compare File'
                });

                var plotEl = document.getElementById('plot');
                plotEl.style.display = 'block';
                Plotly.newPlot('plot', file1Traces.concat(file2Traces), layout, {responsive:true, displayModeBar:true});
                _comparePlotsActive = true;
                showMessage('Comparison plot created! File 2 traces shown with dashed lines and [F2] labels.', 'success');
                plotEl.scrollIntoView({behavior:'smooth'});
            } catch(e) {
                showMessage('Comparison plot failed: ' + e.message, 'error');
            }
        }

        function clearComparePlot() {
            _comparePlotsActive = false;
            Plotly.purge('plot');
            document.getElementById('plot').innerHTML =
                '<div style="text-align:center;padding:50px;color:#666;"><h3>Plot cleared. Create a new plot.</h3></div>';
            showMessage('Comparison overlay cleared.', 'info');
        }

        // ── Help modal ────────────────────────────────────────────────────────
        function showHelp() {
            document.getElementById('helpModal').classList.remove('hidden');
            switchHelpTab('about');
        }
        function closeHelp() {
            document.getElementById('helpModal').classList.add('hidden');
        }
        function switchHelpTab(name) {
            var tabs  = ['about','quickstart','features','format','shortcuts'];
            tabs.forEach(function(t) {
                var btn  = document.getElementById('htab-' + t);
                var pane = document.getElementById('hpane-' + t);
                if (t === name) {
                    btn.classList.add('htab-active');
                    pane.classList.remove('hidden');
                } else {
                    btn.classList.remove('htab-active');
                    pane.classList.add('hidden');
                }
            });
        }
        function switchHomePage(name) {
            var tabs = ['about','quickstart','features','format','shortcuts'];
            tabs.forEach(function(t) {
                var btn  = document.getElementById('hptab-' + t);
                var pane = document.getElementById('hppane-' + t);
                if (t === name) {
                    if (btn)  btn.classList.add('htab-active');
                    if (pane) pane.classList.remove('hidden');
                } else {
                    if (btn)  btn.classList.remove('htab-active');
                    if (pane) pane.classList.add('hidden');
                }
            });
        }

        // ── Data info modal ───────────────────────────────────────────────────
        function closeDataInfo() {
            document.getElementById('dataInfoModal').classList.add('hidden');
        }

        async function showDataInfo() {
            if (!dataLoaded) { showMessage('No data loaded. Please load a SPEC file first.', 'error'); return; }
            try {
                const response = await fetch('/data_info');
                if (!response.ok) { showMessage('No data loaded. Please load a SPEC file first.', 'error'); return; }
                const info = await response.json();

                // Subtitle
                document.getElementById('dataInfoSubtitle').textContent = info.sample_name || '';

                // Build info rows
                const rows = [
                    ['&#128196; File', info.sample_name || 'N/A'],
                    ['&#128290; Total Scans', info.total_scans],
                    ['&#128202; Data Points', info.total_points],
                    ['&#9989; Scan Info', info.scan_info_available ? 'Available' : 'Not available'],
                    ['&#128203; Scan Numbers', info.scan_numbers.join(', ')],
                    ['&#128294; Available Columns (' + info.available_columns.length + ')',
                        info.available_columns.join(', ')],
                ];

                var html = '';
                rows.forEach(function(r) {
                    html += '<div style="background:var(--surf2);border:1px solid var(--bdr);border-radius:var(--rs);padding:11px 14px;">' +
                        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--txt3);margin-bottom:5px;">' + r[0] + '</div>' +
                        '<div style="font-size:13px;color:var(--txt);word-break:break-all;line-height:1.5;">' + r[1] + '</div>' +
                        '</div>';
                });
                document.getElementById('dataInfoContent').innerHTML = html;
                document.getElementById('dataInfoModal').classList.remove('hidden');
            } catch (error) {
                showMessage('Error getting data info: ' + error.message, 'error');
            }
        }

        // Close all main panels (ensures scan table never bleeds into other views)
        function closeAllPanels(except) {
            var panels = ['plotControls', 'scanTableContainer', 'exportControls',
                          'folderTimelinePanel'];
            panels.forEach(function(id) {
                if (id !== except) {
                    var el = document.getElementById(id);
                    if (el) el.classList.add('hidden');
                }
            });
        }

        function showPlotControls() {
            const controls = document.getElementById('plotControls');
            if (!dataLoaded) { showMessage('Please load data first before accessing plot controls.', 'error'); return; }
            if (controls.classList.contains('hidden')) {
                closeAllPanels('plotControls');
                controls.classList.remove('hidden');
                controls.scrollIntoView({ behavior: 'smooth' });
                showMessage('Plot controls opened! Select your X and Y axes.', 'info');
            } else {
                controls.classList.add('hidden');
            }
        }

        function showScanTable() {
            const table = document.getElementById('scanTableContainer');
            if (!dataLoaded) { showMessage('Please load data first to view scan information.', 'error'); return; }
            if (table.classList.contains('hidden')) {
                closeAllPanels('scanTableContainer');
                table.classList.remove('hidden');
                table.scrollIntoView({ behavior: 'smooth' });
                showMessage('Scan information table opened!', 'info');
            } else {
                table.classList.add('hidden');
                return;
            }
            // Always reload table with current file's data (handles file-switch case)
            loadScanTable();
        }

        function showExportControls() {
            const controls = document.getElementById('exportControls');
            if (!dataLoaded) { showMessage('Please load data first before accessing export options.', 'error'); return; }
            if (controls.classList.contains('hidden')) {
                closeAllPanels('exportControls');
                controls.classList.remove('hidden');
                controls.scrollIntoView({ behavior: 'smooth' });
                showMessage('Export controls opened!', 'info');
                updateExportInfo();
            } else {
                controls.classList.add('hidden');
            }
        }

        async function loadScanTable() {
            try {
                const response = await fetch('/scan_info');
                if (response.ok) {
                    const data = await response.json();
                    displayScanTable(data.scan_table);
                } else {
                    document.getElementById('scanTableContent').innerHTML =
                        '<div style="text-align:center;padding:50px;color:#666;"><h4>No scan information available</h4></div>';
                }
            } catch (error) {
                console.error('Error loading scan table:', error);
            }
        }

        let _lastScanData = [];

        function displayScanTable(scanData) {
            _lastScanData = scanData;
            let tableHTML = '<table class="scan-table"><thead><tr>' +
                '<th>Scan #</th><th>Command</th><th>Timestamp</th>' +
                '<th>Temp</th><th>Count Time</th><th>Data Points</th><th>Comments</th>' +
                '</tr></thead><tbody>';

            scanData.forEach(function(scan) {
                tableHTML += '<tr>' +
                    '<td class="scan-number">' +
                        '<button class="scan-link" ' +
                            'onclick="findScanData(' + scan.scan_number + ')" ' +
                            'title="Click to navigate file browser to data folder for scan ' + scan.scan_number + '">' +
                            scan.scan_number +
                        '</button>' +
                    '</td>' +
                    '<td><div class="command">' + scan.command + '</div></td>' +
                    '<td><div class="timestamp">' + scan.timestamp + '</div></td>' +
                    '<td><div class="temperature">' + (scan.temperature || 'N/A') + '</div></td>' +
                    '<td>' + (scan.count_time || 'N/A') + '</td>' +
                    '<td>' + scan.data_points + '</td>' +
                    '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;">' + (scan.comments || '') + '</td>' +
                    '</tr>';
            });

            tableHTML += '</tbody></table>';
            document.getElementById('scanTableContent').innerHTML = tableHTML;
        }

        // ── Scan data folder navigation ───────────────────────────────────────
        async function findScanData(scanNum) {
            var box = document.getElementById('scanDataPath');
            if (box) {
                box.style.display = 'block';
                box.style.borderColor = '#93c5fd';
                box.style.background = '#eff6ff';
                box.innerHTML = '&#128269; Searching for scan <strong>' + scanNum + '</strong> data folder&hellip;';
            }
            try {
                const r = await fetch('/find_scan_data', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({scan_number: scanNum})
                });
                if (!r.ok) {
                    var err = await r.json();
                    if (box) {
                        box.style.borderColor = '#fca5a5';
                        box.style.background = '#fef2f2';
                        box.innerHTML = '&#10060; Error: ' + (err.detail || 'unknown error');
                    }
                    return;
                }
                const d = await r.json();
                if (d.found) {
                    if (box) {
                        box.style.borderColor = '#86efac';
                        box.style.background = '#f0fdf4';
                        box.innerHTML =
                            '&#128193; <strong>Scan ' + scanNum + ' data folder:</strong><br>' +
                            '<a href="#" class="_scanpath" data-abspath="' + encodeURIComponent(d.path) + '" ' +
                            'style="font-size:12px;word-break:break-all;color:#1e40af;font-family:monospace;">' +
                            d.path + '</a>' +
                            ' <span style="font-size:11px;color:#059669;">(click to open in file browser)</span>';
                        box.querySelector('._scanpath').addEventListener('click', function(e) {
                            e.preventDefault();
                            openScanFolder(decodeURIComponent(this.dataset.abspath));
                        });
                    }
                } else if (d.strategy === 'tiffs_dir') {
                    if (box) {
                        box.style.borderColor = '#fde68a';
                        box.style.background = '#fffbeb';
                        box.innerHTML =
                            '&#128193; <strong>Scan ' + scanNum + '</strong> &mdash; ' +
                            'folder <code>' + d.folder + '</code> not found. ' +
                            'Nearest location:<br>' +
                            '<a href="#" class="_scanpath" data-abspath="' + encodeURIComponent(d.path) + '" ' +
                            'style="font-size:12px;word-break:break-all;color:#92400e;font-family:monospace;">' +
                            d.path + '</a>' +
                            ' <span style="font-size:11px;color:#d97706;">(click to open in file browser)</span>';
                        box.querySelector('._scanpath').addEventListener('click', function(e) {
                            e.preventDefault();
                            openScanFolder(decodeURIComponent(this.dataset.abspath));
                        });
                    }
                } else {
                    if (box) {
                        box.style.borderColor = '#fca5a5';
                        box.style.background = '#fef2f2';
                        box.innerHTML =
                            '&#10060; <strong>Scan ' + scanNum + '</strong> &mdash; folder ' +
                            '<code>' + d.folder + '</code> not found under root path.';
                    }
                }
            } catch(e) {
                if (box) {
                    box.style.borderColor = '#fca5a5';
                    box.style.background = '#fef2f2';
                    box.innerHTML = '&#10060; Error: ' + e.message;
                }
            }
        }

        async function openScanFolder(absPath) {
            // Open the file browser and navigate to the clicked path
            var fb = document.getElementById('fileBrowser');
            if (fb && fb.classList.contains('hidden')) {
                fb.classList.remove('hidden');
            }
            await browseDirAbs(absPath);
            if (fb) fb.scrollIntoView({behavior: 'smooth'});
        }

        async function browseDirAbs(absPath) {
            // Navigate the MAIN file browser to an absolute path via /browse_abs
            // (does not change root_path on the server)
            try {
                const r = await fetch('/browse_abs', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({abs_path: absPath})
                });
                if (!r.ok) {
                    const err = await r.json();
                    showMessage('Could not open folder: ' + (err.detail || absPath), 'error');
                    return;
                }
                const data = await r.json();
                displayFilesAbs(data);
            } catch(e) {
                showMessage('Navigation error: ' + e.message, 'error');
            }
        }

        function displayFilesAbs(data) {
            // Render browse_abs results into the main file browser panel
            var breadcrumb = document.getElementById('breadcrumb');
            var fileList   = document.getElementById('fileList');
            if (!fileList) return;

            if (breadcrumb) breadcrumb.textContent = data.abs_path || '';
            var pathInput = document.getElementById('pathInput');
            if (pathInput) pathInput.value = data.abs_path || '';

            var html = '';
            (data.items || []).forEach(function(item) {
                var nameSafe = item.name.replace(/&/g,'&amp;').replace(/</g,'&lt;');
                var absSafe  = encodeURIComponent(item.abs_path);
                var sizeTxt  = item.size ? formatBytes(item.size) : '';
                if (item.type === 'directory') {
                    var parentIcon = item.is_parent ? '&#128281;' : '&#128193;';
                    html += '<div class="file-item _absdir" data-abs="' + absSafe + '">' +
                            '<span class="icon">' + parentIcon + '</span>' +
                            '<span class="name">' + nameSafe + '</span>' +
                            '</div>';
                } else if (item.is_spec) {
                    // SPEC data files — clickable to load
                    html += '<div class="file-item spec-file _absfile" data-abs="' + absSafe + '">' +
                            '<span class="icon">&#128202;</span>' +
                            '<span class="name">' + nameSafe + '</span>' +
                            '<span class="size">' + sizeTxt + '</span>' +
                            '<span class="spec-badge">SPEC</span>' +
                            '</div>';
                } else if (item.file_kind === 'image') {
                    // Detector image files (.cbf, .tif, .h5, etc.) — display only
                    html += '<div class="file-item" style="opacity:0.85;" data-abs="' + absSafe + '">' +
                            '<span class="icon">&#128247;</span>' +
                            '<span class="name">' + nameSafe + '</span>' +
                            '<span class="size">' + sizeTxt + '</span>' +
                            '<span class="spec-badge" style="background:#0d9488;">IMG</span>' +
                            '</div>';
                } else {
                    // Other files — display only
                    html += '<div class="file-item" style="opacity:0.7;" data-abs="' + absSafe + '">' +
                            '<span class="icon">&#128196;</span>' +
                            '<span class="name">' + nameSafe + '</span>' +
                            '<span class="size">' + sizeTxt + '</span>' +
                            '</div>';
                }
            });
            fileList.innerHTML = html ||
                '<div style="padding:20px;text-align:center;color:var(--txt3);">Empty folder</div>';

            // Attach click handlers (avoid inline onclick to dodge escaping issues)
            fileList.querySelectorAll('._absdir').forEach(function(el) {
                el.addEventListener('click', function() {
                    browseDirAbs(decodeURIComponent(this.dataset.abs));
                });
            });
            fileList.querySelectorAll('._absfile').forEach(function(el) {
                el.addEventListener('click', function() {
                    loadFile(decodeURIComponent(this.dataset.abs));
                });
            });
        }

        // ── X-axis auto-selection from scan command ───────────────────────────
        function extractMotorFromCommand(command, availableCols) {
            // command examples: "ascan mond 6.8 6.9 160 0.1"
            //                   "flyscan phi 0 365 3650 0.1"
            //                   "dscan th -1 1 100 0.5"
            if (!command) return null;
            var tokens = command.trim().split(/\s+/);
            // token[0] = scan type, token[1] = motor
            if (tokens.length < 2) return null;
            var motor = tokens[1];
            // Case-sensitive first, then case-insensitive
            if (availableCols.indexOf(motor) >= 0) return motor;
            var motorLow = motor.toLowerCase();
            for (var k = 0; k < availableCols.length; k++) {
                if (availableCols[k].toLowerCase() === motorLow) return availableCols[k];
            }
            return null;
        }

        function setXAxisFromScanNum(scanNum, availableCols) {
            // Look up the scan command in _lastScanData, auto-set X dropdown
            var scanEntry = null;
            for (var i = 0; i < _lastScanData.length; i++) {
                // Use loose equality to handle int/string differences
                if (_lastScanData[i].scan_number == scanNum) { scanEntry = _lastScanData[i]; break; }
            }
            if (!scanEntry) return;
            var motor = extractMotorFromCommand(scanEntry.command, availableCols);
            if (motor) {
                var xAxis = document.getElementById('xAxis');
                if (xAxis) xAxis.value = motor;
            }
        }

        function onScanSelectionChange() {
            // Called when user changes scan selection - auto-update X axis
            var scansEl = document.getElementById('scans');
            if (!scansEl || !scansEl.options.length) return;
            var availableCols = Array.from(document.getElementById('xAxis').options)
                .map(function(o) { return o.value; }).filter(function(v) { return v !== ''; });
            // Use the LAST selected scan for X-axis auto-select
            var selectedOpts = Array.from(scansEl.selectedOptions);
            if (selectedOpts.length === 0) return;
            var lastSelected = parseInt(selectedOpts[selectedOpts.length - 1].value);
            setXAxisFromScanNum(lastSelected, availableCols);
        }

        async function updatePlotControls() {
            try {
                const response = await fetch('/data_info');
                if (!response.ok) return;
                const info = await response.json();

                // ── Populate X-axis dropdown ───────────────────────────────
                const xAxis = document.getElementById('xAxis');
                xAxis.innerHTML = '<option value="">Select X column...</option>';
                info.available_columns.forEach(function(col) {
                    const option = document.createElement('option');
                    option.value = col; option.textContent = col; xAxis.appendChild(option);
                });

                // ── Populate Y-axis dropdown ───────────────────────────────
                const yAxis = document.getElementById('yAxis');
                yAxis.innerHTML = '';
                info.available_columns.forEach(function(col) {
                    const option = document.createElement('option');
                    option.value = col; option.textContent = col; yAxis.appendChild(option);
                });
                const defaultY = ['ic1','diode','I0','I1'].filter(function(col) {
                    return info.available_columns.indexOf(col) >= 0;
                }).slice(0, 1);
                if (defaultY.length === 0) {
                    info.available_columns.slice(0, 1).forEach(function(c) { defaultY.push(c); });
                }
                Array.from(yAxis.options).forEach(function(option) {
                    option.selected = defaultY.indexOf(option.value) >= 0;
                });

                // ── Populate scans dropdown: only select LAST scan ─────────
                const lastScanNum = info.scan_numbers.length > 0
                    ? info.scan_numbers[info.scan_numbers.length - 1] : null;
                const scans = document.getElementById('scans');
                scans.innerHTML = '';
                info.scan_numbers.forEach(function(scan) {
                    const option = document.createElement('option');
                    option.value = scan;
                    option.textContent = 'Scan ' + scan;
                    option.selected = (scan === lastScanNum);  // only last scan selected
                    scans.appendChild(option);
                });

                // ── Fetch scan_info to populate _lastScanData and auto-set X ──
                try {
                    const siResp = await fetch('/scan_info');
                    if (siResp.ok) {
                        const siData = await siResp.json();
                        _lastScanData = siData.scan_table;  // keep in sync
                        if (lastScanNum !== null) {
                            // Find last scan entry (use == for int/string safety)
                            var lastEntry = null;
                            for (var idx = 0; idx < siData.scan_table.length; idx++) {
                                if (siData.scan_table[idx].scan_number == lastScanNum) {
                                    lastEntry = siData.scan_table[idx]; break;
                                }
                            }
                            if (lastEntry) {
                                var motor = extractMotorFromCommand(lastEntry.command, info.available_columns);
                                if (motor) {
                                    xAxis.value = motor;
                                }
                            }
                        }
                    }
                } catch(e) { /* ignore scan_info fetch errors */ }

                // Fallback X-axis if nothing was auto-selected
                if (!xAxis.value) {
                    xAxis.value = info.available_columns.find(function(col) {
                        return ['Time','Epoch','time','epoch'].indexOf(col) >= 0;
                    }) || info.available_columns[0] || '';
                }

                // ── Fit controls ───────────────────────────────────────────
                const fitYColumn = document.getElementById('fitYColumn');
                fitYColumn.innerHTML = '';
                info.available_columns.forEach(function(col) {
                    const o = document.createElement('option');
                    o.value = col; o.textContent = col; fitYColumn.appendChild(o);
                });

                const fitScan = document.getElementById('fitScan');
                fitScan.innerHTML = '';
                info.scan_numbers.forEach(function(s) {
                    const o = document.createElement('option');
                    o.value = s; o.textContent = 'Scan ' + s; fitScan.appendChild(o);
                });

            } catch (error) {
                console.error('Error updating plot controls:', error);
            }
        }

        async function updateExportInfo() {
            try {
                const response = await fetch('/export_info');
                if (response.ok) {
                    exportInfo = await response.json();

                    const exportScans = document.getElementById('exportScans');
                    exportScans.innerHTML = '';
                    exportInfo.available_scans.forEach(function(scan) {
                        const option = document.createElement('option');
                        option.value = scan; option.textContent = 'Scan ' + scan; option.selected = true;
                        exportScans.appendChild(option);
                    });

                    const exportColumns = document.getElementById('exportColumns');
                    exportColumns.innerHTML = '';
                    exportInfo.available_columns.forEach(function(col) {
                        const option = document.createElement('option');
                        option.value = col; option.textContent = col; option.selected = true;
                        exportColumns.appendChild(option);
                    });

                    const plottedInfo = document.getElementById('plottedDataInfo');
                    if (exportInfo.has_plotted_data) {
                        const plotInfo = exportInfo.plotted_data_info;
                        plottedInfo.innerHTML = 'Last plot: ' + plotInfo.y_columns.join(', ') + ' vs ' + plotInfo.x_column + ' (' + plotInfo.rows + ' rows, scans: ' + plotInfo.scans.join(', ') + ')';
                        document.getElementById('exportPlotted').disabled = false;
                    } else {
                        plottedInfo.innerHTML = 'No plot data available. Create a plot first.';
                        document.getElementById('exportPlotted').disabled = true;
                    }
                }
            } catch (error) {
                console.error('Error updating export info:', error);
            }
        }

        async function createCustomPlot() {
            try {
                const xColumn = document.getElementById('xAxis').value;
                const yColumns = Array.from(document.getElementById('yAxis').selectedOptions).map(function(o) { return o.value; });
                const selectedScans = Array.from(document.getElementById('scans').selectedOptions).map(function(o) { return parseInt(o.value); });
                const plotType = document.getElementById('plotType').value;
                const normalize = document.getElementById('normalize').checked;
                const logScale = document.getElementById('logScale').checked;

                if (!xColumn) { showMessage('Please select an X-axis column.', 'error'); return; }
                if (yColumns.length === 0) { showMessage('Please select at least one Y-axis column.', 'error'); return; }
                if (selectedScans.length === 0) { showMessage('Please select at least one scan.', 'error'); return; }

                showMessage('Creating plot: ' + yColumns.join(', ') + ' vs ' + xColumn + '...', 'info');

                const response = await fetch('/plot', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({x_column: xColumn, y_columns: yColumns, scans: selectedScans,
                                          plot_type: plotType, normalize: normalize, log_scale: logScale})
                });

                if (response.ok) {
                    const result = await response.json();
                    var plotEl = document.getElementById('plot');
                    plotEl.style.display = 'block';
                    Plotly.newPlot('plot', result.plot.data, result.plot.layout, {
                        responsive: true, displayModeBar: true,
                        modeBarButtonsToAdd: ['drawline','drawopenpath','drawclosedpath','drawcircle','drawrect','eraseshape']
                    });
                    showMessage('Plot created: ' + yColumns.join(', ') + ' vs ' + xColumn, 'success');
                    updateExportInfo();
                    await runFits(xColumn, yColumns, selectedScans);
                    plotEl.scrollIntoView({ behavior: 'smooth' });
                } else {
                    const error = await response.json();
                    showMessage('Plot error: ' + error.detail, 'error');
                }
            } catch (error) {
                showMessage('Failed to create plot: ' + error.message, 'error');
            }
        }

        async function exportCSV() {
            try {
                const exportType = document.querySelector('input[name="exportType"]:checked').value;
                let exportRequest = { export_type: exportType };

                if (exportType === 'selected') {
                    const selectedScans = Array.from(document.getElementById('exportScans').selectedOptions).map(function(o) { return parseInt(o.value); });
                    const selectedColumns = Array.from(document.getElementById('exportColumns').selectedOptions).map(function(o) { return o.value; });
                    if (selectedScans.length === 0) { showMessage('Please select at least one scan.', 'error'); return; }
                    exportRequest.scans = selectedScans;
                    exportRequest.columns = selectedColumns;
                }

                showMessage('Preparing CSV export...', 'info');
                const response = await fetch('/export_csv', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(exportRequest)
                });

                if (response.ok) {
                    const contentDisposition = response.headers.get('Content-Disposition');
                    let filename = 'spec_data.csv';
                    if (contentDisposition) {
                        const m = contentDisposition.match(/filename=(.+)/);
                        if (m) filename = m[1];
                    }
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url; a.download = filename;
                    document.body.appendChild(a); a.click();
                    window.URL.revokeObjectURL(url); document.body.removeChild(a);
                    showMessage('CSV downloaded: ' + filename, 'success');
                } else {
                    const error = await response.json();
                    showMessage('Export failed: ' + error.detail, 'error');
                }
            } catch (error) {
                showMessage('Export error: ' + error.message, 'error');
            }
        }

        document.addEventListener('change', function(event) {
            if (event.target.name === 'exportType') {
                document.getElementById('customExportOptions').style.display =
                    event.target.value === 'selected' ? 'block' : 'none';
            }
        });

        function formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        document.getElementById('xAxis').addEventListener('change', function() {
            const yAxis = document.getElementById('yAxis');
            const xValue = this.value;
            Array.from(yAxis.options).forEach(function(option) {
                if (option.value === xValue && option.selected) option.selected = false;
            });
        });

        document.addEventListener('keydown', function(event) {
            // Escape closes any open modal
            if (event.key === 'Escape') {
                closeDataInfo();
                closeHelp();
                return;
            }
            if (event.ctrlKey || event.metaKey) {
                switch(event.key) {
                    case 'o': event.preventDefault(); toggleFileBrowser(); break;
                    case 'p': event.preventDefault(); if (dataLoaded) showPlotControls(); break;
                    case 't': event.preventDefault(); if (dataLoaded) showScanTable(); break;
                    case 'e': event.preventDefault(); if (dataLoaded) showExportControls(); break;
                    case 'i': event.preventDefault(); showDataInfo(); break;
                    case 's':
                        if (!document.getElementById('exportControls').classList.contains('hidden')) {
                            event.preventDefault(); exportCSV();
                        }
                        break;
                }
            }
        });

        // ── Auto-refresh ──────────────────────────────────────────────────────
        let _refreshTimer = null;
        let _lastMtime = null;
        let _lastSize  = null;

        async function toggleAutoRefresh() {
            const btn = document.getElementById('refreshBtn');
            const statusEl = document.getElementById('refreshStatus');
            if (_refreshTimer) {
                clearInterval(_refreshTimer);
                _refreshTimer = null;
                btn.innerHTML = '&#9654; Auto-Refresh';
                btn.classList.remove('stop');
                if (statusEl) statusEl.style.display = 'none';
                return;
            }
            if (!dataLoaded) { showMessage('Load a file first before enabling auto-refresh.', 'error'); return; }

            try {
                const r = await fetch('/file_status');
                if (!r.ok) { showMessage('Could not get file status.', 'error'); return; }
                const s = await r.json();
                _lastMtime = s.mtime; _lastSize = s.size;
            } catch(e) { showMessage(e.message, 'error'); return; }

            const intervalSec = parseInt(document.getElementById('refreshInterval').value);
            btn.innerHTML = '&#9646;&#9646; Stop';
            btn.classList.add('stop');
            if (statusEl) { statusEl.style.display = 'inline'; statusEl.textContent = 'Watching \u2014 every ' + intervalSec + 's'; }

            _refreshTimer = setInterval(async function() {
                try {
                    const r = await fetch('/file_status');
                    if (!r.ok) return;
                    const s = await r.json();

                    if (s.mtime !== _lastMtime || s.size !== _lastSize) {
                        _lastMtime = s.mtime; _lastSize = s.size;
                        if (document.getElementById('refreshStatus')) document.getElementById('refreshStatus').textContent = 'Reloading\u2026';

                        const rr = await fetch('/reload_file', {method:'POST'});
                        if (!rr.ok) return;
                        const info = await rr.json();

                        await updatePlotControls();
                        updateExportInfo();
                        // Only refresh scan table if it's currently open
                        if (!document.getElementById('scanTableContainer').classList.contains('hidden')) {
                            await loadScanTable();
                        }

                        // Auto-refresh: only replot the last scan number
                        const infoR = await (await fetch('/data_info')).json();
                        const scanNums = infoR.scan_numbers;
                        if (scanNums && scanNums.length > 0) {
                            const lastScan = scanNums[scanNums.length - 1];
                            const x  = document.getElementById('xAxis') ? document.getElementById('xAxis').value : null;
                            const ys = document.getElementById('yAxis') ?
                                Array.from(document.getElementById('yAxis').selectedOptions).map(function(o) { return o.value; }) : [];
                            if (x && ys.length > 0) {
                                const pr = await fetch('/plot', {
                                    method:'POST',
                                    headers:{'Content-Type':'application/json'},
                                    body: JSON.stringify({
                                        x_column: x, y_columns: ys, scans: [lastScan],
                                        plot_type: document.getElementById('plotType') ? document.getElementById('plotType').value : 'line',
                                        normalize: document.getElementById('normalize') ? document.getElementById('normalize').checked : false,
                                        log_scale: document.getElementById('logScale') ? document.getElementById('logScale').checked : false
                                    })
                                });
                                if (pr.ok) {
                                    const res = await pr.json();
                                    Plotly.react('plot', res.plot.data, res.plot.layout, {responsive:true, displayModeBar:true});
                                }
                            }
                        }

                        const now = new Date().toLocaleTimeString();
                        document.getElementById('refreshStatus').textContent =
                            'Updated at ' + now + ' - ' + info.total_scans + ' scans, ' + info.total_points + ' pts';
                        showMessage('File updated: ' + info.total_scans + ' scans, ' + info.total_points + ' pts', 'success');
                    } else {
                        const now = new Date().toLocaleTimeString();
                        document.getElementById('refreshStatus').textContent = 'Watching - no change as of ' + now;
                    }
                } catch(e) { console.error('Auto-refresh error:', e); }
            }, intervalSec * 1000);
        }

        function data_store_js() {
            const x  = document.getElementById('xAxis') ? document.getElementById('xAxis').value : null;
            const ys = document.getElementById('yAxis') ?
                Array.from(document.getElementById('yAxis').selectedOptions).map(function(o) { return o.value; }) : [];
            const sc = document.getElementById('scans') ?
                Array.from(document.getElementById('scans').selectedOptions).map(function(o) { return parseInt(o.value); }) : [];
            if (!x || !ys.length || !sc.length) return null;
            return {
                x_column: x, y_columns: ys, scans: sc,
                plot_type: document.getElementById('plotType') ? document.getElementById('plotType').value : 'line',
                normalize: document.getElementById('normalize') ? document.getElementById('normalize').checked : false,
                log_scale: document.getElementById('logScale') ? document.getElementById('logScale').checked : false,
            };
        }

        function toggleFitOptions() {
            const val = document.getElementById('fitType').value;
            document.getElementById('fitOptions').style.display = val === 'none' ? 'none' : 'block';
        }

        async function runFits(xColumn, yColumns, scans) {
            const fitType = document.getElementById('fitType').value;
            if (fitType === 'none') {
                document.getElementById('fitResultsPanel').classList.add('hidden');
                return;
            }

            const fitYCol  = document.getElementById('fitYColumn').value;
            const fitScanV = parseInt(document.getElementById('fitScan').value);
            if (!fitYCol || isNaN(fitScanV)) return;

            try {
                const r = await fetch('/fit', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({x_column: xColumn, y_column: fitYCol, scan: fitScanV, fit_type: fitType})
                });
                const result = await r.json();

                if (result.success && result.fit_curve) {
                    Plotly.addTraces('plot', [{
                        x: result.fit_curve.x,
                        y: result.fit_curve.y,
                        mode: 'lines',
                        name: (result.stats.fit_type || fitType) + ' fit - Scan ' + fitScanV + ' ' + fitYCol,
                        line: {dash: 'dash', width: 2, color: '#e74c3c'}
                    }]);
                }

                const s = result.stats;
                const fmt = function(v) { return (v !== undefined && v !== null) ? Number(v).toPrecision(6) : 'N/A'; };
                let html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;">';

                const cards = [
                    ['Peak Position', fmt(s.peak_position !== undefined ? s.peak_position : s.peak_x), s.fit_type ? 'from ' + s.fit_type + ' fit' : 'x at max y'],
                    ['FWHM',          fmt(s.fwhm),      s.fit_type || ''],
                    ['Mean (y)',       fmt(s.mean),      ''],
                    ['Max (y)',        fmt(s.max),       ''],
                    ['Min (y)',        fmt(s.min),       ''],
                    ['Std Dev',       fmt(s.std),       ''],
                    ['Delta (Max-Min)', fmt(s.delta),   ''],
                    ['Amplitude',     fmt(s.amplitude), s.fit_type || ''],
                ];

                for (let ci = 0; ci < cards.length; ci++) {
                    const label = cards[ci][0], value = cards[ci][1], note = cards[ci][2];
                    if (value === 'N/A') continue;
                    html += '<div style="background:var(--surf2);border-radius:var(--rs);padding:12px;border-left:4px solid var(--pri);">' +
                        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--txt3);">' + label + '</div>' +
                        '<div style="font-size:18px;font-weight:700;color:var(--txt);margin:4px 0;font-family:monospace;">' + value + '</div>' +
                        (note ? '<div style="font-size:10px;color:var(--txt3);">' + note + '</div>' : '') +
                        '</div>';
                }

                if (!result.success) {
                    html += '<div style="grid-column:1/-1;background:#fff3cd;border-radius:8px;padding:12px;color:#856404;">' +
                        'Fit failed: ' + result.error + ' - showing raw stats only.</div>';
                }

                html += '</div>';
                document.getElementById('fitResultsContent').innerHTML = html;
                document.getElementById('fitResultsPanel').classList.remove('hidden');
                document.getElementById('fitResultsPanel').scrollIntoView({behavior: 'smooth'});

            } catch(e) {
                showMessage('Fit error: ' + e.message, 'error');
            }
        }

        // ── Notebook ──────────────────────────────────────────────────────────
        let _notes = [];

        function showNotebook() {
            const panel = document.getElementById('notebookPanel');
            panel.classList.toggle('hidden');
            if (!panel.classList.contains('hidden')) {
                panel.scrollIntoView({behavior: 'smooth'});
            }
        }

        // ── Folder Timeline ───────────────────────────────────────────────────
        let _timelineData = null;  // cached response from /folder_timeline

        function showFolderTimeline() {
            // Determine current folder from breadcrumb or pathInput
            var folderPath = '';
            var bc = document.getElementById('breadcrumb');
            if (bc && bc.textContent.trim()) {
                folderPath = bc.textContent.trim();
            } else {
                var pi = document.getElementById('pathInput');
                if (pi) folderPath = pi.value.trim();
            }
            if (!folderPath) {
                showMessage('Navigate to a folder in the File Browser first, then click Timeline.', 'error');
                return;
            }
            var panel = document.getElementById('folderTimelinePanel');
            if (panel) {
                closeAllPanels('folderTimelinePanel');
                panel.classList.remove('hidden');
                panel.scrollIntoView({behavior: 'smooth'});
            }
            loadFolderTimeline(folderPath);
        }

        async function loadFolderTimeline(folderPath) {
            var content = document.getElementById('folderTimelineContent');
            var sub     = document.getElementById('folderTimelineSub');
            // Clear any previous scan-path result
            var tsdp = document.getElementById('timelineScanDataPath');
            if (tsdp) { tsdp.style.display = 'none'; tsdp.innerHTML = ''; }
            if (content) content.innerHTML =
                '<div style="text-align:center;padding:40px;color:var(--txt3);">' +
                '&#128197; Scanning SPEC files in <code>' + folderPath + '</code>&hellip;</div>';
            try {
                var r = await fetch('/folder_timeline', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({abs_path: folderPath})
                });
                if (!r.ok) {
                    var err = await r.json();
                    if (content) content.innerHTML =
                        '<div style="padding:20px;color:#dc2626;">&#10060; ' +
                        (err.detail || 'Failed to load timeline') + '</div>';
                    return;
                }
                var data = await r.json();
                _timelineData = data;
                if (sub) sub.textContent =
                    data.folder + '  \u2014  ' + data.total + ' scans across all SPEC files';
                displayFolderTimeline(data);
            } catch(e) {
                if (content) content.innerHTML =
                    '<div style="padding:20px;color:#dc2626;">&#10060; Error: ' + e.message + '</div>';
            }
        }

        function displayFolderTimeline(data) {
            var content = document.getElementById('folderTimelineContent');
            if (!content) return;
            if (!data.rows || data.rows.length === 0) {
                content.innerHTML =
                    '<div style="padding:30px;text-align:center;color:var(--txt3);">' +
                    'No SPEC scans found in this folder.</div>';
                return;
            }

            // Assign a distinct colour to each unique SPEC file name
            var fileNames = [];
            data.rows.forEach(function(r) {
                if (fileNames.indexOf(r.spec_file) < 0) fileNames.push(r.spec_file);
            });
            var palette = ['#bfdbfe','#bbf7d0','#fde68a','#fecaca','#ddd6fe',
                           '#fed7aa','#a7f3d0','#fbcfe8','#e0e7ff','#d1fae5'];
            var fileColor = {};
            fileNames.forEach(function(n, i) {
                fileColor[n] = palette[i % palette.length];
            });

            var html = '<div class="timeline-table-scroll"><table class="scan-table"><thead><tr>' +
                '<th>Timestamp</th>' +
                '<th>SPEC File</th>' +
                '<th>Scan #</th>' +
                '<th>Command</th>' +
                '<th>Temp (K)</th>' +
                '<th>Count Time</th>' +
                '<th>Data Points</th>' +
                '<th>Comments</th>' +
                '</tr></thead><tbody>';

            data.rows.forEach(function(row, idx) {
                var bg    = fileColor[row.spec_file] || '#f0f9ff';
                var badge = '<span class="tl-file-badge" style="background:' + bg + ';">' +
                            row.spec_file + '</span>';
                var ts    = row.timestamp ?
                    '<span class="tl-ts">' + row.timestamp + '</span>' : '&mdash;';
                var temp  = row.temperature ? row.temperature + ' K' : '&mdash;';
                var ct    = row.count_time || '&mdash;';
                var dp    = row.data_points > 0 ? row.data_points : '&mdash;';
                var cmd   = '<div class="command" style="max-width:160px;">' +
                            (row.command || '&mdash;') + '</div>';
                var cmt   = '<div style="max-width:200px;overflow:hidden;text-overflow:ellipsis;' +
                            'white-space:nowrap;font-size:11px;color:var(--txt2);">' +
                            (row.comments || '') + '</div>';
                // Scan # becomes a clickable button; store lookup params in data- attrs
                var scanBtn = '<button class="scan-link _tl-scan" ' +
                    'data-scan="' + row.scan_number + '" ' +
                    'data-spec="' + encodeURIComponent(row.spec_file) + '" ' +
                    'data-folder="' + encodeURIComponent(data.folder) + '" ' +
                    'title="Click to find data folder for ' + row.spec_file + ' scan ' + row.scan_number + '">' +
                    row.scan_number + '</button>';
                html += '<tr>' +
                    '<td>' + ts + '</td>' +
                    '<td>' + badge + '</td>' +
                    '<td style="text-align:center;">' + scanBtn + '</td>' +
                    '<td>' + cmd + '</td>' +
                    '<td style="text-align:center;">' + temp + '</td>' +
                    '<td style="text-align:center;">' + ct + '</td>' +
                    '<td style="text-align:center;">' + dp + '</td>' +
                    '<td>' + cmt + '</td>' +
                    '</tr>';
            });

            html += '</tbody></table></div>';
            content.innerHTML = html;

            // Attach click handlers after innerHTML is set (no inline onclick)
            content.querySelectorAll('._tl-scan').forEach(function(btn) {
                btn.addEventListener('click', function() {
                    findTimelineScanData(
                        parseInt(this.dataset.scan),
                        decodeURIComponent(this.dataset.spec),
                        decodeURIComponent(this.dataset.folder)
                    );
                });
            });
        }

        async function findTimelineScanData(scanNum, specFile, folderPath) {
            var box = document.getElementById('timelineScanDataPath');
            if (box) {
                box.style.display  = 'block';
                box.style.borderColor = '#93c5fd';
                box.style.background  = '#eff6ff';
                box.innerHTML = '&#128269; Searching for <strong>' + specFile +
                    '</strong> scan <strong>' + scanNum + '</strong> data folder&hellip;';
            }
            try {
                var r = await fetch('/find_scan_data', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        scan_number: scanNum,
                        spec_file:   specFile,
                        spec_parent: folderPath
                    })
                });
                if (!r.ok) {
                    var err = await r.json();
                    if (box) {
                        box.style.borderColor = '#fca5a5';
                        box.style.background  = '#fef2f2';
                        box.innerHTML = '&#10060; Error: ' + (err.detail || 'unknown error');
                    }
                    return;
                }
                var d = await r.json();
                if (d.found) {
                    if (box) {
                        box.style.borderColor = '#86efac';
                        box.style.background  = '#f0fdf4';
                        box.innerHTML =
                            '&#128193; <strong>' + specFile + ' &mdash; Scan ' + scanNum +
                            ' data folder:</strong><br>' +
                            '<a href="#" class="_tlscanpath" data-abspath="' +
                            encodeURIComponent(d.path) + '" ' +
                            'style="font-size:12px;word-break:break-all;color:#1e40af;font-family:monospace;">' +
                            d.path + '</a>' +
                            ' <span style="font-size:11px;color:#059669;">(click to open in file browser)</span>';
                        box.querySelector('._tlscanpath').addEventListener('click', function(e) {
                            e.preventDefault();
                            openScanFolder(decodeURIComponent(this.dataset.abspath));
                        });
                    }
                } else if (d.strategy === 'tiffs_dir') {
                    if (box) {
                        box.style.borderColor = '#fde68a';
                        box.style.background  = '#fffbeb';
                        box.innerHTML =
                            '&#128193; <strong>' + specFile + ' &mdash; Scan ' + scanNum +
                            '</strong> &mdash; folder <code>' + d.folder + '</code> not found. ' +
                            'Nearest location:<br>' +
                            '<a href="#" class="_tlscanpath" data-abspath="' +
                            encodeURIComponent(d.path) + '" ' +
                            'style="font-size:12px;word-break:break-all;color:#92400e;font-family:monospace;">' +
                            d.path + '</a>' +
                            ' <span style="font-size:11px;color:#d97706;">(click to open in file browser)</span>';
                        box.querySelector('._tlscanpath').addEventListener('click', function(e) {
                            e.preventDefault();
                            openScanFolder(decodeURIComponent(this.dataset.abspath));
                        });
                    }
                } else {
                    if (box) {
                        box.style.borderColor = '#fca5a5';
                        box.style.background  = '#fef2f2';
                        box.innerHTML =
                            '&#10060; <strong>' + specFile + ' &mdash; Scan ' + scanNum +
                            '</strong> &mdash; folder <code>' + d.folder + '</code> not found.';
                    }
                }
            } catch(e) {
                if (box) {
                    box.style.borderColor = '#fca5a5';
                    box.style.background  = '#fef2f2';
                    box.innerHTML = '&#10060; Error: ' + e.message;
                }
            }
        }

        function downloadTimelineCSV() {
            if (!_timelineData || !_timelineData.rows || _timelineData.rows.length === 0) {
                showMessage('No timeline data to export.', 'error'); return;
            }
            var header = 'Timestamp,SPEC File,Scan #,Command,Temp (K),Count Time,Data Points,Comments';
            var lines  = [header];
            _timelineData.rows.forEach(function(r) {
                function esc(v) {
                    var s = (v === null || v === undefined) ? '' : String(v);
                    return '"' + s.replace(/"/g, '""') + '"';
                }
                lines.push([
                    esc(r.timestamp), esc(r.spec_file), r.scan_number,
                    esc(r.command), esc(r.temperature), esc(r.count_time),
                    r.data_points, esc(r.comments)
                ].join(','));
            });
            var blob = new Blob([lines.join('\\n')], {type:'text/csv'});
            var a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'folder_timeline.csv';
            a.click();
        }

        // ── Motor Positions ───────────────────────────────────────────────────
        let _motorData = null;   // {motors, scans} from /motor_positions

        function showMotorPositions() {
            if (!dataLoaded) { showMessage('Load a SPEC file first.', 'error'); return; }
            const panel = document.getElementById('motorPanel');
            if (panel.classList.contains('hidden')) {
                panel.classList.remove('hidden');
                panel.scrollIntoView({behavior: 'smooth'});
                loadMotorPositions();
            } else {
                panel.classList.add('hidden');
            }
        }

        async function loadMotorPositions() {
            try {
                const r = await fetch('/motor_positions');
                if (!r.ok) {
                    document.getElementById('motorTableContent').innerHTML =
                        '<div style="padding:20px;text-align:center;color:#dc2626;">No motor position data found in this file.</div>';
                    return;
                }
                _motorData = await r.json();
                renderMotorTable(_motorData);
            } catch(e) {
                document.getElementById('motorTableContent').innerHTML =
                    '<div style="padding:20px;color:#dc2626;">Error: ' + e.message + '</div>';
            }
        }

        function renderMotorTable(data) {
            const filterVal   = (document.getElementById('motorFilter')   || {}).value || '';
            const hideZero    = (document.getElementById('motorHideZero') || {}).checked || false;
            const filterLower = filterVal.toLowerCase();

            const motors = data.motors || [];
            const scans  = data.scans  || [];

            if (motors.length === 0) {
                document.getElementById('motorTableContent').innerHTML =
                    '<div style="padding:20px;text-align:center;color:var(--txt3);">No motor names (#O lines) found in this file.</div>';
                return;
            }

            // Build header row: Name | Mnemonic | Scan1 | Scan2 | ...
            let hdr = '<tr><th style="min-width:130px;">Motor Name</th>' +
                      '<th style="min-width:80px;">Mnemonic</th>';
            scans.forEach(function(s) {
                hdr += '<th class="scan-hdr">S' + s.scan_number + '</th>';
            });
            hdr += '</tr>';

            // Build data rows — one row per motor
            let rowsHTML = '';
            let visibleCount = 0;

            motors.forEach(function(m) {
                // Apply name/mnemonic filter
                if (filterLower &&
                    m.name.toLowerCase().indexOf(filterLower) < 0 &&
                    m.mnemonic.toLowerCase().indexOf(filterLower) < 0) return;

                // Collect positions for this motor across all scans
                var positions = scans.map(function(s) {
                    var pos = s.positions;
                    if (!pos || m.index >= pos.length) return null;
                    return pos[m.index];
                });

                // Apply hide-zero filter
                if (hideZero) {
                    var allZeroOrNull = positions.every(function(p) {
                        return p === null || p === 0 || p === 0.0;
                    });
                    if (allZeroOrNull) return;
                }

                visibleCount++;
                var row = '<tr>' +
                    '<td class="mtr-name">' + m.name + '</td>' +
                    '<td class="mtr-mnem">' + m.mnemonic + '</td>';

                positions.forEach(function(p) {
                    if (p === null || p === undefined) {
                        row += '<td class="pos-na">—</td>';
                    } else if (p === 0) {
                        row += '<td class="pos-zero">0</td>';
                    } else {
                        // Format: up to 6 significant figures, trim trailing zeros
                        var formatted = parseFloat(p.toPrecision(6)).toString();
                        row += '<td class="pos-val">' + formatted + '</td>';
                    }
                });
                row += '</tr>';
                rowsHTML += row;
            });

            if (!rowsHTML) {
                rowsHTML = '<tr><td colspan="' + (scans.length + 2) +
                    '" style="text-align:center;padding:20px;color:var(--txt3);">No motors match your filter.</td></tr>';
            }

            var label = document.getElementById('motorCountLabel');
            if (label) label.textContent = visibleCount + ' of ' + motors.length + ' motors shown';

            document.getElementById('motorTableContent').innerHTML =
                '<table class="motor-table"><thead>' + hdr + '</thead><tbody>' + rowsHTML + '</tbody></table>';
        }

        function filterMotorTable() {
            if (_motorData) renderMotorTable(_motorData);
        }

        function downloadMotorCSV() {
            if (!_motorData || !_motorData.motors || !_motorData.scans) {
                showMessage('Load motor data first.', 'error'); return;
            }
            const motors = _motorData.motors;
            const scans  = _motorData.scans;
            const header = ['Motor Name', 'Mnemonic'].concat(scans.map(function(s) { return 'Scan ' + s.scan_number; }));
            const rows   = motors.map(function(m) {
                var cells = ['"' + m.name + '"', m.mnemonic];
                scans.forEach(function(s) {
                    var p = (s.positions && m.index < s.positions.length) ? s.positions[m.index] : null;
                    cells.push(p === null || p === undefined ? '' : p);
                });
                return cells.join(',');
            });
            const csv  = [header.join(',')].concat(rows).join('\\n');
            const blob = new Blob([csv], {type: 'text/csv'});
            const a    = document.createElement('a');
            a.href     = URL.createObjectURL(blob);
            const fname = exportInfo ? exportInfo.filename || 'spec' : 'spec';
            a.download = 'motor_positions_' + fname + '_' + new Date().toISOString().slice(0,10) + '.csv';
            a.click(); URL.revokeObjectURL(a.href);
        }

        function addNote() {
            const input = document.getElementById('noteInput');
            const text = input.value.trim();
            if (!text) { showMessage('Please type a note first.', 'error'); return; }
            _notes.push({time: new Date().toLocaleString(), text: text});
            input.value = '';
            renderNotes();
        }

        function renderNotes() {
            const container = document.getElementById('notebookEntries');
            if (_notes.length === 0) { container.innerHTML = ''; return; }
            let html = '';
            for (let i = 0; i < _notes.length; i++) {
                const n = _notes[i];
                const safeText = n.text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
                html += '<div class="notebook-entry">' +
                    '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">' +
                    '<span style="font-size:11px;color:#92400e;white-space:nowrap;">' + n.time + '</span>' +
                    '<button onclick="deleteNote(' + i + ')" ' +
                    'style="background:none;border:none;cursor:pointer;color:#dc2626;font-size:14px;padding:0;flex-shrink:0;">&#128465;</button>' +
                    '</div>' +
                    '<div style="margin-top:6px;white-space:pre-wrap;">' + safeText + '</div>' +
                    '</div>';
            }
            container.innerHTML = html;
        }

        function deleteNote(i) { _notes.splice(i, 1); renderNotes(); }

        function clearNotes() {
            if (_notes.length === 0) return;
            if (!confirm('Clear all notes? This cannot be undone.')) return;
            _notes = []; renderNotes();
        }

        function downloadNotes() {
            if (_notes.length === 0) { showMessage('No notes to download.', 'error'); return; }
            const filename = exportInfo ? exportInfo.filename || 'spec' : 'spec';
            let text = 'Lab Notebook - ' + filename + '\\nExported: ' + new Date().toLocaleString() + '\\n' +
                '============================================================\\n\\n';
            for (let i = 0; i < _notes.length; i++) {
                text += '[' + (i+1) + '] ' + _notes[i].time + '\\n' + _notes[i].text + '\\n\\n';
            }
            const blob = new Blob([text], {type: 'text/plain'});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'notebook_' + filename + '_' + new Date().toISOString().slice(0,10) + '.txt';
            a.click(); URL.revokeObjectURL(a.href);
        }

        async function downloadPlotWithNotes() {
            const filename = exportInfo ? exportInfo.filename || 'spec' : 'spec';
            let plotImgSrc = '';
            try {
                plotImgSrc = await Plotly.toImage('plot', {format: 'png', width: 900, height: 500});
            } catch(e) {
                showMessage('Could not capture plot image. Create a plot first.', 'error');
                return;
            }

            let notesHTML = '';
            if (_notes.length > 0) {
                notesHTML = '<h2 style="color:#374151;border-bottom:2px solid #6366f1;padding-bottom:8px;">Lab Notebook</h2>';
                for (let i = 0; i < _notes.length; i++) {
                    notesHTML += '<div style="margin:12px 0;padding:12px 16px;background:#f8f9fa;border-left:4px solid #6366f1;border-radius:0 8px 8px 0;">' +
                        '<div style="font-size:11px;color:#6b7280;margin-bottom:4px;">[' + (i+1) + '] ' + _notes[i].time + '</div>' +
                        '<div style="font-size:14px;color:#111827;white-space:pre-wrap;">' + _notes[i].text.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</div>' +
                        '</div>';
                }
            } else {
                notesHTML = '<p style="color:#6b7280;font-style:italic;">No notes recorded.</p>';
            }

            const html = '<!DOCTYPE html><html><head><meta charset="UTF-8">' +
                '<title>SPEC Plot + Notes - ' + filename + '</title>' +
                '<style>body{font-family:Arial,sans-serif;max-width:960px;margin:40px auto;padding:0 20px;color:#111827;}' +
                'h1{color:#374151;}img{width:100%;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.1);margin-bottom:24px;}</style>' +
                '</head><body>' +
                '<h1>&#128202; SPEC Analysis: ' + filename + '</h1>' +
                '<p style="color:#6b7280;font-size:13px;">Exported: ' + new Date().toLocaleString() + '</p>' +
                '<img src="' + plotImgSrc + '" alt="SPEC Plot">' +
                notesHTML +
                '</body></html>';

            const blob = new Blob([html], {type: 'text/html'});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'plot_notes_' + filename + '_' + new Date().toISOString().slice(0,10) + '.html';
            a.click(); URL.revokeObjectURL(a.href);
        }

        // ── Scan Table CSV Download ────────────────────────────────────────────
        function downloadScanTableCSV() {
            if (!_lastScanData || _lastScanData.length === 0) {
                showMessage('No scan table data. Load a SPEC file and open the Scan Table first.', 'error');
                return;
            }
            const headers = ['Scan #','Command','Timestamp','Temperature','Count Time','Data Points','Comments'];
            const rows = _lastScanData.map(function(s) {
                return [
                    s.scan_number,
                    '"' + String(s.command   || '').replace(/"/g,'""') + '"',
                    '"' + String(s.timestamp || '').replace(/"/g,'""') + '"',
                    s.temperature || '',
                    s.count_time  || '',
                    s.data_points || '',
                    '"' + String(s.comments  || '').replace(/"/g,'""') + '"'
                ];
            });
            const csv = [headers.join(',')].concat(rows.map(function(r) { return r.join(','); })).join('\\n');
            const blob = new Blob([csv], {type: 'text/csv'});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            const fname = exportInfo ? exportInfo.filename || 'spec' : 'spec';
            a.download = 'scan_table_' + fname + '_' + new Date().toISOString().slice(0,10) + '.csv';
            a.click(); URL.revokeObjectURL(a.href);
        }

        function toggleDark() {
            const isDark = document.body.classList.toggle('dark');
            const btn = document.getElementById('darkBtn');
            if (btn) btn.innerHTML = isDark ? '&#9728;&#65039; Light' : '&#127769; Dark';
        }

        window.onload = function() {
            showMessage('SPEC Overview ready! Shortcuts: Ctrl+O (browse), Ctrl+P (plot), Ctrl+T (table), Ctrl+E (export)', 'info');
        };
    </script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    print("Starting SPEC Overview...")
    print("Open: http://localhost:8000")
    print("Root: /nfs/chess/id4b/")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
