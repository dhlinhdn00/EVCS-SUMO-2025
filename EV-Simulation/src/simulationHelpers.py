"""
EVCS-SUMO-2025
Simulation of Charging Needs of Electrical Car Drivers using SUMO

Author
------
DAO Hoai Linh  
LIS-LAB, Aix-Marseille University
"""

import os
import xml.etree.ElementTree as ET
from lxml import etree
import subprocess
import traci
import sumolib
from sumolib import net
import random
import itertools
import pandas as pd
from collections import OrderedDict, defaultdict
import sys
import networkx as nx
import math
import csv
import shutil
from pyproj import Transformer, CRS
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.errors import TopologicalError
from pathlib import Path
import re
import glob
from config import RANDOMTRIPS_PY
import tempfile
from typing import Dict, List, Tuple, Optional, Set, Iterable


def extractMaxComponent(input_file, output_file):
    max_count = 0
    max_edges = []

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'(?=^Component:\s*#\d+\s+Edge\s+Count:)', content, flags=re.MULTILINE)

    for block in blocks:
        m = re.match(r'^Component:\s*#\d+\s+Edge\s+Count:\s*(\d+)\s*\n(.*)', block, flags=re.DOTALL)
        if not m:
            continue
        count = int(m.group(1))
        edges_part = m.group(2).strip()
        edges = re.split(r'\s+', edges_part)

        if count > max_count:
            max_count = count
            max_edges = edges

    with open(output_file, 'w', encoding='utf-8') as f:
        for e in max_edges:
            f.write(f"{e}\n")

    print(f"[DONE] Found component with {max_count} edges, written to {output_file}")

def parseShape(shape_str: str):
    """Parses a SUMO shape string into a Shapely Polygon."""
    try:
        coords = [tuple(map(float, p.split(','))) for p in shape_str.strip().split()]
    except ValueError:
        return None
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if poly.is_valid else None

def selectBoundaryEdges(edges, geom, ratio=0.1):
    """Select edges whose midpoint is close to the boundary of the given geometry."""

    if geom is None:
        return []
    minx, miny, maxx, maxy = geom.bounds
    th = min(maxx - minx, maxy - miny) * ratio
    boundary = geom.boundary
    return [e for e in edges if Point(e.getShape()[len(e.getShape())//2]).distance(boundary) < th]

def hasReverse(edge):
    fn, tn = edge.getFromNode(), edge.getToNode()
    return any(o.getToNode() is fn for o in tn.getOutgoing())

def isValidEdge(edge):
    return (not edge.getID().startswith('-')  # skip reverse direction
            and not edge.getID().endswith(('-source', '-sink'))
            and edge.getShape()
            and hasReverse(edge))            # true two‑way road

def reachable(edge_from, edge_to, net):
    return net.getShortestPath(edge_from, edge_to)[0] is not None

def filterReachable(pool, net, sample):
    """Keep edges that can reach **and** be reached from ≥1 peer in *pool*."""
    if len(pool) <= 1:
        return pool
    keep = []
    for e in pool:
        targets = random.sample([x for x in pool if x is not e], min(sample, len(pool)-1))
        ok_out = any(reachable(e, t, net) for t in targets)
        ok_in  = any(reachable(t, e, net) for t in targets)
        if ok_out and ok_in:
            keep.append(e)
    return keep

def filterSources(pool, net):
    return [e for e in pool 
        if any(reachable(e, o, net) for o in pool if o is not e)]

def filterSinks(pool, net):
    return [e for e in pool 
            if any(reachable(o, e, net) for o in pool if o is not e)]

def generateOds(csv_path: str, od_dir: str, region_ids: dict, real_origin: str = "marseille",
                 exclude_cols: set = None, trips_ratio: float = 1.0, 
                 scale_out: float = 1.0, scale_in: float = 0.0):
    """
    Generate OD matrix files from a CSV table.
    - Each row in CSV corresponds to an hour of day.
    - Output is one OD matrix file per hour.
    """
    if abs((scale_out + scale_in) - 1.0) > 1e-6:
        print("[WARNING] Sum of scale_out and scale_in is not equal to 1.0. Resetting to default: scale_out = 1.0; scale_in = 0.0")
        scale_out = 1.0
        scale_in = 0.0

    os.makedirs(od_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    if exclude_cols is None:
        exclude_cols = {'total', 'intra'}
    hours = df.iloc[:, 0].tolist()
    csv_zones = [c for c in df.columns[1:] if c not in exclude_cols]

    norm_region_ids = {k: v for k, v in region_ids.items()}
    generated = []

    for idx, hour in enumerate(hours):
        out_file = os.path.join(od_dir, f"od_matrix_{hour:02d}.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("$OR;D2\n")
            f.write(f"{int(hour)}.00 {int(hour) + 1}.00\n")
            f.write("1.00\n")
            row = df.iloc[idx]

            raw_intra = float(row.get("intra", 0.0))
            for orig, orig_id in norm_region_ids.items():
                for dest, dest_id in norm_region_ids.items():
                    val = 0.0
                    # Intra
                    if orig == dest == real_origin:
                        val = round(raw_intra * trips_ratio)
                    # Inter: OUT-going
                    elif orig == real_origin and dest in csv_zones:
                        raw = float(row.get(dest, 0.0))
                        val = round(raw * trips_ratio * scale_out)
                    # Inter: IN-coming
                    elif dest == real_origin and orig in csv_zones:
                        raw = float(row.get(orig, 0.0))
                        val = round(raw * trips_ratio * scale_in)

                    # if val: 
                    #     f.write(f"{int(orig_id):>11}{int(dest_id):>11}{val:>11.2f}\n")

                    f.write(f"{int(orig_id):>11}{int(dest_id):>11}{val:>11.2f}\n")
        generated.append((hour, out_file))

    return generated

def mergeTrips(trips_dir: str, output_file: str):
    """
    Merge all hourly trip files into a single file, sorted by 'depart' time.
    """
    files = sorted(glob.glob(os.path.join(trips_dir, "trips_??.xml")))
    if not files:
        raise FileNotFoundError(f"Not found trips_*.xml in {trips_dir}")

    tree_all = ET.parse(files[0])
    root_all = tree_all.getroot()

    for f in files[1:]:
        for elt in ET.parse(f).getroot():
            root_all.append(elt)

    root_all[:] = sorted(root_all, key=lambda e: float(e.get("depart", "0")))
    tree_all.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"[DONE] Merged {len(files)} trip files to {output_file}")

def classifyVtypes(vtypes: list, ev_brands: list):
    """Split vTypes into ICE and EV categories by brand patterns."""
    cars, evs = [], []
    for v in vtypes:
        vid = v.get("id", "")
        if any(p.lower() in vid.lower() for p in ev_brands):
            evs.append(v)
        else:
            cars.append(v)
    return cars, evs

def assignProbabilitiesToVtypes(vtypes_xml: str, dist_id: str, ev_brands: list,
                                   ev_ratio: float, output_xml: str):
    """
    Assign probability to each vType based on EV ratio within a vTypeDistribution.
    """
    tree = ET.parse(vtypes_xml)
    root = tree.getroot()
    car_dist = root.find(f".//vTypeDistribution[@id='{dist_id}']")
    if car_dist is None:
        raise ValueError(f"vTypeDistribution not found: id='{dist_id}'")

    vtypes = car_dist.findall("vType")
    cars, evs = classifyVtypes(vtypes, ev_brands)

    print(f"[DETECTED] There are {len(cars)} ICEs, {len(evs)} EVs")
    random.shuffle(vtypes)

    n_cars = len(cars)
    n_evs = len(evs)
    p_car = (1 - ev_ratio) / n_cars if n_cars else 0
    p_ev = ev_ratio / n_evs if n_evs else 0

    for v in vtypes:
        vid = v.get("id", "")
        p = p_ev if v in evs else p_car
        v.set("probability", f"{p:.6f}")

    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    print(f"[DONE] vType probabilities written to {output_xml}")

def updateSumoCfg(cfg_path: str, output_path: str, replacements: dict):
    parser = etree.XMLParser(remove_comments=False)
    tree = etree.parse(str(cfg_path), parser)
    root = tree.getroot()

    for elem in root.iter():
        if 'value' in elem.attrib:
            tag = etree.QName(elem).localname  
            if tag in replacements:
                new_value = str(replacements[tag])
                print(f"Updating {tag}: {elem.attrib['value']} -> {new_value}")
                elem.attrib['value'] = new_value

    tree.write(str(output_path), encoding='UTF-8', xml_declaration=True, pretty_print=True)

def downscaleTrips(src: Path, keep: float) -> Path:
    if keep >= 0.999:
        return src
    dst = src.with_suffix('.scaled.xml')
    with src.open() as fi, dst.open('w') as fo:
        for line in fi:
            if line.lstrip().startswith(('<trip', '<flow')) and random.random() > keep:
                continue
            fo.write(line)
    return dst

def od2tripsForAllOLD(taz_file: str, trips_dir: str, od_dir: str, dist_id: str=None):
    """
    Run od2trips on all 24 hourly OD matrix files.
    """
    os.makedirs(trips_dir, exist_ok=True)
    trips = []

    for h in range(0, 24):
        od_file = os.path.join(od_dir, f"od_matrix_{h:02d}.txt")
        out_trips = os.path.join(trips_dir, f"trips_{h:02d}.xml")

        cmd = [
            "od2trips",
            "--taz-files", taz_file,
            "--od-matrix-files", od_file,
            "--prefix", f"h{h:02d}_",
            "--begin", str(h * 3600),
            "--end",   str((h + 1) * 3600),
            "--random",
            "--spread.uniform", "true",
            "--different-source-sink", "true",     
            "--departlane", "best",    
            "--departpos",  "random_free",  
            "--departspeed","max",             
        ]
        if dist_id is not None:
            cmd.extend(["--vtype", dist_id])
        
        cmd.extend(["-o", out_trips])
        subprocess.run(cmd, check=True)
        trips.append(out_trips)
        print("Generated", out_trips)

    return trips

def countFilesInDir(directory):
    return len([
        name for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name))
    ])





def parse_taz_edges(taz_file: str) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = defaultdict(set)
    tree = ET.parse(taz_file)
    root = tree.getroot()
    for taz in root.iter("taz"):
        taz_id = taz.attrib["id"]
        for child in taz:
            if child.tag in ("tazSource", "tazSink"):
                mapping[taz_id].add(child.attrib["id"])
    return mapping

def read_od_matrix(fname: str) -> Iterable[Tuple[str, str, int]]:
    """Yield (O, D, count) from a SUMO OD‑matrix text file."""
    with open(fname) as f:
        for line in f:
            if line.lstrip().startswith("$") or line.strip() == "" or "*" in line:
                continue  # header / comment
            parts = line.split()
            if len(parts) != 3:
                continue
            yield parts[0], parts[1], int(float(parts[2]))

def write_od_file(entries: List[Tuple[str, str, int]], out_path: str, *, begin: int, end: int):
    with open(out_path, "w") as f:
        f.write("$OR;D2\n")
        f.write(f"{begin/3600:05.2f} {end/3600:05.2f}\n")
        f.write("1.00\n")
        for o, d, c in entries:
            f.write(f"{o}\t{d}\t{c}\n")


def write_weight_files(edges: Iterable[str], prefix: str):
    src = prefix + ".src.xml"
    dst = prefix + ".dst.xml"
    tmpl = """<edgedata>\n    <interval id=\"zone\" begin=\"0\" end=\"3600\">\n{body}    </interval>\n</edgedata>\n"""
    body = "".join([f"        <edge id=\"{e}\" value=\"1.0\"/>\n" for e in edges])
    xml = tmpl.format(body=body)
    for path in (src, dst):
        with open(path, "w") as f:
            f.write(xml)



def call_od2trips(taz_file: str, od_file: str, out_xml: str, hour: int, vtype: str | None):
    cmd = [
        "od2trips",
        "--taz-files", taz_file,
        "--od-matrix-files", od_file,
        "--prefix", f"h{hour:02d}_",
        "--begin", str(hour * 3600),
        "--end", str((hour + 1) * 3600),
        "--random",
        "--spread.uniform", "true",
        "--different-source-sink", "true",
        "--departlane", "best",
        "--departpos", "random_free",
        "--departspeed", "max",
        "-o", out_xml,
    ]
    if vtype:
        cmd += ["--vtype", vtype]
    subprocess.run(cmd, check=True)


def call_randomtrips(*, rand_script: str, net_file: str, out_trips: str, seed: int, min_d: int, max_d: int,
                     begin: int, end: int, period: float, dist_id: str | None, weights_prefix: str):
    rt_cmd = [
        "python3", rand_script,
        "-n", net_file,
        "-o", out_trips,
        "-t", f"type=\"{dist_id}\" departLane=\"best\" departSpeed=\"max\"" if dist_id else "",
        "-s", str(seed),
        "--min-distance", str(min_d), "--max-distance", str(max_d),
        "-b", str(begin), "-e", str(end),
        "--period", f"{period:.4f}", "--poisson",
        "--weights-prefix", weights_prefix,
    ]
    # cleanup empty strings
    rt_cmd = [x for x in rt_cmd if x]
    subprocess.run(rt_cmd, check=True)

################################################################################
# 5.  MAIN ENTRY
################################################################################

def _merge_xml(inputs: List[Path], output: Path):
    """Gộp các <routes> file vào *output* (ghi đè nếu có)."""
    if not inputs:
        return False
    tree_main = ET.parse(inputs[0])
    root_main = tree_main.getroot()
    for p in inputs[1:]:
        for elem in ET.parse(p).getroot():
            root_main.append(elem)
    tree_main.write(output, encoding="utf-8", xml_declaration=True)
    return True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def od2trips_mixed(
    *,
    taz_file: str,
    net_file: str,
    trips_dir: str,
    inter_dir: Optional[str] = None,
    intra_dir: Optional[str] = None,
    od_dir: Optional[str] = None,
    vtype: Optional[str] = None,
    seed: int = 42,
    min_dist: int = 5000,
    max_dist: int = 40000,
) -> List[str]:
    """Sinh **24** file `trips_{HH}.xml` (1 file / giờ).

    - Nếu *inter_dir* / *intra_dir* → dùng chúng.
    - Nếu chỉ có *od_dir*          → tự tách O≠D / O=D.

    Hàm tự dò 2 pattern file: `od_inter_XX.txt` và `od_matrix_XX.txt`
    (tương tự cho intra).
    """

    # --- Kiểm tra ---
    for p in (taz_file, net_file):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    trips_dir = Path(trips_dir)
    trips_dir.mkdir(parents=True, exist_ok=True)

    taz_edges = parse_taz_edges(taz_file)
    hourly_outputs: List[str] = []

    def _find_od(path_dir: str, patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            cand = Path(path_dir) / pat
            if cand.is_file():
                return str(cand)
        return None

    for h in range(24):
        hour_files: List[Path] = []

        # ---------------- Inter ----------------
        if inter_dir or od_dir:
            inter_path = None
            if inter_dir:
                inter_path = _find_od(
                    inter_dir,
                    [f"od_inter_{h:02d}.txt", f"od_matrix_{h:02d}.txt"]
                )
            elif od_dir:
                inter_path = Path(od_dir) / f"od_matrix_{h:02d}.txt"
                if inter_path.is_file():
                    inter_path = str(inter_path)
                else:
                    inter_path = None
            if inter_path:
                inter_entries = [e for e in read_od_matrix(inter_path) if e[0] != e[1]]
                if inter_entries:
                    tmp_od = trips_dir / f"od_inter_tmp_{h:02d}.txt"
                    write_od_file(inter_entries, tmp_od, begin=h*3600, end=(h+1)*3600)
                    out_xml = trips_dir / f"trips_inter_{h:02d}.xml"
                    call_od2trips(taz_file, str(tmp_od), str(out_xml), h, vtype)
                    hour_files.append(out_xml)

        # ---------------- Intra ----------------
        if intra_dir or od_dir:
            intra_path = None
            if intra_dir:
                intra_path = _find_od(
                    intra_dir,
                    [f"od_intra_{h:02d}.txt", f"od_matrix_{h:02d}.txt"]
                )
            elif od_dir:
                intra_path = Path(od_dir) / f"od_matrix_{h:02d}.txt"
                if intra_path.is_file():
                    intra_path = str(intra_path)
                else:
                    intra_path = None
            if intra_path:
                intra_counts: defaultdict[str, int] = defaultdict(int)
                for o, d, c in read_od_matrix(intra_path):
                    if o == d:
                        intra_counts[o] += c
                for taz_id, cnt in intra_counts.items():
                    if cnt <= 0:
                        continue
                    edges = taz_edges.get(taz_id)
                    if not edges:
                        continue
                    period = 3600.0 / cnt
                    weights_prefix = trips_dir / f"w_{taz_id}_{h:02d}"
                    write_weight_files(edges, str(weights_prefix))
                    intra_out = trips_dir / f"trips_intra_{taz_id}_{h:02d}.xml"
                    call_randomtrips(
                        rand_script=RANDOMTRIPS_PY,
                        net_file=net_file,
                        out_trips=str(intra_out),
                        seed=seed,
                        min_d=min_dist,
                        max_d=max_dist,
                        begin=h*3600,
                        end=(h+1)*3600,
                        period=period,
                        dist_id=vtype,
                        weights_prefix=str(weights_prefix),
                    )
                    hour_files.append(intra_out)

        # ---------------- Merge per‑hour ----------------
        merged_hour = trips_dir / f"trips_{h:02d}.xml"
        if hour_files:
            _merge_xml(hour_files, merged_hour)
        else:
            print(f"[WARN] no OD entries found for hour {h:02d}")
        hourly_outputs.append(str(merged_hour))

    return hourly_outputs


def parse_hhmm(time_str: str) -> int:
    """
    Chuyển chuỗi "H.MM" hoặc "H.MMSS" thành giây từ nửa đêm.
    Ví dụ:
      "7.00" -> 7*3600 + 0*60 = 25200
      "7.20" -> 7*3600 + 20*60 = 27600
      "7.5"  -> 7*3600 + 5*60  = 26100  (nếu chỉ có 1 chữ số phút)
      "7"    -> 7*3600         = 25200  (nếu không có dấu chấm)
    Nếu phần sau dấu chấm không phải số nguyên, fallback sang parse float giờ.
    """
    if "." in time_str:
        hh_part, mm_part = time_str.split(".", 1)
        try:
            hh = int(hh_part)
            mm = int(mm_part)
        except ValueError:
            # fallback: xem như số giờ thập phân
            return int(round(float(time_str) * 3600))
        return hh * 3600 + mm * 60
    else:
        try:
            hh = int(time_str)
            return hh * 3600
        except ValueError:
            return int(round(float(time_str) * 3600))


def od2tripsForSubs(
    taz_file: str,
    trips_dir: str,
    od_dir: str,
    dist_id: Optional[str] = None
) -> List[str]:
    """
    Chạy od2trips trên tất cả file OD trong od_dir, dùng header thời gian (dòng 2 của file OD),
    trong đó header dạng "H.MM H.MM" là giờ + phút (không phải số giờ thập phân).

    Output trips được lưu vào trips_dir với tên trips_HH_SofM.xml
    """
    os.makedirs(trips_dir, exist_ok=True)
    trips_files: List[str] = []

    # Regex để khớp tên file dạng "od_matrix_HH[_SofM].txt"
    pattern = re.compile(r"^od_matrix_(\d{2})(?:_(\d+)of(\d+))?\.txt$")

    for fname in sorted(os.listdir(od_dir)):
        m = pattern.match(fname)
        if not m:
            continue

        hour_str = m.group(1)          # "HH"
        sub_idx_str = m.group(2)       # nếu có, ví dụ "2"
        total_splits_str = m.group(3)  # nếu có, ví dụ "3"

        if sub_idx_str is None or total_splits_str is None:
            splits = 1
            sub_idx = 1
        else:
            splits = int(total_splits_str)
            sub_idx = int(sub_idx_str)

        od_file = os.path.join(od_dir, fname)

        # --- ĐỌC HEADER để lấy thời gian BEGIN/END chính xác ---
        with open(od_file, 'r') as f:
            lines = f.readlines()
        time_line = None
        count_non_comment = 0
        for line in lines:
            if line.strip().startswith('*'):
                continue
            count_non_comment += 1
            if count_non_comment == 2:
                time_line = line.strip()
                break
        if time_line is None:
            raise RuntimeError(f"Không tìm thấy dòng thời gian trong {od_file}")

        parts = time_line.split()
        if len(parts) < 2:
            raise RuntimeError(f"Không parse được time_line '{time_line}' trong {od_file}")

        # Dùng parse_hhmm để hiểu đúng phút
        begin_sec = parse_hhmm(parts[0])
        end_sec   = parse_hhmm(parts[1])

        # Tạo tên file trips output, ví dụ "trips_07_2of3.xml"
        trip_name = f"trips_{hour_str}_{sub_idx}of{splits}.xml"
        out_trips = os.path.join(trips_dir, trip_name)

        # Tạo prefix cho ID xe, ví dụ "h07_2of3_"
        prefix = f"h{hour_str}_{sub_idx}of{splits}_"

        # Xây dựng danh sách lệnh od2trips
        cmd = [
            "od2trips",
            "--taz-files", taz_file,
            "--od-matrix-files", od_file,
            "--prefix", prefix,
            "--begin", str(begin_sec),
            "--end",   str(end_sec),
            "--random",
            "--spread.uniform", "true",
            "--different-source-sink", "true",
            "--departlane", "best",
            "--departpos", "random_free",
            "--departspeed", "max",
        ]
        # Nếu người dùng muốn gán vType
        if dist_id is not None:
            cmd.extend(["--vtype", dist_id])

        # Cuối cùng thêm "-o output_file"
        cmd.extend(["-o", out_trips])

        # Chạy od2trips
        subprocess.run(cmd, check=True)
        trips_files.append(out_trips)
        print(f"[INFO] Generated {out_trips}  "
              f"(header: {parts[0]}–{parts[1]}, "
              f"begin_sec={begin_sec}, end_sec={end_sec}, "
              f"hour={hour_str}, split={sub_idx}/{splits})")

    return trips_files