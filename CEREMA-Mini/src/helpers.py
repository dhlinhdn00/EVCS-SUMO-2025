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

def od2tripsForAll(taz_file: str, trips_dir: str, od_dir: str, dist_id: str=None):
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

def updateSumoCfg(cfg_path, output_path, replacements):
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