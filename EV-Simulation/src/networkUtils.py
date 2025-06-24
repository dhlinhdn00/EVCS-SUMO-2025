"""
EVCS-SUMO-2025
Simulation of Charging Needs of Electrical Car Drivers using SUMO

Author
------
DAO Hoai Linh  
LIS-LAB, Aix-Marseille University
"""
import requests
import json
import os
import subprocess
import re
import xml.etree.ElementTree as ET
from pyproj import Transformer, CRS
from shapely.geometry import LineString, MultiLineString, MultiPoint
from lxml import etree
from xml.sax.saxutils import escape
import time
import logging
from shapely.ops import unary_union, polygonize
from pathlib import Path
from typing import Dict
from config import NETCHECK_PY
import tempfile

_OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"

_POLYGON_QUERY_TEMPLATE = """
[out:json];
rel["admin_level"="8"]["name"~"^{commune}$", i];
out geom;
"""

_POPULATION_QUERY_TEMPLATE = """
[out:json];
rel["admin_level"="8"]["name"~"^{commune}$", i];
out tags;
"""


_MAIN_QUERY_TEMPLATE = """
[out:xml];
(
  way["highway"~"^living_street|motorway(_link)?|primary(_link)?|residential|secondary(_link)?|tertiary(_link)?|trunk(_link)?|unclassified$"]
  ["motor_vehicle"!~"no"]
  ({south},{west},{north},{east});
  >;
);
out meta;
"""

_DELAY_BETWEEN_REQUESTS = 1.2

_POLY_OUTPUT_TEMPLATE = "<polygons>\n{polys}</polygons>"
_POLY_ELEMENT_TEMPLATE = ('  <poly id="{id}" type="{region}" color="{color}" fill="1" '
                         'lineWidth="1" layer="0" shape="{shape}"/>\n')


_RE_WIDTH = re.compile(r'(\bwidth=")(\d+),(\d+)(")')
_RE_WIDTH_TEXT = re.compile(r'(\bwidth=")(narrow|medium|wide)(")')
_WIDTH_MAP = {"narrow": "2.5", "medium": "3.0", "wide": "4.0"}
_RE_LAYER = re.compile(r'(\blayer=")([^";|]+)([;|][^"]*)(")')

_RAW_OSM_NAME = "raw-osm"
_PROCESSED_OSM_NAME = "processed-osm"
_RAW_NETWORK_NAME = "raw-network"
_CONNECTED_NETWORK_NAME = "connected-network"
_POLY_NAME = "regions-based"


def mergeRegionsJSON(communes_path: str, id_path: str, color_path: str, output_path: str='combined_regions.json') -> None:
    """
    Merge communes, region IDs, and colors into a single JSON structure.

    Args:
        communes_path (str or Path): Path to communes.json
        id_path (str or Path): Path to id.json
        color_path (str or Path): Path to color.json
        output_path (str or Path): Output path for combined JSON file

    Output JSON structure:
    {
        "region_name": {
            "id": "region_id",
            "color": "r,g,b",
            "communes": [...]
        },
        ...
    }
    """
    with open(communes_path, 'r', encoding='utf-8') as f:
        communes_data = json.load(f)

    with open(id_path, 'r', encoding='utf-8') as f:
        id_data = json.load(f)

    with open(color_path, 'r', encoding='utf-8') as f:
        color_data = json.load(f)

    combined_data = {}
    for region, region_id in id_data.items():
        if region == "outside":
            combined_data[region] = {
                "id": region_id,
                "color": None,
                "communes": None,
            }
        else:
            combined_data[region] = {
                "id": region_id,
                "color": color_data.get(region, "0.5,0.5,0.5"),
                "communes": communes_data.get(region, [])
            }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print(f"Combined data saved to: {output_path}")

def fixOSMFile(infile: str, outfile: str) -> None:
    """
    Clean up an OSM/XML file for SUMO:
      - Normalize width attributes (1,6 → 1.6; narrow/medium/wide → 2.5/3.0/4.0)
      - Simplify layer="1;foo" → layer="1"
      - Assign priority="major/medium/minor" based on highway type if missing
    """
    def _assignPriority(line: str) -> str:
        if '<way' not in line:
            return line
        if 'type="' in line:
            if 'highway.primary' in line or 'highway.trunk' in line:
                tag = 'priority="major"'
            elif 'highway.secondary' in line or 'highway.tertiary' in line:
                tag = 'priority="medium"'
            else:
                tag = 'priority="minor"'
            # only inject if missing
            if 'priority=' not in line:
                line = re.sub(r'(<way\b[^>]*)(>)', rf'\1 {tag}\2', line)
        return line

    with open(infile, 'r', encoding='utf-8') as fin, \
         open(outfile, 'w', encoding='utf-8') as fout:
        for raw in fin:
            s = _RE_WIDTH.sub(lambda m: f'{m.group(1)}{m.group(2)}.{m.group(3)}{m.group(4)}', raw)
            s = _RE_WIDTH_TEXT.sub(lambda m: f'{m.group(1)}{_WIDTH_MAP[m.group(2)]}{m.group(3)}', s)
            s = _RE_LAYER.sub(lambda m: f'{m.group(1)}{m.group(2)}{m.group(4)}', s)
            fout.write(_assignPriority(s))

def _setupLogging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file,
        filemode="a",
        format="%(asctime)s %(levelname)s %(message)s"
    )
    return logging.getLogger(__name__)

def _sanitizeID(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^\w\-]", "_", s)
    return s

def generateNetwork( 
    base_folder: str,
    regions_json: str,
    buffer_degree: float = 0.001
):
    from pathlib import Path
    import requests, time, subprocess, json, xml.etree.ElementTree as ET
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import polygonize, unary_union
    from pyproj import Transformer, CRS
    from html import escape

    base = Path(base_folder)
    base.mkdir(parents=True, exist_ok=True)
    (base / "OSMs").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)

    raw_osm = base / "OSMs" / f"{_RAW_OSM_NAME}.osm"
    processed_osm = base / "OSMs" / f"{_PROCESSED_OSM_NAME}.osm"
    net_xml = base / f"{_RAW_NETWORK_NAME}.net.xml"
    poly_xml = base / f"{_POLY_NAME}.poly.xml"
    log_file = base / "logs" / "generate_network.log"

    logger = _setupLogging(log_file)
    logger.info("Pipeline start for %s", base.name)

    with open(regions_json, 'r', encoding='utf-8') as f:
        regions_data = json.load(f)

    region_bboxes = {}
    population_data = {}

    for region, info in regions_data.items():
        communes = info.get('communes')
        if not communes:
            continue

        s, w, n, e = 90.0, 180.0, -90.0, -180.0
        for commune in communes:
            logger.info("Fetching bbox and population for %s (%s)", commune, region)
            try:
                r = requests.get(_OVERPASS_API_URL, params={'data': _POLYGON_QUERY_TEMPLATE.format(commune=commune)}, timeout=60)
                r.raise_for_status()
                elems = r.json().get('elements', [])
                rel = next((el for el in elems if el['type'] == 'relation'), None)
                if rel and 'bounds' in rel:
                    b = rel['bounds']
                    s, w, n, e = min(s,b['minlat']), min(w,b['minlon']), max(n,b['maxlat']), max(e,b['maxlon'])

                r_pop = requests.get(_OVERPASS_API_URL, params={'data': _POPULATION_QUERY_TEMPLATE.format(commune=commune)}, timeout=60)
                r_pop.raise_for_status()
                elems_pop = r_pop.json().get("elements", [])
                if elems_pop:
                    tags = elems_pop[0].get("tags", {})
                    population = tags.get("population")
                    if population:
                        try:
                            population_data[commune] = int(population)
                        except ValueError:
                            logger.warning("Population value for %s is not integer: %s", commune, population)
                    else:
                        logger.warning("No population tag for %s", commune)
            except Exception as ex:
                logger.error("Error fetching data for %s: %s", commune, ex)
            time.sleep(_DELAY_BETWEEN_REQUESTS)

        region_bboxes[region] = {'south': s, 'west': w, 'north': n, 'east': e}
        logger.info("Region bbox %s: %s", region, region_bboxes[region])

    gs = min(b['south'] for b in region_bboxes.values()) - buffer_degree
    gw = min(b['west']  for b in region_bboxes.values()) - buffer_degree
    gn = max(b['north'] for b in region_bboxes.values()) + buffer_degree
    ge = max(b['east']  for b in region_bboxes.values()) + buffer_degree
    logger.info("Global bbox buffered: %s,%s,%s,%s", gs, gw, gn, ge)

    logger.info("Fetching highways and running netconvert")
    hq = _MAIN_QUERY_TEMPLATE.format(south=gs, west=gw, north=gn, east=ge)
    r = requests.get(_OVERPASS_API_URL, params={'data': hq}); r.raise_for_status()
    raw_osm.write_text(r.text, encoding='utf-8')
    logger.info("Saved raw OSM to %s", raw_osm)

    fixOSMFile(str(raw_osm), str(processed_osm))
    cmd = [
        'netconvert', '--osm-files', str(processed_osm),
        '-o', str(net_xml), '--output.street-names', 'true',
        '--output.original-names', 'true', '--offset.disable-normalization'
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    logger.info("netconvert stdout:\n%s", proc.stdout)
    if proc.stderr:
        logger.warning("netconvert stderr:\n%s", proc.stderr)
    proc.check_returncode()
    logger.info("Network generated: %s", net_xml)

    tree = ET.parse(str(net_xml))
    root = tree.getroot()
    proj = root.find('location').attrib.get('projParameter')
    transformer = Transformer.from_crs('epsg:4326', CRS.from_proj4(proj), always_xy=True)

    poly_str = ''
    for region, info in regions_data.items():
        communes = info.get('communes')
        color = info.get('color') or '0.8,0.8,0.8'
        if not communes:
            continue
        for commune in communes:
            logger.info("Building polygon for %s", commune)
            try:
                r = requests.get(_OVERPASS_API_URL, params={'data': _POLYGON_QUERY_TEMPLATE.format(commune=commune)}, timeout=60)
                r.raise_for_status()
                elems = r.json()['elements']
                nodes = {el['id']: el for el in elems if el['type'] == 'node'}
                ways = {el['id']: el for el in elems if el['type'] == 'way'}
                rels = [el for el in elems if el['type'] == 'relation']
                if not rels:
                    continue
                rel = rels[0]
                lines = []
                part_shapes = []
                for m in rel['members']:
                    if m['type'] == 'way':
                        way = ways.get(m['ref'])
                        coords = m.get('geometry') or []
                        if not coords and way:
                            coords = [{'lon': nodes[n]['lon'], 'lat': nodes[n]['lat']} for n in way.get('nodes', []) if n in nodes]
                        pts = [transformer.transform(pt['lon'], pt['lat']) for pt in coords]
                        if len(pts) > 1:
                            lines.append(LineString(pts))
                merged = unary_union(MultiLineString(lines))
                polygons = list(polygonize(merged))
                num_parts = len(polygons)
                total_pop = population_data.get(commune)
                parts = {}
                for i, poly in enumerate(polygons):
                    shape = ' '.join(f"{x:.2f},{y:.2f}" for x, y in poly.exterior.coords)
                    pid = _sanitizeID(f"{commune}_{i}")
                    poly_str += _POLY_ELEMENT_TEMPLATE.format(
                        id=pid, region=escape(region), color=color, shape=shape
                    )
                    if total_pop:
                        base = total_pop // num_parts
                        extra = 1 if i < (total_pop % num_parts) else 0
                        parts[pid] = base + extra
                if total_pop:
                    population_data[commune] = {
                        "population_total": total_pop,
                        "parts": parts
                    }
            except Exception as ex:
                logger.error("Polygon error %s: %s", commune, ex)
            time.sleep(_DELAY_BETWEEN_REQUESTS)

    Path(poly_xml).write_text(_POLY_OUTPUT_TEMPLATE.format(polys=poly_str), encoding='utf-8')
    logger.info("Polygon file written: %s", poly_xml)

    pop_out = base / "AMP_population.json"
    with open(pop_out, "w", encoding="utf-8") as f:
        json.dump(population_data, f, indent=2, ensure_ascii=False)
    logger.info("Population data written to: %s", pop_out)

def splitNetwork(
    base_folder: str,
    sub_folder_name: str,
    region_ids: list[str]
) -> None:
    """Extract a regional sub‑network from a full SUMO network.

    Parameters
    ----------
    base_folder : str
        Directory that already contains the *full* network
        ``<base>.net.xml`` and matching polygon file ``<base>.poly.xml``.
    sub_folder_name : str
        Name of the sub‑directory to create under *base_folder* where the
        trimmed network and filtered polygons will be written.
    region_ids : list[str]
        Values that must match the ``type="…"`` attribute of <poly> elements in
        the full polygon file.  Only those polygons are kept; edges whose
        *centroid* lies inside the *union bounding box* of these polygons are
        retained in the sub‑network.
    """

    base = Path(base_folder)
    if not base.is_dir():
        raise FileNotFoundError(f"base_folder not found or not a directory: {base_folder}")

    connected_net_xml   = base / f"{_CONNECTED_NETWORK_NAME}.net.xml"
    full_poly_xml  = base / f"{_POLY_NAME}.poly.xml"
    if not connected_net_xml.exists():
        raise FileNotFoundError(f"Connected network file missing: {connected_net_xml}")
    if not full_poly_xml.exists():
        raise FileNotFoundError(f"Full polygon file missing: {full_poly_xml}")

    # Create output sub‑folder
    out_dir = base / "subnetworks" / sub_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)  

    log_file = out_dir / "logs" / f"split_network.log"
    logger   = _setupLogging(log_file)
    logger.info("splitNetwork started | regions=%s", region_ids)


    tree = etree.parse(str(full_poly_xml))
    root = tree.getroot()               # <polygons>

    new_root = etree.Element("polygons")
    union_minx = union_miny = float("inf")
    union_maxx = union_maxy = float("-inf")

    for poly in root.findall("poly"):
        if poly.get("type") not in region_ids:
            continue
        new_root.append(poly)           # keep in sub‑poly file

        shape = poly.get("shape", "").strip()
        if not shape:
            continue
        # "x0,y0 x1,y1 …" -> list[(float,float)]
        try:
            coords = [(float(x), float(y)) for x, y in (p.split(",") for p in shape.split())]
        except ValueError:
            logger.warning("Invalid shape coords in poly id=%s", poly.get("id"))
            continue

        xs, ys   = zip(*coords)
        union_minx = min(union_minx, min(xs))
        union_miny = min(union_miny, min(ys))
        union_maxx = max(union_maxx, max(xs))
        union_maxy = max(union_maxy, max(ys))

    if union_minx == float("inf"):
        raise RuntimeError("No matching polygons found – check region_ids.")

    # Write filtered polygon file
    joined_ids   = "_".join(region_ids)
    sub_poly_xml = out_dir / f"{joined_ids}.poly.xml"
    sub_poly_xml.write_bytes(
        etree.tostring(new_root, pretty_print=True, xml_declaration=True, encoding="utf-8")
    )
    logger.info("Filtered polygon file written: %s", sub_poly_xml)


    bbox_str = f"{union_minx:.2f},{union_miny:.2f},{union_maxx:.2f},{union_maxy:.2f}"


    sub_net_xml = out_dir / f"connected_{joined_ids}.net.xml"

    cmd = [
        "netconvert",
        "--net-file", str(connected_net_xml),
        "--output-file", str(sub_net_xml),
        "--offset.disable-normalization",
        "--keep-edges.in-boundary", bbox_str,
    ]
    logger.info("Running netconvert: %s", " ".join(cmd))

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    logger.info("netconvert stdout:\n%s", proc.stdout)
    if proc.stderr:
        logger.warning("netconvert stderr:\n%s", proc.stderr)

    try:
        proc.check_returncode()
    except subprocess.CalledProcessError as e:
        logger.error("netconvert failed (code %d)", e.returncode)
        raise

    logger.info("Sub‑network generated: %s", sub_net_xml)
    print(f"Sub‑network saved to: {sub_net_xml}")
    print(f"Filtered polygons  : {sub_poly_xml}")

def showNetworkStats(net_xml: str | Path, v: float = 20) -> Dict[str, float]:
    """
    Quick statistics for a SUMO *.net.xml* file.

    Note that v is velocity use to compute the maximum number of vehicles 
    that appear on the map, according to the GreenShield formula (km/h)

    Returns
    -------
    dict
        {
            "junctions"       : int,
            "signal_junctions": int,
            "edges"           : int,
            "lanes"           : int,
            "components"      : int,   # 1 = connected
            "coverage_area"   : float, # m^2 (convex hull of junctions)
            "total_length"    : float  # m 
            "max_vehicles"    : int    # vehicles
        }
    """
    net_xml = Path(net_xml)
    if not net_xml.exists():
        raise FileNotFoundError(net_xml)

    root = ET.parse(net_xml).getroot()

    junctions = root.findall("junction")
    num_junctions = len(junctions)
    num_signal    = sum(j.attrib.get("type") == "traffic_light" for j in junctions)

    edges = [
        e for e in root.findall("edge")
        if e.attrib.get("function", "normal") != "internal"
    ]
    num_edges = len(edges)
    num_lanes = sum(len(e.findall("lane")) for e in edges)


    with tempfile.NamedTemporaryFile(prefix="netcheck_", suffix=".txt", delete=False) as tmp:
        comp_file = tmp.name

    try:
        cmd = [
            "python", NETCHECK_PY,
            str(net_xml),
            "--vclass", "passenger",
            "--component-output", comp_file,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if proc.returncode != 0:
            components = 0  # failed
        else:
            components = 0
            seen_edge  = False
            with open(comp_file, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()

                    if line.startswith("#"):
                        continue  # skip header

                    if line == "":
                        if seen_edge:
                            components += 1
                            seen_edge = False
                        continue

                    seen_edge = True

            if seen_edge:
                components += 1

    finally:
        try:
            os.remove(comp_file)
        except OSError:
            pass


    pts = [
        (float(j.attrib["x"]), float(j.attrib["y"]))
        for j in junctions if "x" in j.attrib and "y" in j.attrib
    ]
    coverage = MultiPoint(pts).convex_hull.area if len(pts) >= 3 else 0.0

    total_edges_length = 0.0
    for e in edges:
        first_lane = e.find("lane")
        if first_lane is not None and "length" in first_lane.attrib:
            try:
                total_edges_length += float(first_lane.attrib["length"])
            except ValueError:
                pass

    # GreenShield Equation
    total_lanes_length = (num_lanes / num_edges) * total_edges_length
    VF = 50 # Free Velocity (km/h)
    KJ = 100 # Jam Density (veh/km). Assume that mean lenght of car is abt 5m.
    
    k = int(((1 - v/VF) * KJ) * (total_lanes_length / 1000))

    return {
        "junctions"       : num_junctions,
        "signal_junctions": num_signal,
        "edges"           : num_edges,
        "lanes"           : num_lanes,
        "components"      : components,
        "coverage_area"   : coverage,
        "total_length"    : total_edges_length,
        "max_vehicles"    : k
    }

def extractMaxComponent(input_file: str, output_file: str) -> None:
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

    print(f"Found component with {max_count} edges, written to {output_file}")

def makeNetworkConnected(base_folder: str) -> Path:
    """
    Given a directory containing a SUMO network XML (<base_name>.net.xml),
    perform the following pipeline to keep only the largest connected component
    (for vclass="passenger"):

      A) Run netcheck on the original network to compute all components.
         Output is logged; extractMaxComponent writes the edges of the largest component
         into a KEEP_EDGES text file.

      1) Run netconvert (P2_1) on the original network with:
           --keep-edges.input-file <KEEP_EDGES>
           --keep-edges.by-vclass=passenger
           --remove-edges.isolated
         This produces a cleaned intermediate network.

      2) Run netconvert again (P2_2) on the cleaned network with:
           --keep-edges.input-file <KEEP_EDGES>
         to ensure only edges belonging to the largest component remain.

    The final connected network is written as:
        base_folder/connected_<base_name>.net.xml

    All intermediate logs and keep-edges/components files live under:
        base_folder/logs/

    Returns
    -------
    Path
        Path to the newly created connected_<base_name>.net.xml
    """
    base = Path(base_folder)
    if not base.is_dir():
        raise FileNotFoundError(f"{base_folder} is not a valid directory")

    original_net = base / f"{_RAW_NETWORK_NAME}.net.xml"
    if not original_net.exists():
        raise FileNotFoundError(f"Network file not found: {original_net}")

    # Create a logs subdirectory
    logs_dir = base / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path         = logs_dir / f"make_network_connected.log"
    netcheck_log1    = logs_dir / f"raw-network_netcheck.log"
    keep_edges_txt1  = logs_dir / "keep-edges.txt"
    components_txt1  = logs_dir / "components.txt"
    temp_net = base / f"temp-network.net.xml"
    final_net        = base / f"{_CONNECTED_NETWORK_NAME}.net.xml"

    logger = _setupLogging(log_path)
    logger.info("=== Starting makeNetworkConnected for %s ===", _RAW_NETWORK_NAME)

    logger.info("Step A: netcheck on original network to compute connected components")
    components_txt1.write_text("", encoding="utf-8")

    netcheck_cmd_1 = [
        "python", str(NETCHECK_PY),
        str(original_net),
        "--vclass", "passenger",
        "--component-output", str(components_txt1)
    ]
    logger.info("Running: %s", " ".join(map(str, netcheck_cmd_1)))
    with open(netcheck_log1, "w", encoding="utf-8") as f:
        subprocess.run(
            list(map(str, netcheck_cmd_1)),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True
        )
    logger.info("[DONE] Components output written to %s", components_txt1)
    logger.info("[LOG] netcheck output logged in %s", netcheck_log1)

    logger.info("Step B: extractMaxComponent on %s → writing keepEdges to %s", components_txt1, keep_edges_txt1)
    try:
        extractMaxComponent(str(components_txt1), str(keep_edges_txt1))
        logger.info("extractMaxComponent succeeded: kept edges in %s", keep_edges_txt1)
    except Exception as e:
        logger.error("extractMaxComponent failed: %s", e)
        raise

    logger.info("Step 1: netconvert (P2_1) → keep vclass=passenger & remove isolated")
    netconvert_cmd_1 = [
        "netconvert",
        "--net-file", str(original_net),
        "--keep-edges.input-file", str(keep_edges_txt1),
        "--keep-edges.by-vclass=passenger",
        "--remove-edges.isolated",
        "-o", str(temp_net)
    ]
    logger.info("Running: %s", " ".join(map(str, netconvert_cmd_1)))
    proc1 = subprocess.run(
        list(map(str, netconvert_cmd_1)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    logger.info("netconvert 1 stdout:\n%s", proc1.stdout)
    if proc1.stderr:
        logger.warning("netconvert 1 stderr:\n%s", proc1.stderr)
    try:
        proc1.check_returncode()
    except subprocess.CalledProcessError as e:
        logger.error("Step 1 netconvert failed (code %d)", e.returncode)
        raise

    logger.info("Step 2: netconvert 2 → keep only largest component edges")
    netconvert_cmd_2 = [
        "netconvert",
        "--net-file", str(temp_net),
        "--keep-edges.input-file", str(keep_edges_txt1),
        "-o", str(final_net),
        "--offset.disable-normalization"
    ]
    logger.info("Running: %s", " ".join(map(str, netconvert_cmd_2)))
    proc2 = subprocess.run(
        list(map(str, netconvert_cmd_2)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    logger.info("netconvert 2 stdout:\n%s", proc2.stdout)
    if proc2.stderr:
        logger.warning("netconvert 2 stderr:\n%s", proc2.stderr)
    try:
        proc2.check_returncode()
    except subprocess.CalledProcessError as e:
        logger.error("Step 2 netconvert failed (code %d)", e.returncode)
        raise

    try:
        temp_net.unlink()
        logger.info("Deleted temporary intermediate file: %s", temp_net)
    except Exception as e:
        logger.warning("Could not delete temp_net %s: %s", temp_net, e)

    logger.info("=== Completed makeNetworkConnected: final network = %s ===", final_net)
    print(f"Connected network written to: {final_net}")
    return final_net