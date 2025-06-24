from __future__ import annotations

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, List, Union
import xml.etree.ElementTree as ET
import traci
from pathlib import Path
import random
import glob
import gzip
import json
from config import TLSCOORDINATOR_PY, TLSCYCLEADAPTATION_PY, CONTINUOSREROUTER_PY

def outputs_complete(tag: str, outputs_dir: str) -> bool:
    outputs_dir = Path(outputs_dir)
    outdir = outputs_dir / f"hour_{tag}"
    try:
        content = (outdir / "statistics.xml").read_text(encoding="utf-8")
    except Exception:
        return False
    remainder = content.split("-->", maxsplit=1)[-1]
    return "<performance" in remainder


def state_path(end_time: int, state_dir: str) -> Path:
    state_dir = Path(state_dir)
    return state_dir / f"state_{end_time:08d}.xml.gz"


def merge_tls_cycles(hour: int, tls_dir: str) -> Path:
    tls_dir = Path(tls_dir)
    merged_path = tls_dir / f"tls_cycle_merged_{hour:02d}.add.xml"
    if merged_path.exists() and merged_path.stat().st_size > 0:
        return merged_path
    tl_dict: Dict[str, ET.Element] = {}
    for j in range(hour + 1):
        cycle_file = tls_dir / f"tls_cycle_{j:02d}.add.xml"
        if not cycle_file.exists():
            continue
        try:
            tree = ET.parse(cycle_file)
        except Exception:
            continue
        root = tree.getroot()
        for tl in root.findall(".//tlLogic"):
            tl_id = tl.attrib.get("id")
            if tl_id:
                tl_dict[tl_id] = tl
    additional = ET.Element("additional")
    for tl in tl_dict.values():
        additional.append(ET.fromstring(ET.tostring(tl)))
    tree = ET.ElementTree(additional)
    tree.write(merged_path, encoding="utf-8", xml_declaration=True)
    return merged_path


def tls_files_up_to(hour: int, tls_dir: str) -> str:
    tls_dir = Path(tls_dir)
    merged_cycle = merge_tls_cycles(hour, str(tls_dir))
    coord_files: List[str] = [
        str(tls_dir / f"tls_coord_{j:02d}.add.xml")
        for j in range(hour + 1)
        if (tls_dir / f"tls_coord_{j:02d}.add.xml").exists()
    ]
    return ",".join([str(merged_cycle), *coord_files])


def od2trips_for_1day(
    taz_file: str,
    trips_dir: str,
    od_dir: str,
    dist_id: Optional[str] = None,
    seed: Optional[int] = None,
    scale: Optional[float] = None
) -> List[str]:
    os.makedirs(trips_dir, exist_ok=True)
    trips: List[str] = []

    for h in range(24):
        od_file = os.path.join(od_dir, f"od_matrix_{h:02d}.txt")
        out_trips = os.path.join(trips_dir, f"trips_{h:02d}.xml")

        cmd = [
            "od2trips",
            "--taz-files", taz_file,
            "--od-matrix-files", od_file,
            "--prefix", f"h{h:02d}_",
            "--begin", str(h * 3600),
            "--end", str((h + 1) * 3600),
            "--spread.uniform", "true",
            "--different-source-sink", "true",
            "--departlane", "best",
            "--departpos", "random_free",
            "--departspeed", "max",
        ]

        if scale is not None:
            cmd.extend(["--scale", str(scale)])

        if dist_id is not None:
            cmd.extend(["--vtype", dist_id])

        if seed is not None:
            hour_seed = seed + h
            cmd.extend(["--seed", str(hour_seed)])
        else:
            cmd.append("--random")

        cmd.extend(["-o", out_trips])

        subprocess.run(cmd, check=True)

        trips.append(out_trips)
        print(f"Generated {out_trips}")

    return trips


def classify_vtypes(vtypes: list, ev_brands: list):
    cars, evs = [], []
    for v in vtypes:
        vid = v.get("id", "")
        if any(p.lower() in vid.lower() for p in ev_brands):
            evs.append(v)
        else:
            cars.append(v)
    return cars, evs


def assign_probabilities_to_vtypes(vtypes_xml: str, dist_id: str, ev_brands: list,
                                   ev_ratio: float, output_xml: str):

    tree = ET.parse(vtypes_xml)
    root = tree.getroot()
    car_dist = root.find(f".//vTypeDistribution[@id='{dist_id}']")
    if car_dist is None:
        raise ValueError(f"vTypeDistribution not found: id='{dist_id}'")

    vtypes = car_dist.findall("vType")
    cars, evs = classify_vtypes(vtypes, ev_brands)

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


def create_trips_for_1day(
    trips_dir: str,
    taz_xml:   str,
    ods_dir:   str,
    dist_id:   str = "vehDist",
    seed:      int = 42,
    scale:     int = 1
) -> List[Path]:

    trips_dir = Path(trips_dir)
    trips_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = trips_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    trip_gen_log = logs_dir / "trip_gen.log"

    existing_trips = sorted([f for f in trips_dir.glob("*.xml") if f.is_file()])

    if len(existing_trips) < 24:
        with open(trip_gen_log, "a", encoding="utf-8") as log_f:
            log_f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Starting trip generation for 24 hours...\n")
        try:
            start_time = time.time()
            generated = od2trips_for_1day(
                taz_file=str(taz_xml),
                trips_dir=str(trips_dir),
                od_dir=str(ods_dir),
                dist_id=dist_id,
                seed=seed,
                scale=scale
            )
            end_time = time.time()
            runtime = end_time - start_time


            generated_paths = sorted(Path(p) for p in generated)

            with open(trip_gen_log, "a", encoding="utf-8") as log_f:
                log_f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Generated {len(generated_paths)} trip files in {runtime:.2f}s\n\n")
        except Exception as e:
            with open(trip_gen_log, "a", encoding="utf-8") as log_f:
                log_f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Exception in od2trips_for_1day: {e}\n")
                log_f.write("(-----FAILED trip generation-----)\n\n")
            sys.exit(f"[ERROR] od2trips_for_1day raised exception: {e}")

        existing_trips = generated_paths

    if len(existing_trips) != 24:
        with open(trip_gen_log, "a", encoding="utf-8") as log_f:
            log_f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Expected 24 trip files, but found {len(existing_trips)}\n")
            log_f.write("(-----ABORT-----)\n\n")
        sys.exit(f"[ERROR] There must be exactly 24 trip files (found: {len(existing_trips)})")

    return existing_trips

def create_routes_for_1day(
    routes_dir:  Union[str, Path],
    trips_dir:   Union[str, Path],
    net_xml:     Union[str, Path],
    vtypes_xml:  Union[str, Path],
    seed:        int = None,
    threads:     int = 4
) -> List[Path]:
    trips_dir   = Path(trips_dir)
    routes_dir  = Path(routes_dir)
    net_xml     = Path(net_xml)
    vtypes_xml  = Path(vtypes_xml)

    routes_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = routes_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_trips = sorted([f for f in trips_dir.glob("trips_*.xml") if f.is_file()])
    if len(all_trips) != 24:
        sys.exit(f"[ERROR] In {trips_dir}, There must be exactly 24 trip files trips_XX.xml (found: {len(all_trips)})")

    routes: List[Path] = []

    for hour_idx in range(24):
        tag       = f"{hour_idx:02d}"
        trips_xml = trips_dir / f"trips_{tag}.xml"
        route_xml = routes_dir / f"route_{tag}.xml"
        duarouter_log = logs_dir / f"duarouter_{tag}.log"

        if not route_xml.exists() or route_xml.stat().st_size == 0:
            with open(duarouter_log, "a", encoding="utf-8") as log_f:
                log_f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [H{tag}] duarouter start ...\n")

                start_time = time.time()
                cmd = [
                    "duarouter",
                    "-n", str(net_xml),
                    "-r", str(trips_xml),
                    "-a", str(vtypes_xml),
                    "-o", str(route_xml),
                    "--log", str(duarouter_log),
                    "--exit-times",
                    "--named-routes",
                    "--route-length",
                    "--write-costs",
                    "--routing-threads", str(threads),
                    "--ignore-errors",
                ]
                if seed is not None:
                    cmd += ["--seed", str(seed)]

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.stdout:
                    log_f.write(proc.stdout)
                if proc.stderr:
                    log_f.write(proc.stderr)

                if proc.returncode != 0:
                    log_f.write(f"[ERROR] duarouter failed at hour {tag} (returncode={proc.returncode})\n")
                    log_f.write("(-----SKIPPED duarouter-----)\n\n")
                    sys.exit(f"[ERROR] duarouter failed at hour {tag} (check {duarouter_log})")
                else:
                    runtime = time.time() - start_time
                    log_f.write(f"(-----FINISHED----- duarouter in {runtime:.2f}s)\n\n")

        else:
            with open(duarouter_log, "a", encoding="utf-8") as log_f:
                log_f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [H{tag}] route already exists – skipped duarouter\n")
                log_f.write("(-----SKIPPED duarouter-----)\n\n")

        routes.append(route_xml)

    return routes

def merge_trips(trips_dir: str, output_file: str):
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

def create_tls_for_1day(
    routes_dir: Union[str, Path],
    tls_dir:    Union[str, Path],
    net_xml:    Union[str, Path],
    program_id: str = "a"
) -> None:
    routes_dir = Path(routes_dir)
    tls_dir    = Path(tls_dir)
    net_xml    = Path(net_xml)

    logs_dir = tls_dir / "logs"
    tls_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    route_files = sorted([
        f for f in routes_dir.glob("route_??.xml")
        if f.is_file() and not f.name.endswith(".alt.xml")
    ])
    if len(route_files) != 24:
        sys.exit(f"[ERROR] Trong {routes_dir}, cần đúng 24 file route_XX.xml (hiện có: {len(route_files)})")

    for hour_idx in range(24):
        tag       = f"{hour_idx:02d}"
        route_xml = routes_dir / f"route_{tag}.xml"
        tls_cycle = tls_dir / f"tls_cycle_{tag}.add.xml"
        tls_coord = tls_dir / f"tls_coord_{tag}.add.xml"
        adapt_log = logs_dir / f"tls_adapt_{tag}.log"
        coord_log = logs_dir / f"tls_coord_{tag}.log"

        if not tls_cycle.exists() or tls_cycle.stat().st_size == 0:
            with open(adapt_log, "a", encoding="utf-8") as log_f:
                log_f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [H{tag}] tlsCycleAdaptation start ...\n")

                start_time = time.time()
                cmd = [
                    "python3", TLSCYCLEADAPTATION_PY,
                    "-n", str(net_xml),
                    "-r", str(route_xml),
                    "-b", str(hour_idx * 3600),
                    "-o", str(tls_cycle),
                    "--min-cycle", "40",
                    "--max-cycle", "120",
                    "--yellow-time", "3",
                    "-p", program_id,
                    "--verbose"
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.stdout:
                    log_f.write(proc.stdout)
                if proc.stderr:
                    log_f.write(proc.stderr)

                if proc.returncode != 0:
                    log_f.write(f"[ERROR] tlsCycleAdaptation failed at hour {tag} (returncode={proc.returncode})\n")
                    log_f.write("(-----SKIPPED tlsCycle-----)\n\n")
                    continue

                runtime = time.time() - start_time
                log_f.write(f"(-----FINISHED----- tlsCycle in {runtime:.2f}s)\n\n")
        else:
            with open(adapt_log, "a", encoding="utf-8") as log_f:
                log_f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [H{tag}] tls_cycle already exists – skipped\n")
                log_f.write("(-----SKIPPED tlsCycle-----)\n\n")


        if not tls_coord.exists() or tls_coord.stat().st_size == 0:
            with open(coord_log, "a", encoding="utf-8") as log_f:
                log_f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [H{tag}] tlsCoordinator start ...\n")

                start_time = time.time()
                cmd = [
                    "python3", TLSCOORDINATOR_PY,
                    "-n", str(net_xml),
                    "-r", str(route_xml),
                    "-a", str(tls_cycle),
                    "-o", str(tls_coord),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.stdout:
                    log_f.write(proc.stdout)
                if proc.stderr:
                    log_f.write(proc.stderr)

                if proc.returncode != 0:
                    log_f.write(f"[ERROR] tlsCoordinator failed at hour {tag} (returncode={proc.returncode})\n")
                    log_f.write("(-----SKIPPED tlsCoord-----)\n\n")
                    continue

                runtime = time.time() - start_time
                log_f.write(f"(-----FINISHED----- tlsCoord in {runtime:.2f}s)\n\n")
        else:
            with open(coord_log, "a", encoding="utf-8") as log_f:
                log_f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [H{tag}] tls_coord already exists – skipped\n")
                log_f.write("(-----SKIPPED tlsCoord-----)\n\n")


def create_continuous_rerouter(
    net_xml: Union[str, Path],
    rerouter_xml: Union[str, Path],
    # 0h -> 6h (0 -> 21600) | 21h -> 24h (75600 -> 86400)
    begin: int = 0,   
    end: int = 21600    
) -> None:
    cmd = [
        "python3", CONTINUOSREROUTER_PY,
        "-n", str(net_xml),
        "-o", str(rerouter_xml),
        "-l", 
        "--vclass", "passenger",
        "-b", str(begin),
        "-e", str(end),
        "--turn-defaults", "30,40,30"
    ]
    print(f"[create_continuous_rerouter] Generating rerouter for {net_xml} -> {rerouter_xml}")
    subprocess.run(cmd, check=True)

def parse_congested_edges(edgedata_path: Path, threshold: float) -> list[str]:
    edge_to_occs: dict[str, list[float]] = {}

    tree = ET.parse(edgedata_path)
    root = tree.getroot()

    for edge_elem in root.findall(".//edge"):
        edge_id = edge_elem.get("id")
        if edge_id not in edge_to_occs:
            edge_to_occs[edge_id] = []

        for lane_elem in edge_elem.findall("lane"):
            occ_attr = lane_elem.get("occupancy")
            if occ_attr is None:
                continue
            try:
                occ_val = float(occ_attr)
            except ValueError:
                continue
            edge_to_occs[edge_id].append(occ_val)

    edge_avg: dict[str, float] = {}
    for eid, occ_list in edge_to_occs.items():
        if not occ_list:
            edge_avg[eid] = 0.0
        else:
            edge_avg[eid] = sum(occ_list) / len(occ_list)

    congested_edges = [eid for eid, avg in edge_avg.items() if avg > threshold]
    return congested_edges


def inspect_tls_phases(net_xml: str, tls_cycle_add: str = "", traci_port: int = 8873):

    sumo_cmd = ["sumo", "-n", net_xml]
    if tls_cycle_add:
        sumo_cmd += ["--additional-files", tls_cycle_add]

    traci.start(sumo_cmd, port=traci_port)

    tls_ids = traci.trafficlight.getIDList()
    print(f"Found {len(tls_ids)} traffic lights:\n")

    for tlsID in tls_ids:
        print(f"===== TLS ID = {tlsID} =====")
        controlled_links = traci.trafficlight.getControlledLinks(tlsID)
        for phase_idx, links in enumerate(controlled_links):
            in_lanes = [inLane for (inLane, outLane, viaIdx) in links]
            print(f"  Phase {phase_idx}: in-lanes = {in_lanes}")
        print()

    traci.close()

def create_routes_json(routes_xml: str, routes_json: str):
    routes_xml = Path(routes_xml)
    routes_json = Path(routes_json)
    seen = set()
    n_written = 0
    

    with gzip.open(routes_json, "wt", encoding="utf-8") as out:
        for ev, elem in ET.iterparse(routes_xml, events=("end",)):
            tag = elem.tag
            if tag == "route":
                rid   = elem.attrib.get("id")
                if rid in seen:         
                    elem.clear(); continue
                edges = elem.attrib["edges"].split()
                json.dump({"id": rid, "edges": edges}, out)
                out.write("\n")
                seen.add(rid)
                n_written += 1
            elif tag == "vehicle":     
                rid = elem.attrib.get("route")
                if not rid and "edges" in elem.attrib:
                    rid = elem.attrib.get("id")  
                    edges = elem.attrib["edges"].split()
                    if rid not in seen:
                        json.dump({"id": rid, "edges": edges}, out)
                        out.write("\n")
                        seen.add(rid); n_written += 1
            elem.clear()

    print(f"Wrote {n_written:,} unique routes to {routes_json} (gzip compressed)")