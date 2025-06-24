from __future__ import annotations
from dataclasses import dataclass
import signal, sys, typing as t
from pathlib import Path
import traci, traci.exceptions as tc
import shutil, datetime as dt
import json
import gzip
from typing import List, Tuple

EV_BRANDS = {"Renault", "Tesla", "Citroen", "Peugeot",
             "Dacia", "Volkswagen", "BMW", "Fiat", "KIA"}

@dataclass(slots=True)
class SimulationConfig:
    BEGIN: int              
    END: int            
    STEP_LENGTH: float    
    CHUNK_LENGTH: int        
    THREADS: int              
    TIME_TO_TELEPORT: int    
    TRACI_PORT: int        
    TARGET_UTIL: float        
    MAX_ADD_PER_CHUNK: int  
    LOW_BATTERY_THRESHOLD: float
    TLS_UPDATE_TIME: int
    CAP_CHECK_INTERVAL: int
    HALTING_VEHS_THRESHOLD: int
    EXTREME_JAM_STOP: int
    VEH_LENGTH: float
    SAT_FACTOR: float
    NET_XML: Path; ROUTES_XML: Path
    TLS_CYCLE_XML: Path; TLS_COORD_XML: Path
    OUTPUTS_DIR: Path; STATES_DIR: Path
    ROUTE_POOL_JSON: Path

BAT_PARAMS = (
    "has.battery.device",
    "device.battery.capacity",
    "device.battery.actualBatteryCapacity",
)

def _is_ev_type(type_id: str, cache: dict[str, bool]) -> bool:
    if type_id not in cache:
        cache[type_id] = any(b in type_id for b in EV_BRANDS)
    return cache[type_id]

def _get_battery_state(veh_id: str) -> tuple[float, float]:
    try:
        cur = float(traci.vehicle.getParameter(
            veh_id, "device.battery.actualBatteryCapacity") or 0)
        cap = float(traci.vehicle.getParameter(
            veh_id, "device.battery.capacity") or 0)
        return cur, cap
    except tc.TraCIException:
        return 0.0, 0.0
    
def init_simulation(cfg: SimulationConfig) -> dict[str, t.Any]:
    """Khởi tạo TraCI & mọi cấu trúc, trả context."""
    # chọn state mới nhất
    state_files = sorted(cfg.STATES_DIR.glob("state_*.xml"),
                         key=lambda p: int(p.stem.split("_")[1]))
    load_state = state_files[-1] if state_files else None

    traci.start(_build_cmd(cfg, load_state), port=cfg.TRACI_PORT)

    ctx = {
        "cfg": cfg,
        "batt_fp": _open_battery_log(cfg),
        "logged_veh": set(),
        "congested": set(),
        "tls_info": _build_tls_table(cfg),
        "type_cache": {}, 

        "chunk_end": cfg.BEGIN + cfg.CHUNK_LENGTH,
        "next_cap": cfg.BEGIN,

        "route_pool": _load_route_pool(cfg.ROUTE_POOL_JSON),
        "cursor": 0,
        "capacity": _calc_capacity(cfg),
    }

    signal.signal(signal.SIGINT, lambda *_: graceful_exit(ctx, 130))
    return ctx


def _build_cmd(cfg: SimulationConfig, load_state: Path | None) -> list[str]:
    c = cfg
    cmd = [
        "sumo", "-n", str(c.NET_XML), "-r", str(c.ROUTES_XML),
        "--threads", str(c.THREADS),
        "--step-length", str(c.STEP_LENGTH),
        "--begin", str(c.BEGIN), "--end", str(c.END + 1),
        "--summary-output",   str(c.OUTPUTS_DIR / "summary.xml"),
        "--tripinfo-output",  str(c.OUTPUTS_DIR / "tripinfo.xml"),
        "--statistic-output", str(c.OUTPUTS_DIR / "statistics.xml"),
        "--vehroute-output",  str(c.OUTPUTS_DIR / "vehRoutes.xml"),
        "--lanechange-output",str(c.OUTPUTS_DIR / "laneChanges.xml"),
        "--collision-output", str(c.OUTPUTS_DIR / "collisions.xml"),
        "--edgedata-output",  str(c.OUTPUTS_DIR / "edgedata.xml"),
        "--lanedata-output",  str(c.OUTPUTS_DIR / "lanedata.xml"),
        "--queue-output",     str(c.OUTPUTS_DIR / "queue.xml"),
        "--battery-output",   str(c.OUTPUTS_DIR / "battery.xml"),
        "--battery-output.precision", "4",
        "--time-to-teleport", str(c.TIME_TO_TELEPORT),
        "--ignore-junction-blocker", "20",
        "--lateral-resolution", "0.4",
        "--ignore-route-errors",
        "--log", str(c.OUTPUTS_DIR / "sumo.log"),
        "--no-step-log", "--duration-log.statistics",
        "-a", f"{c.TLS_CYCLE_XML},{c.TLS_COORD_XML}",
        "--save-state.rng", "true",           
    ]
    if load_state:
        cmd += ["--load-state", str(load_state)]
    return cmd

def _open_battery_log(cfg: SimulationConfig) -> t.TextIO:
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fp = (cfg.OUTPUTS_DIR / "battery_events.csv").open("a", encoding="utf-8")
    if fp.tell() == 0:
        fp.write("time,veh_id,x,y,edge_id,current_charge,capacity\n")
        fp.flush()
    return fp

def step_once(ctx):
    traci.simulationStep()
    now = traci.simulation.getTime()

    _battery(now, ctx)
    _congestion(now, ctx)
    _tls(now, ctx)

    if now >= ctx["chunk_end"] or now >= ctx["cfg"].END:
        _checkpoint(now, ctx)
        ctx["chunk_end"] += ctx["cfg"].CHUNK_LENGTH

def _checkpoint(now: float, ctx):
    cfg = ctx["cfg"]
    tag = f"{int(now):07d}"
    folder = cfg.OUTPUTS_DIR / tag
    folder.mkdir(exist_ok=True)
    print(f"[{dt.datetime.now():%H:%M:%S}] CHUNK {tag}")

    tmp = save_state(ctx, tag)
    shutil.move(tmp, folder / "state.xml")

    for fn in ("summary.xml","tripinfo.xml",
               "edgeData.xml","laneData.xml",
               "vehRoutes.xml","laneChanges.xml",
               "statistics.xml","battery.xml",
               "queue.xml","collisions.xml"):
        src = cfg.OUTPUTS_DIR / fn
        if src.exists():
            shutil.copy2(src, folder / fn)
            with open(src, "wb"): pass
            open(src, "wb").close()

    ctx["batt_fp"].close()
    bat = cfg.OUTPUTS_DIR / "battery_events.csv"
    shutil.move(bat, folder / "battery_events.csv")
    bat.touch()
    ctx["batt_fp"] = bat.open("a", encoding="utf-8")
    ctx["batt_fp"].write("time,veh_id,x,y,edge_id,current_charge,capacity\n")

    _scale_next_chunk_demand(ctx)

    traci.simulation.loadState(str(folder / "state.xml"))
    _resubscribe_after_load(ctx)


def _scale_next_chunk_demand(ctx):
    cfg   = ctx["cfg"]
    live  = len(traci.vehicle.getIDList())
    cap   = ctx["capacity"]
    util  = live / cap
    remain_frac = max(0., 1 - util)
    target_add  = int(min(cfg.MAX_ADD_PER_CHUNK,
                          remain_frac * cap * (1 / cfg.CHUNK_LENGTH)))  

    print(f"  -> live={live}, util={util:.2%}, add_next={target_add}")

    pool, cursor = ctx["route_pool"], ctx["cursor"]
    for k in range(target_add):
        rid, edges = pool[(cursor + k) % len(pool)]
        new_rid = f"r_{cursor+k}"
        vid     = f"v_{cursor+k}"
        try:
            traci.route.add(new_rid, edges)
            traci.vehicle.addFull(vid, new_rid,
                                  depart=traci.simulation.getTime() + ctx["cfg"].STEP_LENGTH)
        except tc.TraCIException:
            pass
    ctx["cursor"] = cursor + target_add

def _resubscribe_after_load(ctx): 
    pass

def graceful_exit(ctx: dict[str, t.Any], code=0):
    try:
        save_state(ctx, int(traci.simulation.getTime()))
    finally:
        ctx["batt_fp"].close()
        traci.close(False)
    sys.exit(code)

def save_state(ctx: dict[str, t.Any], tag: int | str):
    fp = ctx["cfg"].STATES_DIR  / f"state_{tag}.xml"
    traci.simulation.saveState(str(fp))
    return fp

def _lane_over(cap_cfg: SimulationConfig, lane: str) -> bool:
    L = traci.lane.getLength(lane)
    return traci.lane.getLastStepVehicleNumber(lane) > int((L / cap_cfg.VEH_LENGTH) * cap_cfg.SAT_FACTOR)

def _battery(now: float, ctx: dict[str, t.Any]):
    cfg        = ctx["cfg"]
    type_cache = ctx["type_cache"]
    logged     = ctx["logged_veh"]

    for vid in traci.vehicle.getIDList():
        if vid in logged:
            continue

        vtype = traci.vehicle.getTypeID(vid)
        if not _is_ev_type(vtype, type_cache):
            continue  

        try:
            cur = float(traci.vehicle.getParameter(
                vid, "device.battery.actualBatteryCapacity") or 0)
            cap = float(traci.vehicle.getParameter(
                vid, "device.battery.capacity") or 0)
        except tc.TraCIException:
            type_cache[vtype] = False
            continue

        if cap == 0 or (cur / cap) >= cfg.LOW_BATTERY_THRESHOLD:
            continue

        x, y  = traci.vehicle.getPosition(vid)
        edge  = traci.vehicle.getRoadID(vid)
        fp    = ctx["batt_fp"]
        fp.write(f"{now:.1f},{vid},{x:.2f},{y:.2f},{edge},{cur:.3f},{cap:.3f}\n")
        fp.flush()
        logged.add(vid)


def _congestion(now: float, ctx: dict[str, t.Any]):
    cfg, congested = ctx["cfg"], ctx["congested"]
    if now >= ctx["next_cap"]:
        congested.clear()
        congested.update(l for l in traci.lane.getIDList() if _lane_over(cfg, l))
        ctx["next_cap"] += cfg.CAP_CHECK_INTERVAL

    for vid in traci.simulation.getDepartedIDList():
        try:
            if traci.vehicle.getLaneID(vid) in congested:
                traci.vehicle.remove(vid)
        except tc.TraCIException: pass

    for lane in congested:
        if traci.lane.getLastStepHaltingNumber(lane) > cfg.EXTREME_JAM_STOP:
            for vid in traci.lane.getLastStepVehicleIDs(lane):
                try: traci.vehicle.remove(vid)
                except tc.TraCIException: pass

    if now >= 6*3600 and (now % 60) < cfg.STEP_LENGTH:
        edges = [e for e in traci.edge.getIDList()
                 if traci.edge.getLastStepHaltingNumber(e) > cfg.HALTING_VEHS_THRESHOLD]
        heavy = len(edges) > 5
        for e in edges:
            for vid in traci.edge.getLastStepVehicleIDs(e):
                try:
                    if heavy:
                        traci.vehicle.remove(vid)
                    else:
                        curr = traci.vehicle.getRoadID(vid)
                        dest = traci.vehicle.getRoute(vid)[-1]
                        new = traci.simulation.findRoute(curr,dest).edges
                        if new: traci.vehicle.setRoute(vid,new)
                except tc.TraCIException: pass

def _build_tls_table(cfg: SimulationConfig):
    res = {}
    for tls in traci.trafficlight.getIDList():
        links = traci.trafficlight.getControlledLinks(tls)
        best = {"ew": (None, -1), "ns": (None, -1)}
        for idx, grp in enumerate(links):
            ew = sum(1 for l,_,_ in grp if abs(traci.lane.getShape(l)[-1][0]-traci.lane.getShape(l)[0][0]) >
                                               abs(traci.lane.getShape(l)[-1][1]-traci.lane.getShape(l)[0][1]))
            ns = len(grp) - ew
            if ew > best["ew"][1]: best["ew"] = (idx, ew)
            if ns > best["ns"][1]: best["ns"] = (idx, ns)
        if best["ew"][0] is not None and best["ns"][0] is not None:
            res[tls] = {"ew": best["ew"][0], "ns": best["ns"][0]}
    return res

def _tls(now: float, ctx: dict[str, t.Any]):
    cfg = ctx["cfg"]
    if (now % cfg.TLS_UPDATE_TIME) >= cfg.STEP_LENGTH: return
    for tls, idxs in ctx["tls_info"].items():
        try:
            num_p = len(traci.trafficlight.getAllProgramLogics(tls)[0].phases)
        except (IndexError, tc.TraCIException): continue
        if num_p <= 1: continue
        ew_cnt = sum(traci.lane.getLastStepVehicleNumber(l[0])
                     for l in traci.trafficlight.getControlledLinks(tls)[idxs["ew"]])
        ns_cnt = sum(traci.lane.getLastStepVehicleNumber(l[0])
                     for l in traci.trafficlight.getControlledLinks(tls)[idxs["ns"]])
        choose = idxs["ew"] if ew_cnt > ns_cnt else idxs["ns"]
        traci.trafficlight.setPhase(tls, choose % num_p)

def _calc_capacity(cfg: SimulationConfig) -> int:
    total = 0.0
    for lane_id in traci.lane.getIDList():
        if lane_id.startswith(":"):             
            continue
        try:
            L = traci.lane.getLength(lane_id)     # mét
            total += (L / cfg.VEH_LENGTH) * cfg.SAT_FACTOR
        except traci.exceptions.TraCIException:
            continue                             
    return int(total)

def _load_route_pool(path: Path) -> List[Tuple[str, list[str]]]:
    opener = gzip.open if path.suffix in (".gz", ".gzip") else open

    pool: list[tuple[str, list[str]]] = []
    with opener(path, "rt", encoding="utf-8") as f:
        first = f.readline()
        if not first.strip():
            raise RuntimeError(f"empty file: {path}")
        is_jsonl = not first.lstrip().startswith("[")   
        f.seek(0)                    
        if is_jsonl:
            for ln in f:
                if ln.strip():
                    obj = json.loads(ln)
                    pool.append((obj["id"], obj["edges"]))
        else:
            data = json.load(f)                      
            pool = [(d["id"], d["edges"]) for d in data]

    if not pool:
        raise RuntimeError(f"empty Route-pool: {path}")
    return pool
