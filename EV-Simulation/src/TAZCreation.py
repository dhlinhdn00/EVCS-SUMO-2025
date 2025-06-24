import argparse
import json
from simulationHelpers import *

def createTAZ(grouped_poly_xml: str, net_xml: str, taz_ids: dict, taz_output_xml: str, sample_inner: int = 40, sample_cross: int = 40, border_ratio: float = 0.4, boudary_ratio: float = 0.1, filter_mode: str = 'sampling'):
    """
    Create TAZ file with 3 modes:
      - 'sampling'  : sampling + reachability
      - 'full'      : full-connectivity
      - 'spatial'   : only spatial-boundary sampling
    """
    print("[1] Reading bassins ...")
    region_geoms = defaultdict(list)
    for p in ET.parse(grouped_poly_xml).getroot().findall('poly'):
        r = p.get('type'); s = p.get('shape','')
        g = parseShape(s)
        if r and g:
            region_geoms[r].append(g)
    for r, L in list(region_geoms.items()):
        region_geoms[r] = unary_union(L) if len(L)>1 else L[0]

    net = sumolib.net.readNet(net_xml)
    edges_by_region = defaultdict(list)
    outside = []
    for e in net.getEdges():
        if not isValidEdge(e): continue
        mid = Point(e.getShape()[len(e.getShape())//2])
        placed = False
        for r,g in region_geoms.items():
            if g.contains(mid):
                edges_by_region[r].append(e)
                placed = True
                break
        if not placed:
            outside.append(e)
    edges_by_region['outside'] = outside

    if filter_mode == 'sampling':
        print("[2a] Inner‐bassin sampling…")
        for r,pool in list(edges_by_region.items()):
            kept = filterReachable(pool, net, sample_inner)
            edges_by_region[r] = kept

        print("[2b] Cross‐bassin sampling…")
        for r,pool in list(edges_by_region.items()):
            others = [e for rr,pl in edges_by_region.items() if rr!=r for e in pl]
            kept = [e for e in pool 
                    if any(reachable(e,t,net) for t in random.sample(others, min(sample_cross,len(others))))
                    and any(reachable(t,e,net) for t in random.sample(others, min(sample_cross,len(others))))]
            edges_by_region[r] = kept

        print("[3] Writing TAZ (sampling)…")
        root = ET.Element('tazs')
        for r,pool in edges_by_region.items():
            if not pool: continue
            tid = taz_ids.get(r.lower())
            if tid is None:
                print(f"! no TAZ id for {r}, skip"); continue

            B = selectBoundaryEdges(pool, region_geoms.get(r))
            I = [e for e in pool if e not in B]
            total = len(pool)
            nB = int(border_ratio * total)
            nI = total - nB
            chosen = random.sample(B, min(nB,len(B))) + \
                     random.sample(I, min(nI,len(I)))

            cent = region_geoms[r].centroid if r in region_geoms else Point(0,0)
            taz = ET.SubElement(root,'taz', id=str(tid),
                                x=f"{cent.x:.2f}", y=f"{cent.y:.2f}")
            for e in sorted(chosen, key=lambda e:e.getID()):
                ET.SubElement(taz,'tazSource', id=e.getID(), weight='1.0')
                ET.SubElement(taz,'tazSink',   id=e.getID(), weight='1.0')

    elif filter_mode == 'full':
        print("[2a] Inner‐bassin full...")
        srcs_by_region, snks_by_region = {}, {}
        for r,pool in edges_by_region.items():
            srcs = filterSources(pool, net)
            snks = filterSinks(pool, net)
            srcs_by_region[r], snks_by_region[r] = srcs, snks

        print("[2b] Cross‐bassin full...")
        for r,pool in list(edges_by_region.items()):
            others = [e for rr,pl in edges_by_region.items() if rr!=r for e in pl]
            srcs_by_region[r] = [e for e in srcs_by_region[r]
                                 if any(reachable(e,o,net) for o in others)]
            snks_by_region[r] = [e for e in snks_by_region[r]
                                 if any(reachable(o,e,net) for o in others)]

        print("[3] Writing TAZ (full)...")
        root = ET.Element('tazs')
        for r in edges_by_region:
            srcs, snks = srcs_by_region[r], snks_by_region[r]
            if not srcs and not snks: continue
            tid = taz_ids.get(r.lower())
            if tid is None:
                print(f"! no TAZ id for {r}, skip"); continue

            cent = region_geoms[r].centroid if r in region_geoms else Point(0,0)
            taz = ET.SubElement(root,'taz', id=str(tid),
                                x=f"{cent.x:.2f}", y=f"{cent.y:.2f}")
            for e in sorted(srcs, key=lambda e:e.getID()):
                ET.SubElement(taz,'tazSource', id=e.getID(), weight='1.0')
            for e in sorted(snks, key=lambda e:e.getID()):
                ET.SubElement(taz,'tazSink',   id=e.getID(), weight='1.0')

    elif filter_mode == 'spatial':
        print("[3] Writing TAZ (spatial)…")
        root = ET.Element('tazs')
        for r, pool in edges_by_region.items():
            tid = taz_ids.get(r.lower())
            if tid is None:
                print(f"! no TAZ id for {r}, skip")
                continue

            if border_ratio == -1:
                chosen = pool[:]
            elif r == 'outside':
                chosen = pool[:]
            else:
                B = selectBoundaryEdges(pool, region_geoms.get(r))
                I = [e for e in pool if e not in B]
                total = len(B) + len(I)
                nB = int(border_ratio * total)
                nI = total - nB
                chosen = random.sample(B, min(nB, len(B))) + \
                        random.sample(I, min(nI, len(I)))

            cent = (region_geoms[r].centroid
                    if r in region_geoms else Point(0, 0))
            taz = ET.SubElement(root, 'taz', id=str(tid),
                                x=f"{cent.x:.2f}", y=f"{cent.y:.2f}")
            for e in sorted(chosen, key=lambda e: e.getID()):
                ET.SubElement(taz, 'tazSource', id=e.getID(), weight='1.0')
                ET.SubElement(taz, 'tazSink',   id=e.getID(), weight='1.0')

    else:
        raise ValueError(f"filter_mode must be one of 'sampling','full','spatial'")

    ET.ElementTree(root).write(taz_output_xml,
                               encoding='utf-8',
                               xml_declaration=True)
    print(f"[DONE] ({filter_mode}) wrote {taz_output_xml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a TAZ XML file from grouped polygons, a SUMO network, and a region-to-TAZ-ID mapping"
    )
    parser.add_argument(
        "-g", "--grouped-poly-xml",
        required=True,
        help="Path to the input grouped polygons XML file"
    )
    parser.add_argument(
        "-n", "--net-xml",
        required=True,
        help="Path to the SUMO network XML file"
    )
    parser.add_argument(
        "-t", "--taz-ids",
        required=True,
        help="Path to the JSON file mapping region names to TAZ IDs"
    )
    parser.add_argument(
        "-o", "--taz-output-xml",
        required=True,
        help="Path for the output TAZ XML file"
    )
    parser.add_argument(
        "--sample-inner",
        type=int,
        default=40,
        help="Number of inner-basin samples (default: 40)"
    )
    parser.add_argument(
        "--sample-cross",
        type=int,
        default=40,
        help="Number of cross-basin samples (default: 40)"
    )
    parser.add_argument(
        "--border-ratio",
        type=float,
        default=0.4,
        help="Fraction of chosen edges from the boundary (default: 0.4)"
    )
    parser.add_argument(
        "--boundary-ratio",
        type=float,
        default=0.1,
        help="Secondary boundary sampling ratio if used (default: 0.1)"
    )
    parser.add_argument(
        "--filter-mode",
        choices=["sampling", "full", "spatial"],
        default="sampling",
        help="Filtering mode: 'sampling', 'full', or 'spatial' (default: sampling)"
    )

    args = parser.parse_args()

    with open(args.taz_ids, "r") as f:
        taz_ids = json.load(f)

    createTAZ(
        grouped_poly_xml=args.grouped_poly_xml,
        net_xml=args.net_xml,
        taz_ids=taz_ids,
        taz_output_xml=args.taz_output_xml,
        sample_inner=args.sample_inner,
        sample_cross=args.sample_cross,
        border_ratio=args.border_ratio,
        boudary_ratio=args.boundary_ratio,
        filter_mode=args.filter_mode
    )