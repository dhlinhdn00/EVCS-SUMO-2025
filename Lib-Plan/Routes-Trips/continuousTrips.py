import random
import argparse
import sumolib

def generate_continuous_trip(net, start_edge, target_length_m, allow_revisit=False, weight_fun=None):
    """
    Sinh một tuyến đường liền mạch trên mạng `net`, bắt đầu từ `start_edge`,
    cho đến khi tổng chiều dài >= target_length_m.

    Tham số:
    - net: sumolib.net.Net
    - start_edge: sumolib.net.Edge
    - target_length_m: float (m)
    - allow_revisit: bool (có cho phép quay lại cạnh đã đi qua)
    - weight_fun: hàm (edge -> float) để sampling có trọng số; default random.

    Trả về:
    - edges: list các Edge theo thứ tự
    """
    route = [start_edge]
    total = start_edge.getLength()
    visited = set([start_edge.getID()])
    current = start_edge

    while total < target_length_m:
        outs = current.getToNode().getOutgoing()
        # Nếu node cuối không có outgoing, không thể tiếp tục
        if not outs:
            raise RuntimeError(f"Dead end at edge {current.getID()} (no outgoing edges)")
        # Lọc lại theo allow_revisit
        candidates = [e for e in outs if allow_revisit or e.getID() not in visited]
        # Nếu không còn cạnh mới và không cho phép revisit, cho phép revisit tạm thời
        if not candidates:
            candidates = outs
        # Chọn theo trọng số hoặc ngẫu nhiên
        if weight_fun:
            weights = [weight_fun(e) for e in candidates]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0.0
            for e, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    next_edge = e
                    break
        else:
            next_edge = random.choice(candidates)
        route.append(next_edge)
        visited.add(next_edge.getID())
        total += next_edge.getLength()
        current = next_edge
    return route


def write_route_file(route_edges, output_file, trip_id="longTrip", veh_id="veh0"):
    """
    Ghi file route SUMO (.rou.xml) với một route và một vehicle
    """
    edge_ids = [e.getID() for e in route_edges]
    edges_str = " ".join(edge_ids)
    with open(output_file, "w") as f:
        f.write("<routes>\n")
        f.write(f"    <route id=\"{trip_id}\" edges=\"{edges_str}\"/>\n")
        f.write(f"    <vehicle id=\"{veh_id}\" route=\"{trip_id}\" depart=\"0\"/>\n")
        f.write("</routes>\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a single continuous trip of given length on a SUMO network"
    )
    parser.add_argument("-n", "--net-file", required=True,
                        help="Input SUMO network file (.net.xml)")
    parser.add_argument("-x", "--length-km", type=float, default=300.0,
                        help="Target trip length in kilometers (default 300)")
    parser.add_argument("-o", "--output-file", required=True,
                        help="Output route file (.rou.xml)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--allow-revisit", action="store_true",
                        help="Allow revisiting edges if no new outgoing edges remain")
    parser.add_argument("--max-restarts", type=int, default=100,
                        help="Max attempts to restart from a new start edge if stuck (default 100)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    net = sumolib.net.readNet(args.net_file)
    valid_edges = [e for e in net.getEdges() if e.allows("passenger") and e.getToNode().getOutgoing()]
    if not valid_edges:
        raise RuntimeError("Không tìm được cạnh xuất phát phù hợp (cho phép và có outgoing)")

    target_m = args.length_km * 1000
    route = None
    for attempt in range(args.max_restarts):
        start_edge = random.choice(valid_edges)
        try:
            route = generate_continuous_trip(
                net, start_edge, target_m,
                allow_revisit=args.allow_revisit
            )
            break
        except RuntimeError as e:
            # Stuck at dead end, thử lại
            continue

    if route is None:
        raise RuntimeError(f"Không thể tạo tuyến liên tục sau {args.max_restarts} lần thử")

    write_route_file(route, args.output_file)
    total_km = sum(e.getLength() for e in route) / 1000
    print(f"Generated continuous trip ~{total_km:.2f} km; edges: {len(route)}")

if __name__ == "__main__":
    main()