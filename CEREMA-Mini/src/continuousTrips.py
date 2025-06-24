import random
import argparse
import sumolib

def generateContinuousTrip(net, start_edge, target_length_m, allow_revisit=False, weight_fun=None):
    """
    Generate a continuous trip on the given SUMO network, starting from the specified edge,
    until the total trip length is at least target_length_m meters.

    Parameters:
    - net (sumolib.net.Net): the SUMO network object
    - start_edge (sumolib.net.Edge): the starting edge for the trip
    - target_length_m (float): the desired trip length in meters
    - allow_revisit (bool): if True, allows revisiting edges that have already been traversed
    - weight_fun (callable): a function mapping an edge to a float weight for weighted sampling; if None, selects randomly

    Returns:
    - list of sumolib.net.Edge: the sequence of edges comprising the trip
    """
    route = [start_edge]
    total_length = start_edge.getLength()
    visited = set([start_edge.getID()])
    current_edge = start_edge

    while total_length < target_length_m:
        outgoing_edges = current_edge.getToNode().getOutgoing()
        if not outgoing_edges:
            raise RuntimeError(f"Dead end at edge {current_edge.getID()} (no outgoing edges)")
        # Filter candidates based on revisit policy
        candidates = [e for e in outgoing_edges if allow_revisit or e.getID() not in visited]
        # If no new edges and revisits not allowed, temporarily allow revisits
        if not candidates:
            candidates = outgoing_edges
        # Select next edge by weight or randomly
        if weight_fun:
            weights = [weight_fun(e) for e in candidates]
            total_weight = sum(weights)
            r = random.random() * total_weight
            cum = 0.0
            for edge, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    next_edge = edge
                    break
        else:
            next_edge = random.choice(candidates)
        route.append(next_edge)
        visited.add(next_edge.getID())
        total_length += next_edge.getLength()
        current_edge = next_edge

    return route

def writeRouteFile(route_edges, output_file, trip_id="continuousTrip", vehicle_id="veh0"):  
    """
    Write a SUMO route file (.rou.xml) containing a single trip and vehicle.

    Parameters:
    - route_edges (list of sumolib.net.Edge): edges in the generated trip
    - output_file (str): path to the output .rou.xml file
    - trip_id (str): identifier for the trip (default: "continuousTrip")
    - vehicle_id (str): identifier for the vehicle (default: "veh0")
    """
    edge_ids = [edge.getID() for edge in route_edges]
    edges_str = " ".join(edge_ids)
    with open(output_file, "w") as f:
        f.write("<routes>\n")
        f.write(f'    <route id="{trip_id}" edges="{edges_str}"/>\n')
        f.write(f'    <vehicle id="{vehicle_id}" route="{trip_id}" depart="0"/>\n')
        f.write("</routes>\n")

def main():
    parser = argparse.ArgumentParser(
        description="Generate a continuous trip of specified length on a SUMO network"
    )
    parser.add_argument(
        "-n", "--net-file", required=True,
        help="Input SUMO network file (.net.xml)"
    )
    parser.add_argument(
        "-x", "--length-km", type=float, default=300.0,
        help="Target trip length in kilometers (default: 300)"
    )
    parser.add_argument(
        "-o", "--output-file", required=True,
        help="Output route file (.rou.xml)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--allow-revisit", action="store_true",
        help="Allow revisiting edges when no new outgoing edges are available"
    )
    parser.add_argument(
        "--max-restarts", type=int, default=100,
        help="Maximum number of attempts to restart with a new start edge if stuck (default: 100)"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    net = sumolib.net.readNet(args.net_file)
    valid_edges = [
        e for e in net.getEdges()
        if e.allows("passenger") and e.getToNode().getOutgoing()
    ]
    if not valid_edges:
        raise RuntimeError("No suitable start edge found (must allow passenger and have outgoing edges)")

    target_length_m = args.length_km * 1000
    route = None
    for attempt in range(args.max_restarts):
        start_edge = random.choice(valid_edges)
        try:
            route = generateContinuousTrip(
                net, start_edge, target_length_m,
                allow_revisit=args.allow_revisit
            )
            break
        except RuntimeError:
            # Stuck at a dead end, try another start edge
            continue

    if route is None:
        raise RuntimeError(f"Unable to generate a continuous trip after {args.max_restarts} attempts")

    writeRouteFile(route, args.output_file)
    total_km = sum(edge.getLength() for edge in route) / 1000
    print(f"Generated continuous trip ~{total_km:.2f} km; edges: {len(route)}")

if __name__ == "__main__":
    main()
