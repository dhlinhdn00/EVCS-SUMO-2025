from sumolib.net import readNet
from lxml import etree
import re

def checkDisconnectedEdges(net_file, verbose=True):
    """
    Checks for disconnected edges in the network.
    An edge is considered disconnected if its fromNode or toNode is missing,
    or if its fromNode does not have any incoming edges or its toNode does not have any outgoing edges.
    """
    net = readNet(net_file)
    disconnected = []
    for edge in net.getEdges():
        if edge.getFromNode() is None or edge.getToNode() is None:
            disconnected.append(edge.getID())
        elif not edge.getFromNode().getIncoming() or not edge.getToNode().getOutgoing():
            disconnected.append(edge.getID())
    
    if verbose:
        print(f"Found {len(disconnected)} disconnected edges:")
        for eid in disconnected:
            print(f" - {eid}")
    else:
        print(f"Found {len(disconnected)} disconnected edges.")
    return disconnected

def createDisconnectedEdgesPoly(net_file, output_poly_file, color="1,0,0", lineWidth=3):
    """
    Uses the checkDisconnectedEdges function to create an additional file (.poly.xml)
    that highlights disconnected edges for use in SUMO-GUI.
    
    For each disconnected edge, the function obtains its shape (if available) or else uses
    the coordinates of its fromNode and toNode. These are then formatted as a 'poly' element.
    
    Parameters:
      - net_file: Path to the SUMO network file.
      - output_poly_file: Output filename for the additional .poly.xml file.
      - color: Color for the poly elements (e.g., "1,0,0" for red).
      - lineWidth: The line width of the displayed poly.
    """
    net = readNet(net_file)
    disconnected_ids = checkDisconnectedEdges(net_file, verbose=False)
    
    poly_lines = []
    for eid in disconnected_ids:
        edge = net.getEdge(eid)
        if edge is None:
            continue
        # Get the edge shape; if missing, use the coordinates of fromNode and toNode.
        shape = edge.getShape()
        if not shape:
            shape = [edge.getFromNode().getCoord(), edge.getToNode().getCoord()]
        # Format the shape: "x1,y1 x2,y2 ..." with two-decimal precision.
        shape_str = " ".join([f"{x:.2f},{y:.2f}" for x, y in shape])
        # Build a poly element
        poly_lines.append(f'<poly id="{eid}" color="{color}" lineWidth="{lineWidth}" shape="{shape_str}"/>')
    
    # Write the additional file in SUMO XML format.
    with open(output_poly_file, "w", encoding="utf-8") as f:
        f.write("<additional>\n")
        for line in poly_lines:
            f.write(f"  {line}\n")
        f.write("</additional>\n")
        
    print(f"Additional poly file saved to {output_poly_file}")

def checkShortEdges(net_file, min_length=5.0, verbose=True):
    """
    Checks for edges with a length below a given threshold.
    """
    net = readNet(net_file)
    short_edges = []
    for edge in net.getEdges():
        if edge.getLength() < min_length:
            short_edges.append((edge.getID(), edge.getLength()))
    
    if verbose:
        print(f"Found {len(short_edges)} edges shorter than {min_length} meters:")
        for eid, length in short_edges:
            print(f" - {eid}: {length:.2f} m")
    else:
        print(f"Found {len(short_edges)} edges shorter than {min_length} meters.")
    return short_edges

def checkZeroLaneEdges(net_file, verbose=True):
    """
    Checks for edges that have zero lanes.
    """
    net = readNet(net_file)
    zero_lane_edges = []
    for edge in net.getEdges():
        if edge.getLaneNumber() == 0:
            zero_lane_edges.append(edge.getID())
    
    if verbose:
        print(f"Found {len(zero_lane_edges)} edges with zero lanes:")
        for eid in zero_lane_edges:
            print(f" - {eid}")
    else:
        print(f"Found {len(zero_lane_edges)} edges with zero lanes.")
    return zero_lane_edges

def checkSpeedEdges(net_file, min_speed=5.0, max_speed=35.0, verbose=True):
    """
    Checks for edges with speed limits outside the expected range (in m/s).
    """
    net = readNet(net_file)
    low_speed_edges = []
    high_speed_edges = []
    for edge in net.getEdges():
        speed = edge.getSpeed()
        if speed < min_speed:
            low_speed_edges.append((edge.getID(), speed))
        elif speed > max_speed:
            high_speed_edges.append((edge.getID(), speed))
    
    if verbose:
        print(f"Found {len(low_speed_edges)} edges with speed lower than {min_speed} m/s:")
        for eid, speed in low_speed_edges:
            print(f" - {eid}: {speed:.2f} m/s")
        print(f"Found {len(high_speed_edges)} edges with speed higher than {max_speed} m/s:")
        for eid, speed in high_speed_edges:
            print(f" - {eid}: {speed:.2f} m/s")
    else:
        print(f"Found {len(low_speed_edges)} edges with speed lower than {min_speed} m/s.")
        print(f"Found {len(high_speed_edges)} edges with speed higher than {max_speed} m/s.")
    
    return low_speed_edges, high_speed_edges

def checkEdgeCounts(net_file, min_length=5.0, verbose=True):
    """
    Checks overall network edge counts.
    Reports:
      - Total number of edges.
      - Number of valid edges: those that are connected, have at least one lane,
        and have a length >= the specified threshold.
    """
    net = readNet(net_file)
    total_edges = len(net.getEdges())
    valid_edges = []
    for edge in net.getEdges():
        if (edge.getFromNode() is not None and edge.getToNode() is not None and 
            edge.getLaneNumber() > 0 and edge.getLength() >= min_length):
            valid_edges.append(edge.getID())
    
    if verbose:
        print(f"Total number of edges: {total_edges}")
        print(f"Number of valid edges (connected, with lanes, and length >= {min_length} m): {len(valid_edges)}")
    else:
        print(f"Total edges: {total_edges}, Valid edges: {len(valid_edges)}")
    
    return total_edges, valid_edges

def runAllEdgeChecks(net_file, verbose=True):
    """
    Runs a series of checks on the network and outputs a summary including:
      - Disconnected edges.
      - Short edges.
      - Zero-lane edges.
      - Edges with out-of-range speed limits.
      - Overall counts of total edges and valid edges.
    When verbose is False, only summary numbers are printed (and the returned JSON contains counts, not detailed lists).
    """
    if verbose:
        print("=== Running Disconnected Edges Check ===")
    disconnected = checkDisconnectedEdges(net_file, verbose=verbose)
    
    if verbose:
        print("\n=== Running Short Edges Check ===")
    short_edges = checkShortEdges(net_file, min_length=5.0, verbose=verbose)
    
    if verbose:
        print("\n=== Running Zero-Lane Edges Check ===")
    zero_lane = checkZeroLaneEdges(net_file, verbose=verbose)
    
    if verbose:
        print("\n=== Running Speed Edges Check ===")
    low_speed, high_speed = checkSpeedEdges(net_file, min_speed=5.0, max_speed=35.0, verbose=verbose)
    
    if verbose:
        print("\n=== Running Edge Count Summary ===")
    total_edges, valid_edges = checkEdgeCounts(net_file, min_length=5.0, verbose=verbose)
    
    if not verbose:
        # Return only numeric summaries when verbose is False.
        results = {
            "disconnected_edges": len(disconnected),
            "short_edges": len(short_edges),
            "zero_lane_edges": len(zero_lane),
            "low_speed_edges": len(low_speed),
            "high_speed_edges": len(high_speed),
            "total_edges": total_edges,
            "valid_edges": len(valid_edges)
        }
    else:
        results = {
            "disconnected_edges": disconnected,
            "short_edges": short_edges,
            "zero_lane_edges": zero_lane,
            "low_speed_edges": low_speed,
            "high_speed_edges": high_speed,
            "total_edges": total_edges,
            "valid_edges": valid_edges
        }
    return results

def cleanOSM(input_osm_path, output_osm_path):
    """
    The cleanOSM function uses lxml.iterparse to read the OSM file in XML structure,
    iterate over elements (node, way, relation), and perform the following operations:
      - Discard elements with a <tag> that has a highway value we don't want (disused, escape, corridor, crossing).
      - Discard elements with railway=platform.
      - Fix the width format: convert commas to dots (e.g. "1,5" to "1.5"); if the width value is "narrow", "wide", or empty, remove the tag.
      - Fix the layer tag if it contains a ';' (only keep the part before the ';').
    The cleaned elements are then written to the output file.
    """
    context = etree.iterparse(input_osm_path, events=('end',), encoding='utf-8')
    valid_elements = {'node', 'way', 'relation'}

    with open(output_osm_path, 'wb') as out_file:
        out_file.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        out_file.write(b'<osm version="0.6" generator="OSM-cleaner">\n')
        
        for event, elem in context:
            if elem.tag not in valid_elements:
                continue

            discard = False
            for tag in elem.findall("tag"):
                k = tag.get("k")
                v = tag.get("v")

                # Discard ways with highway values we do not want.
                if k == "highway" and v in ["disused", "escape", "corridor", "crossing"]:
                    discard = True
                # Discard railway platform.
                if k == "railway" and v == "platform":
                    discard = True

                # Fix the width format: e.g., "1,5" -> "1.5"; if value is narrow/wide, remove the width tag.
                if k == "width":
                    if ',' in v:
                        tag.set("v", v.replace(',', '.'))
                    elif v.lower() in ["narrow", "wide", ""]:
                        elem.remove(tag)

                # Fix the layer tag if it contains a semicolon (keep only the portion before the ';').
                if k == "layer" and ';' in v:
                    tag.set("v", v.split(';')[0])
                    
            if not discard:
                out_file.write(etree.tostring(elem, encoding='utf-8'))
            elem.clear()

        out_file.write(b'</osm>')
    print(f"Finished cleaning: {input_osm_path} => {output_osm_path}")

def fixOSM(infile, outfile):
    """
    The fixOSM function uses regular expressions to perform these operations:
      - Convert the width value written as text with a comma (e.g. "1,6") to a dot format (i.e. "1.6").
      - Convert textual width values ("narrow", "medium", "wide") to numeric values based on a mapping.
      - Fix the layer tag by removing any extra separators (e.g. remove the portion after ';').
      - Assign a priority attribute to <way> elements based on the highway type (for example, highway.primary or highway.trunk -> "major").
    The function processes the file line by line and writes the transformed output to a new file.
    """
    # Regular expression to fix numeric width values with a comma
    re_width = re.compile(r'(\bwidth=")(\d+),(\d+)(")')
    # Regular expression to fix textual width values
    re_width_text = re.compile(r'(\bwidth=")(narrow|medium|wide)(")')
    width_mapping = {"narrow": "2.5", "medium": "3.0", "wide": "4.0"}
    # Regular expression for layer: remove additional parts after the separator
    re_layer = re.compile(r'(\blayer=")([^";|]+)([;|][^"]*)(")')

    # re_cycleway = re.compile(r'(type=")([^"]*cycleway\.\w+\|)(highway\.[^"]*)(")')
    # re_cycleway_opposite = re.compile(r'(type=")([^"]*cycleway\.opposite_lane\|)(highway\.[^"]*)(")')
    # re_unusable_highway = re.compile(r'(type=")([^"]*highway\.disused[^"]*)(")')
    # re_services_highway = re.compile(r'(type=")([^"]*highway\.services[^"]*)(")')
    # re_road = re.compile(r'(type=")([^"]*highway\.road[^"]*)(")')
    # re_elevator = re.compile(r'(type=")([^"]*highway\.elevator[^"]*)(")')
    # re_escape = re.compile(r'(type=")([^"]*highway\.escape[^"]*)(")')
    # re_bus_stop = re.compile(r'(type=")([^"]*highway\.bus_stop[^"]*)(")')


    def assign_priority(line):
        if '<way' in line:
            # For highway.primary or highway.trunk, assign "major".
            if re.search(r'type="[^"]*(highway\.primary|highway\.trunk)[^"]*"', line):
                if 'priority=' not in line:
                    line = re.sub(r'(<way\b[^>]*)(>)', r'\1 priority="major"\2', line)
            # For highway.secondary or highway.tertiary, assign "medium".
            elif re.search(r'type="[^"]*(highway\.secondary|highway\.tertiary)[^"]*"', line):
                if 'priority=' not in line:
                    line = re.sub(r'(<way\b[^>]*)(>)', r'\1 priority="medium"\2', line)
            # For highway.residential, highway.unclassified, or highway.living_street, assign "minor".
            elif re.search(r'type="[^"]*(highway\.residential|highway\.unclassified|highway\.living_street)[^"]*"', line):
                if 'priority=' not in line:
                    line = re.sub(r'(<way\b[^>]*)(>)', r'\1 priority="minor"\2', line)
        return line

    fixed_lines = []
    with open(infile, "r", encoding="utf-8") as fin:
        for line in fin:
            # Fix numeric width values with a comma
            line = re_width.sub(lambda m: f'{m.group(1)}{m.group(2)}.{m.group(3)}{m.group(4)}', line)
            # Fix textual width values using the mapping
            line = re_width_text.sub(lambda m: f'{m.group(1)}{width_mapping[m.group(2)]}{m.group(3)}', line)
            # Fix the layer tag by removing extra parts after the separator
            line = re_layer.sub(lambda m: f'{m.group(1)}{m.group(2)}{m.group(4)}', line)
            # Assign a priority attribute to <way> elements based on the highway type
            line = assign_priority(line)
            fixed_lines.append(line)
    with open(outfile, "w", encoding="utf-8") as fout:
        fout.writelines(fixed_lines)
    print(f"Finished fixing: {infile} => {outfile}")