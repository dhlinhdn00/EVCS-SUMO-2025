import json
import pandas as pd
import math
import os
from typing import Dict, List

def autoFillMissingInfo(file_path: str) -> None:
    with open(file_path, "r") as f:
        data = json.load(f)

    total_trips = data.get('number_of_trips', 0)
    regional = data.get('regional_trips', {})

    missing_keys = [k for k, v in regional.items() if v is None]
    if missing_keys:
        if len(missing_keys) != 1:
            raise ValueError("There must be exactly 1 missing (null) region.")
        known_sum = sum(v for v in regional.values() if v is not None)
        missing_key = missing_keys[0]
        regional[missing_key] = total_trips - known_sum
        data['regional_trips'] = regional
        print(f"Filled '{missing_key}' = {regional[missing_key]}")

    else:
        print(f"{file_path} is already complete for 'regional_trips'.")

    distribution_ratio = {
        k: v / total_trips for k, v in regional.items()
    }
    data['distribution_trips_ratio'] = distribution_ratio

    incoming_flow = data.get('incoming_flow', 0)
    outgoing_flow = data.get('outgoing_flow', 0)

    if incoming_flow > 0 and outgoing_flow > 0:
        data['regional_incoming_flow_ratio'] = incoming_flow / (incoming_flow + outgoing_flow)
        data['regional_outgoing_flow_ratio'] = outgoing_flow / (incoming_flow + outgoing_flow)
    else:
        data['regional_incoming_flow_ratio'] = None
        data['regional_outgoing_flow_ratio'] = None

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated '{file_path}': added distribution_trips_ratio, incoming_flow_ratio, outgoing_flow_ratio.")

def _ceilDict(values: Dict[str, float]) -> Dict[str, int]:
    """Return a new dict with **ceil** applied to every value."""
    return {k: math.ceil(v) for k, v in values.items()}


def _safeInt(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return int(float(x))


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def buildDetailedData(
    path_json: str,
    path_input_csv: str,
    path_mapping_json: str,
    path_output_csv: str,
    od_folder: str = "OD_matrices",
):
    """Create two artefacts for each hour described in *path_input_csv*:

    **1.** *path_output_csv* – a table that now contains **two columns per non‑main
    region**: one for *outgoing* trips (main → region) and one for *incoming*
    trips (region → main).  A final column ``total`` equals ``ceil(car_trips)``.

    **2.** One TXT file per hour inside *od_folder* that lists three kinds of
    flows: ``main→main`` (intra‑zonal), ``main→region`` (outgoing) and
    ``region→main`` (incoming).  The sum of the values in each TXT therefore
    exactly matches the ``total`` column in the CSV.

    The function expects *path_json* to contain two dictionaries:

    ```json
    {
      "outgoing_ratio": {"R1": 0.1, "R2": 0.15, ...},
      "incoming_ratio": {"R1": 0.08, "R2": 0.12, ...},
      "trips_using_car_ratio": "32%"
    }
    ```

    *Only* the regions that appear in ``outgoing_ratio``/**and** ``incoming_ratio``
    are processed.  The *main* region is inferred as the one with the **largest
    outgoing share** (ties resolved by first occurrence).
    """

    with open(path_json, "r", encoding="utf-8") as f:
        ratios = json.load(f)

    outgoing_ratio: Dict[str, float] | None = ratios.get("outgoing_ratio")
    incoming_ratio: Dict[str, float] | None = ratios.get("incoming_ratio")

    if outgoing_ratio is None or incoming_ratio is None:
        dist_ratio: Dict[str, float] | None = ratios.get("distribution_trips_ratio")
        r_out = ratios.get("regional_outgoing_flow_ratio")
        r_in = ratios.get("regional_incoming_flow_ratio")

        if dist_ratio is None or r_out is None or r_in is None:
            raise ValueError(
                "JSON must contain either 'outgoing_ratio' & 'incoming_ratio' "
                "or the trio 'distribution_trips_ratio', 'regional_outgoing_flow_ratio', "
                "'regional_incoming_flow_ratio'."
            )

        outgoing_ratio = {k: v * r_out for k, v in dist_ratio.items()}
        incoming_ratio = {k: v * r_in for k, v in dist_ratio.items()}

    regions: List[str] = list(outgoing_ratio.keys())
    main_region: str = max(outgoing_ratio.items(), key=lambda kv: kv[1])[0]

    raw_car_ratio = ratios.get("trips_using_car_ratio", "0%")
    if isinstance(raw_car_ratio, str) and raw_car_ratio.endswith("%"):
        car_ratio = float(raw_car_ratio.strip(" %")) / 100.0
    else:
        car_ratio = float(raw_car_ratio)

    with open(path_mapping_json, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    df_in = pd.read_csv(path_input_csv, sep=",")

    os.makedirs(od_folder, exist_ok=True)

    df_rows: List[dict] = []

    for _, hr in df_in.iterrows():
        hour_idx = int(hr["Heures"]) % 24
        total_dep = float(hr["Nombre de départs"])
        total_car = total_dep * car_ratio
        total_car_ceil = math.ceil(total_car)


        outgoing_raw = {r: total_car * outgoing_ratio[r] for r in regions if r != main_region}
        incoming_raw = {r: total_car * incoming_ratio[r] for r in regions if r != main_region}

        outgoing_int = _ceilDict(outgoing_raw)
        incoming_int = _ceilDict(incoming_raw)

        main_main = total_car_ceil - (sum(outgoing_int.values()) + sum(incoming_int.values()))
        main_main = max(main_main, 0)

        row_dict: Dict[str, int] = {"heure": hour_idx}
        for r in regions:
            if r == main_region:
                continue
            row_dict[f"{r}_out"] = outgoing_int[r]
            row_dict[f"{r}_in"] = incoming_int[r]
        row_dict[main_region] = main_main 
        row_dict["total"] = total_car_ceil
        df_rows.append(row_dict)

        start_h, end_h = hour_idx, (hour_idx + 1) % 24
        fname = os.path.join(od_folder, f"OD_{start_h:02d}-{end_h:02d}.txt")

        main_id = _safeInt(mapping[main_region]["id"])

        with open(fname, "w", encoding="utf-8") as fout:
            fout.write("$OR;D2\n")
            fout.write(f"{start_h:02d}.00 {end_h:02d}.00\n")
            fout.write("1.00\n")

            if main_main > 0:
                fout.write(f"{main_id:10d}{main_id:10d}{float(main_main):10.2f}\n")

            for r, cnt in outgoing_int.items():
                if cnt <= 0:
                    continue
                dest_id = _safeInt(mapping[r]["id"])
                fout.write(f"{main_id:10d}{dest_id:10d}{float(cnt):10.2f}\n")

            for r, cnt in incoming_int.items():
                if cnt <= 0:
                    continue
                orig_id = _safeInt(mapping[r]["id"])
                fout.write(f"{orig_id:10d}{main_id:10d}{float(cnt):10.2f}\n")


    col_order = ["heure"]
    for r in regions:
        if r == main_region:
            continue
        col_order.extend([f"{r}_out", f"{r}_in"])
    col_order.extend([main_region, "total"])

    df_out = pd.DataFrame(df_rows, columns=col_order)
    df_out.to_csv(path_output_csv, index=False)

    return df_out
