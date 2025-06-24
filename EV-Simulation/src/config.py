"""
EVCS-SUMO-2025
Simulation of Charging Needs of Electrical Car Drivers using SUMO

Author
------
DAO Hoai Linh  
LIS-LAB, Aix-Marseille University
"""
import os
import glob
from pathlib import Path
import subprocess
import sys
import shutil

import itertools
import random
import pandas as pd
import math
import csv
import re
from collections import OrderedDict, defaultdict
from typing import List, Dict, Tuple
from collections import Counter

import traci
import sumolib
import xml.etree.ElementTree as ET
from pyproj import Transformer, CRS
import networkx as nx
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.errors import TopologicalError

# export LD_LIBRARY_PATH=~/Libs/libnsl
# export SUMO_HOME=~/Envs/sumo-env/lib/python3.10/site-packages/sumo
os.environ["LD_LIBRARY_PATH"] = os.path.expanduser("~/Libs/libnsl")
os.environ["SUMO_HOME"] = os.path.expanduser("~/Envs/sumo-env/lib/python3.10/site-packages/sumo")

# SUMO_TOOLS Definitions
SUMO_TOOLS_DIR = Path("/home/hoai-linh.dao/Envs/sumo-env/lib/python3.10/site-packages/sumo/tools")
CONTINUOSREROUTER_PY = SUMO_TOOLS_DIR / "generateContinuousRerouters.py"
REROUTER_PY = SUMO_TOOLS_DIR / "generateRerouters.py"
NETCHECK_PY = SUMO_TOOLS_DIR / "net/netcheck.py"
RANDOMTRIPS_PY = SUMO_TOOLS_DIR / "randomTrips.py"
FINDALLROUTES_PY = SUMO_TOOLS_DIR / "findAllRoutes.py"
PLOTXMLATTRIBUTES_PY = SUMO_TOOLS_DIR / "visualization/plotXMLAttributes.py"
PLOTTRAJECTORIES_PY = SUMO_TOOLS_DIR / "plot_trajectories.py"
PLOTNETDUMP_PY = SUMO_TOOLS_DIR / "visualization/plot_net_dump.py"
PLOTNETSPEED_PY = SUMO_TOOLS_DIR / "visualization/plot_net_speed.py"
PLOTNETTRAFFICLIGHTS_PY = SUMO_TOOLS_DIR / "visualization/plot_net_trafficLights.py"
PLOTSUMMARY_PY = SUMO_TOOLS_DIR / "visualization/plot_summary.py"
PLOTTRIPINFODISTRIBUTIONS_PY = SUMO_TOOLS_DIR / "visualization/plot_tripinfo_distributions.py"
PLOTCSVTIMELINE_PY = SUMO_TOOLS_DIR / "visualization/plot_csv_timeline.py"
PLOTCSVPIE_PY = SUMO_TOOLS_DIR / "visualization/plot_csv_pie.py"
PLOTCSVBARS_PY = SUMO_TOOLS_DIR / "visualization/plot_csv_bars.py"
MACROUTPUT_PY = SUMO_TOOLS_DIR / "visualization/marcoOutput.py"
ROUTESTATS_PY = SUMO_TOOLS_DIR / "route/routeStats.py"
ROUTECHECK_PY = SUMO_TOOLS_DIR / "route/routecheck.py"
TLSCOORDINATOR_PY = SUMO_TOOLS_DIR / "tlsCoordinator.py"
TLSCYCLEADAPTATION_PY = SUMO_TOOLS_DIR / "tlsCycleAdaptation.py"

TAZ_IDS = {
    'marseille': '1',
    'aix-en-provence': '2',
    'est-etang-de-berre': '3',
    'nord-ouest': '4',
    'ouest-etang-de-berre': '5',
    'sud-est': '6',
    'outside': '99'
}

BORDER_RATIO = 0.40
REAL_ORIGIN   = 'marseille'

CAR_PREFIX = "carDist"              
EV_BRANDS = ["Renault", "Tesla", "Citroen", "Peugeot", "Dacia", "Volkswagen", "BMW", "Fiat", "KIA"]

EV_RATIO = 0.20

DIST_ID = "vehDist"

# Page 11
INCOMING_RATIO = 178729/(178729 + 174729)
OUTGOING_RATIO = 174729/(178729 + 174729)
INCOMING_RATIO, OUTGOING_RATIO

# Page 14 + Page 15
TRIPS_RATIO_0 = 1 # default
TRIPS_RATIO_1 = 0.40 # Marseille 
TRIPS_RATIO_2 = 0.41 # Marseille Bassin
TRIPS_RATIO_3 = 0.52 # AMP Bassin = CEREMA
TRIPS_RATIO_4 = 0.10 # test
TRIPS_RATIO_5 = 0.01 # fast test