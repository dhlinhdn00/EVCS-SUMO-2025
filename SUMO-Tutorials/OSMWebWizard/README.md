# OpenStreetMap Web Wizard Tutorial

## Getting Started Commands

```sh
which sumo
echo $SUMO_HOME

cd $SUMO_HOME/tools
python osmWebWizard.py
```

## OSM Import Configuration

### Options:
- **Add Polygon (Enabled by default)**: Imports all road types (car roads, bicycle lanes, footpaths, railways, etc.).
- **Left-hand Traffic**: Builds the network with left-hand traffic rules (enable manually if necessary).
- **Car-only Network**: Imports only roads that allow passenger cars, reducing network size and complexity.
- **Import Public Transport**: Imports bus and train stops and generates public transport vehicles based on OSM routes.
- **Bicycles**: Adds bicycle lanes if available in OSM data.
- **Pedestrians**: Generates sidewalks and pedestrian crossings.

## Transport Mode Configuration

In the **Demand Generation** panel, transport modes can be enabled/disabled via checkboxes. The OSM Web Wizard generates random demand based on a probability distribution influenced by two parameters:

- **Through Traffic Factor**: Determines how likely edges at the simulation boundary are chosen as departure/arrival points. A higher value increases through traffic.
- **Count Parameter**: Defines the number of vehicles per hour per lane-kilometer.

### Example Calculation:
If the network has 3 edges, totaling **5 km**, with **2 lanes each**, and **Count = 90**, then:

```
5 × 2 × 90 = 900 vehicles/hour
```
This corresponds to `randomTrips p=4`, meaning a new vehicle is inserted every **4 seconds**.

## Road-Type Selection

In the **Road-Type** tab of the OSM Web Wizard, you can choose which road types to download and render.

- For **major traffic simulations**, select only `motorways`, `primary`, `secondary`, and `tertiary` to reduce OSM file size.
- By default, all road types are selected.
- Unchecking **"Add Polygon"** in Demand Generation disables non-road objects (e.g., buildings, waterways), further reducing file size.

## Generating and Running the Scenario

1. Click **Generate Scenario** to automatically create the simulation.
2. The process takes **seconds to minutes**, depending on the scenario size.
3. Once done, **sumo-gui** launches.
4. Start the simulation by clicking **Play**.

## Saving Locations

```
/home/meos/SUMO
```

- Format for child folder: `yyyy-mm-dd-hh-mm-ss`

