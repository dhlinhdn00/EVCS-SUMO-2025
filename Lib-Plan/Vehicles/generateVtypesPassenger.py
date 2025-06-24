# Reference Link: https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#car-following_models
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

def perturb(value, delta, min_val=None, max_val=None, digits=2):
    val = value + random.uniform(-delta, delta)
    if min_val is not None:
        val = max(val, min_val)
    if max_val is not None:
        val = min(val, max_val)
    return round(val, digits)

def generatePassengerVtypesXML(filename="random_passenger_vtypes.xml", n_types=10):
    root = ET.Element("routes")

    car_follow_models = ["Krauss", "IDM", "W99"]
    gui_shapes = [
        "passenger/sedan", "passenger/hatchback", "passenger/wagon", "passenger/van"
    ]
    colors = [
        "1,0,0", "0,1,0", "0,0,1", "1,1,0", "1,0,1", "0,1,1",
        "0.5,0.5,0", "0.2,0.8,0.2", "0.9,0.3,0.3"
    ]

    for i in range(n_types):
        model = random.choice(car_follow_models)
        vtype = ET.SubElement(root, "vType")
        vtype.set("id", f"carType{i+1}")
        vtype.set("vClass", "passenger")
        vtype.set("carFollowModel", model)

        vtype.set("accel", str(perturb(2.6, 0.8, 1.0, 4.0)))
        vtype.set("decel", str(perturb(4.5, 1.0, 2.0, 6.0)))
        vtype.set("sigma", str(perturb(0.5, 0.3, 0.1, 1.0)))
        vtype.set("tau", str(perturb(1.0, 0.5, 0.5, 2.5)))
        vtype.set("length", str(perturb(5.0, 1.5, 3.0, 6.5)))
        vtype.set("minGap", str(perturb(2.5, 0.5, 1.0, 4.0)))
        vtype.set("maxSpeed", str(perturb(13.89, 3.0, 10.0, 20.0)))
        vtype.set("mass", str(perturb(1500, 400, 1000, 2500)))
        vtype.set("width", str(perturb(1.8, 0.2, 1.4, 2.2)))
        vtype.set("height", str(perturb(1.5, 0.2, 1.2, 2.0)))
        vtype.set("speedFactor", f"normc({round(random.uniform(0.9, 1.2), 2)},0.1,0.6,1.5)")
        vtype.set("guiShape", random.choice(gui_shapes))
        vtype.set("color", random.choice(colors))

    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    with open(filename, "w") as f:
        f.write(pretty_xml)

    print(f"XML saved to {filename}")