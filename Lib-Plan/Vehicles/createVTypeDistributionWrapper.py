import sys
import csv
import re
import random
from xml.dom import minidom
from xml.dom.minidom import Document, Element
from sumolib.options import ArgumentParser
from sumolib.vehicletype import CreateVehTypeDistribution


def addChild(self, tag, attrib):
    elem = self.ownerDocument.createElement(tag)
    for key, value in attrib.items():
        elem.setAttribute(key, str(value))
    self.appendChild(elem)
    return elem

Element.addChild = addChild


def get_options(args=None):
    ap = ArgumentParser()
    ap.add_argument("configFile", category="input", help="File path of the config file which defines the car-following parameter distributions")
    ap.add_argument("-o", "--output-file", category="output", dest="outputFile", default="vTypeDistributions.add.xml",
                    help="File path of the output file (if already exists, the script tries to insert the distribution node)")
    ap.add_argument("-n", "--name", dest="vehDistName", default="vehDist",
                    help="Alphanumerical ID used for the created vehicle type distribution")
    ap.add_argument("-s", "--size", type=int, default=100, dest="vehicleCount",
                    help="Number of vTypes in the distribution")
    ap.add_argument("-d", "--decimal-places", type=int, default=3, dest="decimalPlaces",
                    help="Number of decimal places for numeric attribute values")
    ap.add_argument("--resampling", type=int, default=100, dest="nrSamplingAttempts",
                    help="Number of attempts to resample a value until it lies in the specified bounds")
    ap.add_argument("--seed", type=int, help="Random seed", default=42)
    ap.add_argument("--evFile", help="Optional XML file that contains electric vehicle definitions")
    return ap.parse_args(args)


def readConfigFile(options):
    filePath = options.configFile
    result = dict()
    floatRegex = [r'\s*(-?[0-9]+(\.[0-9]+)?)\s*']
    distSyntaxes = {
        'normal': r'normal\(%s\)' % (",".join(2 * floatRegex)),
        'lognormal': r'lognormal\(%s\)' % (",".join(2 * floatRegex)),
        'normalCapped': r'normalCapped\(%s\)' % (",".join(4 * floatRegex)),
        'uniform': r'uniform\(%s\)' % (",".join(2 * floatRegex)),
        'gamma': r'gamma\(%s\)' % (",".join(2 * floatRegex))
    }
    with open(filePath) as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) < 2:
                continue
            attName = row[0].strip()
            isParam = False
            if attName.lower() == "param":
                isParam = True
                if len(row) < 3:
                    continue
                attName = row[1].strip()
                attValue = row[2].strip()
            else:
                attValue = row[1].strip()
            distFound = False
            distAttr = None
            for distName, distSyntax in distSyntaxes.items():
                items = re.findall(distSyntax, attValue)
                if items:
                    distFound = True
                    distPar1 = float(items[0][0])
                    distPar2 = float(items[0][2])
                    if distName == 'normal':
                        distAttr = {"mu": distPar1, "sd": distPar2}
                    elif distName == 'lognormal':
                        distAttr = {"mu": distPar1, "sd": distPar2}
                    elif distName == 'normalCapped':
                        cutLow = float(items[0][4])
                        cutHigh = float(items[0][6])
                        distAttr = {"mu": distPar1, "sd": distPar2, "min": cutLow, "max": cutHigh}
                    elif distName == 'uniform':
                        distAttr = {"a": distPar1, "b": distPar2}
                    elif distName == 'gamma':
                        distAttr = {"alpha": distPar1, "beta": distPar2}
                    attValue = None
                    break
            limits = None
            if len(row) == 3 and not distFound:
                limitValue = row[2].strip()
                items = re.findall(r'\[\s*(-?[0-9]+(\.[0-9]+)?)\s*,\s*(-?[0-9]+(\.[0-9]+)?)\s*\]', limitValue)
                if items:
                    lowerLimit = float(items[0][0])
                    upperLimit = float(items[0][2])
                    limits = (lowerLimit, upperLimit)
            result[attName] = {
                "name": attName,
                "is_param": isParam,
                "distribution": None if not distFound else distName,
                "distribution_params": distAttr,
                "bounds": limits,
                "attribute_value": attValue
            }
    return result


def add_EV_vehicles_to_distribution(dist_elem, evFile, defaults):
    evDoc = minidom.parse(evFile)
    for vtype in evDoc.getElementsByTagName("vType"):
        for attr, val in defaults.items():
            if not vtype.hasAttribute(attr):
                vtype.setAttribute(attr, str(val))
        imported = dist_elem.ownerDocument.importNode(vtype, deep=True)
        dist_elem.appendChild(imported)


def prettify_no_blank_lines(elem):
    rough = elem.toprettyxml(indent="  ")
    return "\n".join([line for line in rough.split("\n") if line.strip()])


def main(options):
    random.seed(options.seed)
    dist_creator = CreateVehTypeDistribution(
        seed=options.seed,
        size=options.vehicleCount,
        name=options.vehDistName,
        resampling=options.nrSamplingAttempts,
        decimal_places=options.decimalPlaces
    )

    params = readConfigFile(options)
    for param_dict in params.values():
        dist_creator.add_attribute(param_dict)

    if ("speedFactor" in params) and ("speedDev" not in params):
        dist_creator.add_attribute({
            "name": "speedDev",
            "is_param": False,
            "distribution": None,
            "distribution_params": None,
            "bounds": None,
            "attribute_value": "0"
        })
        print("Warning: Setting speedDev to 0 because only speedFactor is given.", file=sys.stderr)

    doc = Document()
    root = doc.createElement("additional")
    root.setAttribute("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.setAttribute("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/additional_file.xsd")
    doc.appendChild(root)

    # Add vehicle type distribution
    dist_elem = doc.createElement("vTypeDistribution")
    dist_elem.setAttribute("id", options.vehDistName)
    root.appendChild(dist_elem)
    dist_creator.create_veh_dist(dist_elem)

    # Add electric vehicles into vTypeDistribution
    if options.evFile:
        defaults = {
            "carFollowModel": "EIDM",
            "tau": 1.0,
            "speedDev": 0.1,
            "length": 4.0,
            "width": 1.7,
            "height": 1.4
        }
        add_EV_vehicles_to_distribution(dist_elem, options.evFile, defaults)

    xml_str = prettify_no_blank_lines(doc)
    xml_str = xml_str.replace("</additional>", "")  # optional: remove closing tag
    with open(options.outputFile, "w") as f:
        f.write(xml_str)


if __name__ == "__main__":
    try:
        main(get_options())
    except ValueError as e:
        sys.exit(e)
