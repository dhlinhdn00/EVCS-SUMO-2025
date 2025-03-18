# Aix-Marseille-Provence-Metropole

## OSM
https://wiki.openstreetmap.org/wiki/FR:France

## Overpass Turbo

In overpass-turbo.eu, I usse this query for find Bouches-du-Rhône, Var and  (admin_level=6) and get relations which have admin_level=8:

[out:json][timeout:60];
area
    ["name"="Bouches-du-Rhône"]
    ["boundary"="adminstrative"]
    ["admin_level"="6"]->.dep13;
relation
    ["boundary"="adminstrative"]
    ["admin_level"="8"](area.dep13);
(._;>;);
out_meta;


[out:json][timeout:60];
area
  ["name"="Var"]
  ["boundary"="administrative"]
  ["admin_level"="6"]->.dep;

relation
  ["boundary"="administrative"]
  ["admin_level"="8"]
  (area.dep);
(._;>;);
out meta;

[out:json][timeout:60];
area
  ["name"="Vaucluse"]
  ["boundary"="administrative"]
  ["admin_level"="6"]->.dep;

relation
  ["boundary"="administrative"]
  ["admin_level"="8"]
  (area.dep);
(._;>;);
out meta;



## French Administrative System

| **Level**             | **French Name**                    | **Number** | **OSM Admin Level** |
|----------------------|---------------------------------|------------|------------------|
| **National**        | République Française           | 1          | 2                |
| **Region**         | Région                         | 18         | 4                |
| **Department**     | Département                    | 101        | 6                |
| **Administrative District** | Arrondissement administratif | ~330       | 7                |
| **Municipality**   | Commune                        | ~35,000    | 8                |
| **Intermunicipal Entity** | Métropole, Communauté Urbaine | ~1,200     | 7 or 8           |
| **Municipal District** | Arrondissement Municipal      | 3 cities   | 9 or 10          |


See more: https://www.banatic.interieur.gouv.fr/consultation


ogr2ogr -f "ESRI Shapefile" AMP_final.shp AMP_final.geojson

polyconvert --shapefile-prefixes AMP_final -n dummy.net.xml -o AMP.poly.xml


# highway motorway