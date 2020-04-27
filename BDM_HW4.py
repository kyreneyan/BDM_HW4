import sys
import pyspark
import fiona
import fiona.crs
import shapely
import geopandas as gpd
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext

def createIndex(shapefile):

    import rtree
    import fiona.crs
    import geopandas as gpd
    zones = gpd.read_file(shapefile).to_crs(fiona.crs.from_epsg(2263))
    index = rtree.Rtree()

    for idx,geometry in enumerate(zones.geometry):
        index.insert(idx, geometry.bounds)
    return (index, zones)


def findZone(p, index, zones):
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return idx
    return None


def findBorough(p, index, zones):
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return idx
    return None


def processTrips(pid, records):
    import csv
    import pyproj
    import shapely.geometry as geom
    
    # Create an R-tree index
    proj = pyproj.Proj(init="epsg:2263", preserve_units=True)    
    index1, zone1 = createIndex('boroughs.geojson')    
    index2, zone2 = createIndex('neighborhoods.geojson') 
    
    
    # Skip the header
    if pid==0:
        next(records)
    reader = csv.reader(records)
    counts = {}

    for row in reader:
        if len(row)==11:
            try:
                p_pick = geom.Point(proj(float(row[5]), float(row[6]))) #(pick_long,pick_lan)
                p_drop = geom.Point(proj(float(row[9]), float(row[10]))) #(dropoff_long,dropoff_lan)
                borough = findBorough(p_pick, index1, zone1)# index_pick
                neighbor = findZone(p_drop, index2, zone2)# index_drop
        
            if borough and neighbor:
                key=(borough,neighbor)
                counts[key] = counts.get(key, 0) + 1  
            
    return counts.items()

if __name__ == "__main__":
    
    sc = SparkContext()
    input_file = sys.argv[1]
    output_BDM1 = sys.argv[2]
    
    b_geo = 'boroughs.geojson'
    n_geo = 'neighborhoods.geojson'
    boroughs = gpd.read_file(b_geo).to_crs(fiona.crs.from_epsg(2263))
    neighborhoods = gpd.read_file(n_geo).to_crs(fiona.crs.from_epsg(2263))
    b_name = list(boroughs['boroname'])
    n_name= list(neighborhoods['neighborhood'])

    rdd = sc.textFile(input_file)
    counts = rdd.mapPartitionsWithIndex(processTrips) \
                .reduceByKey(lambda x,y: x+y) \
                .map(lambda x: ((b_name[x[0][0]]), (x[1],n_name[x[0][1]]))) \
                .groupByKey() \
                .map(lambda x: (x[0],sorted(x[1],reverse = True)[:3])) \
                .saveAsTextFile(output_BDM1)
