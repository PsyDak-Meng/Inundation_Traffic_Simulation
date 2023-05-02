import numpy as np
import pandas as pd
import geopandas as gpd
import geopy as gpy
from geopy.extra.rate_limiter import RateLimiter
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import OrderedDict
import rasterio
from rasterio.plot import show
import rasterio.mask
import networkx as nx
from shapely.geometry import Polygon, Point
import warnings
import momepy
import libpysal
import math
import contextily as cx


# preprocess incident data
def rawData(address = './data/ambulance/virginiaBeach_ambulance_timeData.csv'):
    data = pd.read_csv(address)

    data['CallDateTime'] = pd.to_datetime(data['Call Date and Time'], format = "%Y-%m-%d %H:%M:%S")
    data['EntryDateTime'] = pd.to_datetime(data['Entry Date and Time'], format = "%Y-%m-%d %H:%M:%S")
    data['DispatchDateTime'] = pd.to_datetime(data['Dispatch Date and Time'], format = "%Y-%m-%d %H:%M:%S")
    data['EnRouteDateTime'] = pd.to_datetime(data['En route Date and Time'], format = "%Y-%m-%d %H:%M:%S")
    data['OnSceneDateTime'] = pd.to_datetime(data['On Scene Date and Time'], format = "%Y-%m-%d %H:%M:%S")
    data['CloseDateTime'] = pd.to_datetime(data['Close Date and Time'], format = "%Y-%m-%d %H:%M:%S")

    data['Country'] = 'USA'
    data['Address'] = data['Block Address'].str.cat([
        pd.Series(', ', index = data.index),
        data['City'], 
        pd.Series(', ', index = data.index),
        data['State'], 
        pd.Series(', ', index = data.index),
        data['Country']
    ], join="left")

    data['DispatchTime'] = (data['DispatchDateTime'] - data['CallDateTime']).astype("timedelta64[s]")
    data['EnRouteTime'] = (data['EnRouteDateTime'] - data['CallDateTime']).astype("timedelta64[s]")
    data['TravelTime'] = (data['OnSceneDateTime'] - data['EnRouteDateTime']).astype("timedelta64[s]")
    data['ResponseTime'] = (data['OnSceneDateTime'] - data['CallDateTime']).astype("timedelta64[s]")
    data['HourInDay'] = data['CallDateTime'].dt.hour
    data['DayOfWeek'] = data['CallDateTime'].dt.dayofweek
    
    return data

def addOrigin(data, rescueAddress):
    rescue = pd.read_csv(rescueAddress) 
    rescue = gpd.GeoDataFrame(rescue, geometry = gpd.points_from_xy(rescue['lon'], rescue['lat']))
    data = data[data['Rescue Squad Number'].isin(rescue.Number.to_list())]
    data = data.merge(rescue, how = 'left', left_on = 'Rescue Squad Number', right_on = 'Number')
    
    data = gpd.GeoDataFrame(data)
    data = data.set_index('CallDateTime')
    data = data.sort_index()
    
    data['geometry'] = gpd.GeoSeries(data['geometry'], crs = 'EPSG:4326', index = data.index)
    return data

def _organizeData(data):
    data['CallDateTime'] = data.index
    data = data.reset_index(drop = True)
    data = data.loc[:, [
        'Call Priority',
        'CallDateTime', 'EntryDateTime', 'DispatchDateTime', 'EnRouteDateTime', 'OnSceneDateTime', 'CloseDateTime',
        'DispatchTime', 'EnRouteTime', 'TravelTime', 'ResponseTime', 'HourInDay', 'DayOfWeek', 
        'Rescue Squad Number', 'geometry', 
        'Address', 'IncidentFullInfo', 'IncidentPoint',] ]
    data = data.rename(columns = {"geometry": "RescueSquadPoint", 
                                  "Address": "IncidentAddress", 
                                  'Rescue Squad Number': 'RescueSquadNumber',
                                  'Call Priority': 'CallPriority'})
    data.set_geometry("IncidentPoint")
    return data

def geoCoding(data, saveAddress):
    locator = gpy.geocoders.ArcGIS()
    geocode = RateLimiter(locator.geocode, min_delay_seconds = 0.1)

    data['IncidentFullInfo'] = data['Address'].apply(geocode)
    data['IncidentCoor'] = data['IncidentFullInfo'].apply(lambda loc: tuple(loc.point) if loc else None)
    data['IncidentFullInfo'] = data['IncidentFullInfo'].astype(str)
    data[['IncidentLat', 'IncidentLon', 'IncidentElevation']] = pd.DataFrame(data['IncidentCoor'].tolist(), index = data.index)
    data['IncidentPoint'] = gpd.GeoSeries(gpd.points_from_xy(y = data.IncidentLat, x = data.IncidentLon), index = data.index, crs = "EPSG:4326")
    
    data = _organizeData(data)
    data.to_csv(saveAddress, index = False)
    return data

def _string2Points(data, column, crs, index):
    x = [float(location.replace('POINT (', '').replace(')', '').split(' ')[0]) for location in list(data[column].values)]
    y = [float(location.replace('POINT (', '').replace(')', '').split(' ')[1]) for location in list(data[column].values)]
    return gpd.GeoSeries(gpd.points_from_xy(x = x, y = y), crs = crs, index = index)

def reLoadData(address):
    data = pd.read_csv(address, index_col = 'CallDateTime')
    data.index = pd.to_datetime(data.index, format = "%Y-%m-%d %H:%M:%S")
    data = gpd.GeoDataFrame(data)
    data['RescueSquadPoint'] = _string2Points(data, 'RescueSquadPoint', "EPSG:4326", data.index)
    data['IncidentPoint'] = _string2Points(data, 'IncidentPoint', "EPSG:4326", data.index)
    return data
    

# preprocess road data
def _moveDuplicates(joined):
    # move duplicates, keep the row with higher width
    u, c = np.unique(joined.OBJECTID_left.values, return_counts = True)
    duplicates = u[c > 1]
    joined_noDuplicates = joined.copy()
    for dup in duplicates:
        du = joined[joined.OBJECTID_left == dup]
        joined_noDuplicates = joined_noDuplicates[joined_noDuplicates.OBJECTID_left != dup]
        duOne = du[du.aveWidth == du.aveWidth.max()]
        joined_noDuplicates = pd.concat([joined_noDuplicates, duOne])
    return joined_noDuplicates.sort_values(by = ['OBJECTID_left'])

def _createSurface4roads(roads, roadSurfaces):
    # USE: create a geoDataFrame containing the column of average width and full polygon (might include multiple road segments) for each road
    # spatial join road lines and surfaces
    if roads.crs != roadSurfaces.crs:
        return 'crs not consistent'
    roadSurfaces['aveWidth'] = roadSurfaces.Shapearea / roadSurfaces.Shapelen
    roads['midpoint'] = roads.geometry.interpolate(0.5, normalized = True)
    roads = roads.set_geometry("midpoint", crs = roadSurfaces.crs)
    roads = roads.rename(columns = {"geometry": "line"})
    joined = roads.sjoin(roadSurfaces, how = "left", predicate = 'within')
    # move duplicates/nan 
    joined_updated = _moveDuplicates(joined)
    joined_updated.loc[np.isnan(joined_updated.aveWidth), ['aveWidth']] = joined_updated.aveWidth.mean() # assign width to missing roads
    # attach roadSurface polygons
    joined_updated['OBJECTID_right'] = joined_updated.OBJECTID_right.astype('Int64')
    roadSurfaces_temp = roadSurfaces[['OBJECTID', 'geometry']].rename({'OBJECTID': 'OBJECTID_right', 'geometry': 'surfacePolygon'}, axis = 1)
    roadSurfaces_temp.loc[len(roadSurfaces_temp)] = [np.nan, Polygon()]
    roadSurfaces_temp.OBJECTID_right = roadSurfaces_temp.OBJECTID_right.astype('Int64')
    joined_updated = joined_updated.merge(roadSurfaces_temp, how = 'left', on = 'OBJECTID_right')
    joined_updated = joined_updated.set_geometry('surfacePolygon').set_crs(roadSurfaces.crs)
    return joined_updated

def readRoads(roadAddress):
    roads = gpd.read_file(roadAddress)
    roads = roads.loc[-roads['geometry'].duplicated(), :]
    roads['OBJECTID'] = list(range(1, len(roads) + 1))
    roads = roads.reset_index(drop = True)
    return roads

def makeSurface4Lines(roads, surfaceAddress, scale = 2.7):
    roadSurfaces = gpd.read_file(surfaceAddress)
    surfaces4roads = _createSurface4roads(roads, roadSurfaces)

    roads['aveWidth'] = surfaces4roads.aveWidth
    roads['scaledRadius'] = roads['aveWidth'] / 2 * scale
    roads['buffers'] = roads.geometry.buffer(roads['scaledRadius'])
    roads['buffersUnscaled'] = roads.geometry.buffer(roads['aveWidth'] / 2 * 1.5) # may be some errors in raw data, roads look good when multiply by 1.5 
    roads = roads.rename(columns = {"geometry": "line"})
    roads = roads.set_geometry('buffers', crs = roadSurfaces.crs)

    roads['surface'] = [road.intersection(surface) if not road.intersection(surface).is_empty else roadUnscaled \
                            for road, surface, roadUnscaled in zip(roads.geometry, surfaces4roads.geometry, roads.buffersUnscaled)]
    roads = roads.set_geometry('surface', crs = roadSurfaces.crs)
    return roads

def _inundationCutter(inundation, cut, all_touched, invert, addr = './data/inundation/croppedByBridge/croppedByBridge.tif'):
    if inundation.crs != cut.crs:
        return 'crs not consistent'
    # mask the inundation using bridges shp, remove the inundation under bridges
    out_array, _ = rasterio.mask.mask(inundation, cut.geometry, all_touched = all_touched, invert = invert)
    inundation_cropped = rasterio.open(
        addr,
        'w+',
        **inundation.meta
    )
    inundation_cropped.write(out_array)
    return inundation_cropped

def _getMaxWaterDepth(roadGeometry, inundation):
    # roadGeometry should be series, inundation is raster
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inundationOnRoad, _ = rasterio.mask.mask(inundation, roadGeometry)
        inundationOnRoad = np.where(inundationOnRoad == inundation.nodata, - inundationOnRoad, inundationOnRoad)
    return np.max(inundationOnRoad)

def getWaterDepthOnRoads(roads, inundationAddress, inundationCutSaveAddress):
    inundation = rasterio.open(inundationAddress)
    roads_updated_4getInundation = roads.copy().to_crs(str(inundation.crs))
    inundation_cutByRoads = _inundationCutter(inundation, roads_updated_4getInundation, False, False, inundationCutSaveAddress)

    roads['waterDepth'] = roads_updated_4getInundation.loc[:, ['surface']] \
        .apply(lambda x: _getMaxWaterDepth(x, inundation_cutByRoads), axis = 1, raw = True).replace(-inundation.nodata, 0)   
    return roads

# visualzation
def showRoadsInundation(inundation):
    fig = plt.figure(figsize = (100, 50))
    ax = fig.add_subplot()
    ax = show(inundation, ax = ax, cmap = 'pink')
    roads.plot(ax = ax)
    plt.show()

def showInundation(inundation):
    plt.imshow(inundation.read()[0], cmap = 'hot')
    plt.colorbar()
    plt.show()
    
def showBridgesInundation():
    fig = plt.figure(figsize = (100, 50))
    ax = fig.add_subplot()
    ax = show(inundation, ax = ax, cmap = 'pink')
    bridges.plot(ax = ax)
    plt.show()

def showLinesSurfaces_withBounds(roads, roadSurfaces, bounds):
    fig = plt.figure(figsize = (100, 50))
    ax = fig.add_subplot()
    roadsBounded = roads.cx[bounds[0]: bounds[1], bounds[2]: bounds[3]]
    roadSurfacesBounded = roadSurfaces.cx[bounds[0]: bounds[1], bounds[2]: bounds[3]]
    roadsBounded.plot(ax = ax, color='red')
    roadSurfacesBounded.plot(ax = ax)
    plt.show()    

def showMidpointLineSurface(roads, roadSurfaces):
    fig = plt.figure(figsize = (400, 200))
    ax = fig.add_subplot()
    roads.line.plot(ax = ax, linewidth = .75, zorder = 0)
    # roads.midpoint.plot(ax = ax, zorder = 0)
    roadSurfaces.geometry.plot(ax = ax, color = 'red', zorder = 0)
    plt.show()

def getGlobalBounds(gpd):
    # get global bounds of a geopandas df
    xmin = gpd.bounds.minx.min()
    xmax = gpd.bounds.maxx.max()
    ymin = gpd.bounds.miny.min()
    ymax = gpd.bounds.maxy.max()    
    return xmin, xmax, ymin, ymax

def getMiddleBounding(bounds, percent = 0.05):
    xmin = bounds[0]
    xmax = bounds[1]
    ymin = bounds[2]
    ymax = bounds[3]
    xminNew = xmin + ((xmax - xmin) * ((1 - percent) / 2))
    xmaxNew = xminNew + (xmax - xmin) * percent
    yminNew = ymin + ((ymax - ymin) * ((1 - percent) / 2))
    ymaxNew = yminNew + (ymax - ymin) * percent   
    return xminNew, xmaxNew, yminNew, ymaxNew
    

# create graph
def showGraphRoads(roads, graph):
    f, ax = plt.subplots(1, 3, figsize = (100, 50), sharex = True, sharey = True)
    for i, facet in enumerate(ax):
        facet.set_title(("Streets", "Primal graph", "Overlay")[i])
        facet.axis("off")

    roads.plot(color='#e32e00', ax = ax[0])
    nx.draw(graph, {key: [value.x, value.y] for key, value in nx.get_node_attributes(graph, 'midpoint').items()}, ax = ax[1], node_size = 1)
    roads.plot(color = '#e32e00', ax = ax[2], zorder = -1)
    nx.draw(graph, {key: [value.x, value.y] for key, value in nx.get_node_attributes(graph, 'midpoint').items()}, ax = ax[2], node_size = 1)

def roads2Graph(roads):
    roads4graph = roads.copy()
    roads4graph['geometry'] = roads4graph['line'].to_crs(roads4graph.line.crs)
    roads4graph = roads4graph.set_geometry("geometry")
    graph = momepy.gdf_to_nx(roads4graph, approach = 'dual', multigraph = False, angles = False)
    graph = nx.relabel_nodes(graph, nx.get_node_attributes(graph, 'OBJECTID'))
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['weight'] = (graph.nodes[edge[0]]['SHAPElen'] + graph.nodes[edge[1]]['SHAPElen']) / 2    
    return graph

# read rescue
def readRescue(rescueAddress, crs, roads):
    rescue = pd.read_csv(rescueAddress) 
    rescue = gpd.GeoDataFrame(rescue, geometry = gpd.points_from_xy(rescue['lon'], rescue['lat'])).set_crs(crs).to_crs(roads.crs) 
    rescue['OBJECTID_nearestRoad'] = rescue.geometry.apply(lambda x: x.distance(roads.line).sort_values().index[0] + 1)    
    return rescue
    

# additional info
def assignGraphEdge(data, roads, inColumn, outColumn1, outColumn2, max_distance = 500):
    # NOTE: there could be null value if no road is within the scope of search for a location
    roadLines = roads.loc[:, ['OBJECTID', 'line']].set_geometry('line')
    locations = data.loc[:, [inColumn]].set_geometry(inColumn).to_crs(roadLines.crs)
    match = locations.sjoin_nearest(roadLines, how = 'left', max_distance = max_distance, distance_col = 'distance')
    match = match.reset_index().drop_duplicates(subset = ['CallDateTime']).set_index('CallDateTime')
    data[outColumn1] = match['OBJECTID']
    data[outColumn2] = match['distance']
    return data

def _nearestRescue4Incidents(data, rescue):
    # find nearest rescues for all incidents
    incidents = data.DestinationID.values
    voronoi = nx.voronoi_cells(graph, set(rescue.OBJECTID_nearestRoad.unique()), weight = 'weight')
    nearestRescue = []
    for incident in incidents:
        len1 = len(nearestRescue)
        if np.isnan(incident):
            nearestRescue.append(np.nan)
        else:
            for key, value in voronoi.items():
                if int(incident) in list(value):
                    if key == 'unreachable':
                        nearestRescue.append(np.nan)
                    else:
                        nearestRescue.append(key)
                    break 
        len2 = len(nearestRescue)
        if len2 == len1:
            print(incident, 'not in any')
    return nearestRescue

def _generateDistDf(rescue, graph):
    nodeList = range(1, len(list(graph.nodes())) + 1)
    df = pd.DataFrame(nodeList, index = nodeList, columns =['NodeNames'])
    for res in rescue.values:
        resName = res[0]
        resRoadNumber = res[-1]
        distanceDict = nx.single_source_dijkstra_path_length(graph, resRoadNumber, weight='weight')
        orderedResRoadNumber = OrderedDict(sorted(distanceDict.items()))
        orderedResRoadNumberDf = pd.DataFrame.from_dict(orderedResRoadNumber, orient = 'index', columns = ['from' + resName])
        orderedResRoadNumberDf = orderedResRoadNumberDf.reset_index()
        df = df.merge(orderedResRoadNumberDf, how = 'left', left_on = 'NodeNames', right_on = 'index').drop(columns = 'index')
    return df

def _obedianceOfShortestPrinciple(Series, distanceDataFrame):
    DestinationID = Series.DestinationID
    RescueSquadNumber = Series.RescueSquadNumber
    if len(distanceDataFrame[distanceDataFrame.NodeNames == DestinationID]) != 0:
        # NOTE:some incidents are not considered because no road around them
        allDist = list(np.sort(distanceDataFrame[distanceDataFrame.NodeNames == DestinationID].values[0][1:]))
        realDist = distanceDataFrame[distanceDataFrame.NodeNames == DestinationID]['from' + RescueSquadNumber].values[0]
        if np.isnan(realDist):
            # NOTE: some roads are disconnected even in normal time
            realDistRank = np.nan
            realDistIncreaseRatio = np.nan
        else:
            realDistRank = allDist.index(realDist) + 1
            if allDist[0] == 0:
                # NOTE: in case the incident is just beside the rescue station, set the dist to 1
                allDist[0] = 1
            realDistIncreaseRatio = realDist / allDist[0]
    else:
        realDistRank = np.nan
        realDistIncreaseRatio = np.nan
    return realDistRank, realDistIncreaseRatio

def _shortestRouteLength_slowLegacy(row, graph, ifPrintError = False):
    try:
        length = nx.dijkstra_path_length(graph, row.OriginRoadID, row.DestinationID, weight = 'weight')
    except BaseException as ex:
        if ifPrintError == True:
            print(ex)
        length = np.nan
    return length

def _shortestRouteLength(s, distanceDataFrame):
    if np.isnan(s.DestinationID):
        return np.nan
    else:
        return distanceDataFrame[distanceDataFrame.NodeNames == s.DestinationID]['from' + s.RescueSquadNumber].values[0]

def nearestRescueStation(data, rescue):
    # find nearest rescue station
    data['NearestRescue'] = _nearestRescue4Incidents(data, rescue)
    data = data.merge(rescue.loc[:, ["OBJECTID_nearestRoad", "Number"]], how = 'left', left_on = "NearestRescue", right_on = 'OBJECTID_nearestRoad')
    data = data.drop(columns = 'OBJECTID_nearestRoad').rename(columns={"Number": "NearestRescueNumber"})
    return data

def nearnessObediance(data, rescue, graph):
    # find the top nearest rescue stations
    distanceDataFrame = _generateDistDf(rescue, graph)
    obediance = data.apply(_obedianceOfShortestPrinciple, distanceDataFrame = distanceDataFrame, axis = 1, result_type = 'expand')
    data['NearestOrder'] = obediance[0]
    data['DisobediancePathIncrease'] = obediance[1]
    return data

def assumedAveSpeed(data, rescue, graph):
    # calculate shortest path length and ave speed
    distanceDataFrame = _generateDistDf(rescue, graph)
    data['AssumedRouteLength'] = data.apply(_shortestRouteLength, distanceDataFrame = distanceDataFrame, axis = 1)
    data['AverageSpeed'] = data['AssumedRouteLength'] / data['TravelTime']
    return data