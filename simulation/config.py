from road_node import RoadNode
import random
import json

def create_road_nodes(road_graph):
    road_nodes_list = []
    for node_id, neighbors in road_graph.items():
        # Create a RoadNode object
        road_node = RoadNode(node_id)
        road_nodes_list.append(road_node)
    return road_nodes_list

def generate_inundation(road_nodes_list, num_inundation_roads):
    # Choose num_inundation_roads roads to be in inundation roads
    inundation_roads = random.sample(road_nodes_list, num_inundation_roads)

    # Return a list that contains the ids of those road nodes
    return [inundation_road.id for inundation_road in inundation_roads]

# this is a sample road graph, road_graph should be imported from the data file.
# road_graph = {
#     0: [(1, 1), (2, 1), (3,1)],
#     1: [(0, 1), (3, 1), (4,1)],
#     2: [(0, 1), (5, 1), (7,2)],
#     3: [(0, 1), (1, 1), (5, 1), (6, 1), (8, 2)],
#     4: [(1, 1), (6, 1), (9, 2)],
#     5: [(2, 1), (3, 1), (6, 1), (7, 2), (8, 2)],
#     6: [(3, 1), (4, 1), (5, 1), (8, 2), (9, 2)],
#     7: [(2, 2), (5, 2), (10, 3)],
#     8: [(5, 2), (3, 1), (6, 2), (11, 3), (10, 3)],
#     9: [(4, 2), (6, 2), (11, 3)],
#     10: [(7, 3), (8, 3), (11, 3)],
#     11: [(8, 3), (11, 3), (10, 3)]
# }
# # this is real graph data in normal time
# use "graph_flooding.json" if running on disrupted road network
# with open('graph_flooding_copy.json', 'r') as infile:
with open('graph_flooding.json', 'r') as infile:
    road_graph = json.load(infile)
road_graph = {int(k) if k.isdigit() else k: v for k, v in road_graph.items()}

#print(road_graph)

# import road width
# with open('road_width_copy.json', 'r') as infile:
with open('road_width.json', 'r') as infile:
    road_width = json.load(infile)
road_width = {int(k) if k.isdigit() else k: v for k, v in road_width.items()}

# import car generation probability
#with open('p_depart_residential_copy.json', 'r') as infile:
with open('p_depart_residential.json', 'r') as infile:
    p_depart_residential = json.load(infile)
p_res_roads = {int(k) if k.isdigit() else k: v for k, v in p_depart_residential.items()}

#with open('p_depart_business_copy.json', 'r') as infile:
with open('p_depart_business.json', 'r') as infile:
    p_depart_business = json.load(infile)
p_biz_roads = {int(k) if k.isdigit() else k: v for k, v in p_depart_business.items()}

total_time_steps = 36000 # do NOT change, car_generation models the pattern from 8AM to 6PM, 600 min (Xiyu)
road_nodes_list = create_road_nodes(road_graph)
#inundation_roads = {"0":[]}
inundation_roads = {}

# num = 0
# get inundation_roads in the formate of {"0":[1,2],"1":[0],"2":[0]}
for key in road_graph:
    inundation_roads[key]=[]
    for value in road_graph[key]:
        if value[1] == float('inf') and value[0] not in inundation_roads[key]:
            inundation_roads[key].append(value[0])
    # num = num + len(inundation_roads[key])

#print(num)
#print(inundation_roads)

cars_left_simulation = 0
total_cars = 0
cars_in_simulation = 0
cars_cannot_find_route = 0
