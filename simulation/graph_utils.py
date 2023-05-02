import heapq
import random
from car import Car
import math
from config import *
from car_generation import *



def dijkstra(start, end):
    distances = {node: float('infinity') for node in road_graph}
    distances[start] = 0
    previous_nodes = {node: None for node in road_graph}

    priority_queue = [(0, start)]
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        tied_neighbors = []
        min_distance = float('infinity')

        #print(road_graph)
        for neighbor, weight in road_graph[current_node]:
            # if neighbor not in inundation_roads:  # Avoid inundation roads
            if neighbor not in inundation_roads[current_node]:  # Avoid inundation roads
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

                # Check for ties
                if distance < min_distance:
                    min_distance = distance
                    tied_neighbors = [(neighbor, current_node)]
                elif distance == min_distance:
                    tied_neighbors.append((neighbor, current_node))

        # If there's a tie, randomly pick one of the tied neighbors
        if tied_neighbors:
            chosen_neighbor, chosen_previous_node = random.choice(tied_neighbors)
            if distances[chosen_neighbor] == min_distance:
                distances[chosen_neighbor] = min_distance
                previous_nodes[chosen_neighbor] = chosen_previous_node
                heapq.heappush(priority_queue, (min_distance, chosen_neighbor))

    return distances[end], distances, previous_nodes


def compute_route(car):
    shortest_distance, distances, previous_nodes = dijkstra(car.start_point, car.end_point)
    car.route = []

    if shortest_distance == float('infinity'):
        return False  # No route found

    # Reconstruct the shortest path
    current_node = car.end_point
    while current_node != car.start_point:
        car.route.insert(0, current_node)
        current_node = previous_nodes[current_node]

    car.route.insert(0, car.start_point)
    return True


def compute_time_left_at_this_road(car):
    # Calculate time_left_at_this_road of a car based on the number of cars on its road and the length of the edge
    current_road = car.at
    next_road_index = car.route.index(current_road) + 1
    next_road = car.route[next_road_index] if next_road_index < len(car.route) else None

    if next_road is None:
        return 0

    edge_length = None
    for neighbor, weight in road_graph[current_road]:
        if neighbor == next_road:
            edge_length = weight
            break

    if edge_length is None:
        return 0
    
    current_road_node = road_nodes_list[current_road]

    ## edited by dak
    # speed function w.r.t car flow
    current_road_width = road_width[current_road]
    # max speed 80km/h
    # edited by Zefang, V_max in m/s
    V_max = 8000/6/60
    # 3.6576m assumes 12ft lanes
    lane_num = math.floor(current_road_width/3.6576)
    # assume 20ft of space needed per car
    max_cars_on_the_road = math.floor(edge_length/6.096)*lane_num
    # setting speed to 10 mph if at full copacity
    if len(current_road_node.cars_on_the_road) >= max_cars_on_the_road:
        # edited by Zefang, time in second
        return edge_length/1600*6*60
    else:
        reduction_rate = len(current_road_node.cars_on_the_road)/max_cars_on_the_road
        return edge_length/(V_max*(1-reduction_rate))


def generate_cars(number_of_cars, p_res, p_biz):
    global total_cars
    global cars_in_simulation
    global cars_cannot_find_route

    created_cars = []
    not_created_cars = []

    total_cars += number_of_cars

    for _ in range(number_of_cars):
        start_point = int(originDestination_generation(p_res, p_biz, p_res_roads, p_biz_roads))
        end_point = int(originDestination_generation(p_biz, p_res, p_res_roads, p_biz_roads))

        # while start_point == end_point or start_point in inundation_roads or end_point in inundation_roads:
        while start_point == end_point:
            start_point = int(originDestination_generation(p_res, p_biz, p_depart_residential, p_depart_business))
            end_point = int(originDestination_generation(p_biz, p_res, p_depart_residential, p_depart_business))
        
        car = Car(start_point, end_point)

        # Compute the car's route and set its time_left_at_this_road
        route_found = compute_route(car)
        if route_found:
            car.time_left_at_this_road = compute_time_left_at_this_road(car)
            car.time_total = car.time_total + compute_time_left_at_this_road(car)     #hana 

            # Append this car to the cars_on_the_road list of the corresponding road
            road_nodes_list[start_point].cars_on_the_road.append(car)

            # Add the car to the list of created cars
            created_cars.append(car)

            cars_in_simulation += 1

        else:
            # Add the car to the list of not created cars
            not_created_cars.append(car)

            cars_cannot_find_route += 1

    print("Car Created:")
    for i, car in enumerate(created_cars, start=1):
        print(
            f"{i}. start_point={car.start_point}, end_point={car.end_point}, at={car.at}, time_left_at_this_road={car.time_left_at_this_road}, time_total={car.time_total}, route={car.route}")

    print("\nCar Can't be Created (Road not Found):")
    for i, car in enumerate(not_created_cars, start=1):
        print(
            f"{i}. start_point={car.start_point}, end_point={car.end_point}, at={car.at}, time_left_at_this_road={car.time_left_at_this_road}, time_total={car.time_total}, route={car.route}")


def print_cars_on_roads():
    for road_node in road_nodes_list:
        print(f"Road {road_node.id}: {len(road_node.cars_on_the_road)} cars")
        for car in road_node.cars_on_the_road:
            print(f"  Car: start_point={car.start_point}, end_point={car.end_point}, at={car.at}, time_left_at_this_road={car.time_left_at_this_road}, time_total={car.time_total}")
            print(f"  Car route: {car.route}")


def move_cars(totals):
    global cars_left_simulation
    global cars_in_simulation

    cars_left_this_step = 0
    

    for road_node in road_nodes_list:
        cars_to_remove = []

        for car in road_node.cars_on_the_road:
            if car.at == car.end_point:
                # Car has reached its destination
                cars_left_this_step += 1
                cars_to_remove.append(car)
                
                #new from hana and zefang
                totals.append(car.time_total)
                #print(car)
                
            elif car.time_left_at_this_road <= 0:
                # Move car to the next road
                current_road_index = car.route.index(car.at)
                next_road = car.route[current_road_index + 1]
                
                #if next_road not in inundation_roads:
                car.at = next_road
                car.time_left_at_this_road = compute_time_left_at_this_road(car)
                    #new from hana and zefang
                car.time_total = car.time_total + compute_time_left_at_this_road(car)

                road_nodes_list[next_road].new_added_buffer.append(car)
                cars_to_remove.append(car)
            else:
                car.time_left_at_this_road -= 1

        # Remove cars that have moved or left the simulation
        for car in cars_to_remove:
            road_node.cars_on_the_road.remove(car)

            #new from hana and zefang
            #totals.append(car.time_total)

    # Move cars from the new_added_buffer to the cars_on_the_road attribute
    for road_node in road_nodes_list:
        road_node.cars_on_the_road.extend(road_node.new_added_buffer)
        road_node.new_added_buffer.clear()

    cars_left_simulation += cars_left_this_step
    cars_in_simulation -= cars_left_this_step
    print(f"{cars_left_this_step} cars have left the simulation in this step.")


def simulation():
    
    # set the stength of residential departing and business departing
    p_res, p_biz = p_residential_business(numbers_exponential(math.ceil(total_time_steps / 60) + 1))
    totals = []    #new from hana and zefang

    # time steps in seconds
    generation_step = 0
    for step in range(total_time_steps):
        print(f"\nStep {step + 1}:")
        
        # Generate 300 cars every 60 seconds, edited by Zefang
        if (step + 1) % 60 == 0 or step == 0:
            generate_cars(300, p_res[generation_step], p_biz[generation_step])
            generation_step += 1

        # Move the cars every second, Zefang
        move_cars(totals)

        print(f"Total cars generated: {total_cars}")
        print(f"Cars left simulation: {cars_left_simulation}")
        print(f"Cars currently in simulation: {cars_in_simulation}")
        print(f"Cars that couldn't find a route: {cars_cannot_find_route}")

        # print(totals)

    print(f"Length of totals: {np.shape(totals)}")

    print(f"Average time per car: {np.average(totals)}") 
    print(f"Standard Deviation: {np.std(totals)}")
    print(f"Minimum travel time: {np.min(totals)}")
    print(f"Maximum travel time: {np.max(totals)}")

    


