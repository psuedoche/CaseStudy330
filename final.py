import csv
import json
import time
import heapq
from datetime import datetime, timedelta
from math import sqrt
import math
import random

''' 

Preprocessing

'''
def read_csv_to_matrix(file_path):
    data_matrix = []

    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            data_matrix.append(row)
    return data_matrix

def read_graph_nodes_and_edges():
    with open('node_data.json') as nodes_file:
        nodes = json.load(nodes_file)

    edges = {}
    with open('edges.csv', 'r') as edges_csv:
        edges_reader = csv.DictReader(edges_csv)
        for row in edges_reader:
            start_id = int(row['start_id'])
            end_id = int(row['end_id'])
            length = float(row['length'])

            # Additional attributes for each hour of the day
            weekday_hours = [float(row[f'weekday_{i}']) for i in range(24)]
            weekend_hours = [float(row[f'weekend_{i}']) for i in range(24)]

            edge = (start_id, end_id)

            edges[edge] = {
                'length': length,
                'weekday_hours': weekday_hours,
                'weekend_hours': weekend_hours
            }

    return nodes, edges

def create_adjacency(nodes, edges):
    complete_adjacency = {}
    # Populate the adjacency list with edges and weights
    for edge, attributes in edges.items():
        source_node, target_node = edge
        length = attributes["length"]
        speeds = attributes['weekday_hours']
        average_speed = sum(speeds)/len(speeds)
        estimated_time = length / average_speed


        # Add source to target with weight
        if source_node in complete_adjacency:
            complete_adjacency[source_node][target_node] = estimated_time
        else:
            complete_adjacency[source_node] = {target_node: estimated_time}

        # Ensure target node is in the dictionary, even if it has no outgoing edges
        if target_node not in complete_adjacency:
            complete_adjacency[target_node] = {}

    # Ensure all nodes are in the dictionary, even if they have no edges
    for node in nodes:
        if node[0] not in complete_adjacency:
            complete_adjacency[node[0]] = {}
    
    return complete_adjacency

def cluster_nodes(nodes, lat_decimal_places=3, long_decimal_places=3):
    '''
    Create node clusters grouped by the first place values of coordinates

    Return (dict): {(cluster_lat, cluster_lon): 
                            [node_id, node_id....], 
                    ...}
    '''
    clusters = {}
    for node_id, coordinates in nodes.items():
        truncated_lat = round(coordinates['lat'], lat_decimal_places)
        truncated_lon = round(coordinates['lon'], long_decimal_places)

        # Use a tuple (truncated_lat, truncated_lon) as the cluster identifier
        cluster_key = (truncated_lat, truncated_lon)

        if cluster_key not in clusters:
            clusters[cluster_key] = []

        clusters[cluster_key].append(node_id)

    return clusters

def create_clusters_network(nodes, edges, lat_decimal_places=3, long_decimal_places=3, avg_speeds=True):
    '''
    nodes: all nodes
    edges: original edges bewteen all nodes
    decimal_places: the value to truncate at
    '''
    adjacency_dict = {}
    for edge, attributes in edges.items():
        source_node, target_node = edge
        source_node = nodes[str(source_node)]
        target_node = nodes[str(target_node)]

        cluster_key1 = (round(source_node['lat'], lat_decimal_places),round(source_node['lon'], long_decimal_places))
        cluster_key2 = (round(target_node['lat'], lat_decimal_places),round(target_node['lon'], long_decimal_places))

        # check if source and target are in different clusters aka they have different cluster keys 
        # if they are different, add it to the graph as an edge
        if cluster_key1 != cluster_key2:
            length = attributes["length"]
            speeds = attributes['weekday_hours']
            average_speed = sum(speeds)/len(speeds)
            estimated_time = length / average_speed
            # add this edge to the adjacency
            # Add source to target with weight
        if cluster_key1 in adjacency_dict:
            if cluster_key2 in adjacency_dict[cluster_key1]:
                adjacency_dict[cluster_key1][cluster_key2].append(estimated_time)
            else:
                adjacency_dict[cluster_key1][cluster_key2] = [estimated_time]
        else:
            adjacency_dict[cluster_key1] = {cluster_key2: [estimated_time]}

        # Ensure target node is in the dictionary, even if it has no outgoing edges
        if cluster_key2 not in adjacency_dict:
            adjacency_dict[cluster_key2] = {}
    if avg_speeds:
        '''
        Average all speeds for edges
        '''
        for key1, nested_dict in adjacency_dict.items():
                # Iterate over each key-value pair in the second level of the dictionary
                for key2, value in nested_dict.items():            
                    # Calculate the average of the nested lists
                    avg_list = sum(value) / len(value) 
                    
                    # Update the nested dictionary with the average list
                    adjacency_dict[key1][key2] = avg_list
    else:
        '''
        Divide speeds into hours and weekdays
        '''
        for key1, nested_dict in adjacency_dict.items():
            # Iterate over each key-value pair in the second level of the dictionary
            for key2, value in nested_dict.items():            
                # Calculate the average of the nested lists
                # Use zip to group elements at the same index
                zipped_lists = zip(*value)

                # Calculate the average for each group of elements
                averages = [sum(values) / len(values) for values in zipped_lists]

                #split into AM/PM weekday/weekend
                weekday_hours = averages[:24]
                weekend_hours = averages[24:]

                weekday_am_average = sum(weekday_hours[:12]) / 12  # Assuming 0-11 are weekday AM hours
                weekday_pm_average = sum(weekday_hours[12:]) / 12  # Assuming 12-23 are weekday PM hours
                weekend_am_average = sum(weekend_hours[:12]) / 12  # Assuming 24-35 are weekend AM hours
                weekend_pm_average = sum(weekend_hours[12:]) / 12  # Assuming 36-47 are weekend PM hours

                averages = [weekday_am_average, weekday_pm_average, weekend_am_average, weekend_pm_average]
                # Update the nested dictionary with the average list
                adjacency_dict[key1][key2] = averages
    return adjacency_dict

'''

HELPER FUNCTIONS

'''

def euclidean_distance(lat1, lon1, lat2, lon2):
    # Calculate the Euclidean distance between two sets of coordinates
    dx = lon2 - lon1
    dy = lat2 - lat1
    distance = sqrt(dx**2 + dy**2)
    return distance

def dijkstras(graph, start, target):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    priority_queue = [(0, start)]
    heapq.heapify(priority_queue)

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node == target:
            break

        if current_distance <= distances[current_node]:
            for neighbor, weight in graph[(current_node)].items():
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

    return distances[(target)]

def find_closest_coordinates(target_coordinates, known_coordinates, clusters=True):
    min_distance = float('inf')
    closest_coordinates = None

    for key, value in known_coordinates.items(): #nodes or clusters
        if clusters:
            distance = euclidean_distance(target_coordinates[0], target_coordinates[1], key[0], key[1])
        else:
            distance = euclidean_distance(target_coordinates[0], target_coordinates[1], value['lon'], value['lat'])
        if distance < min_distance:
            min_distance = distance
            closest_coordinates = key

    return closest_coordinates

def floyd_warshall(graph):
    vertices = list(graph.keys())

    # Initialize the distance matrix and the dictionary for intermediate vertices
    dist = []
    nIndex = {}
    
    for index, key in enumerate(vertices):
        nIndex[key] = index
    
    for i in vertices:
        row = []
        for j in vertices:
            if i == j:
                row.append(0)
            elif j in graph[i]:
                row.append(graph[i][j])
            else:
                row.append(float('inf'))
        dist.append(row)

    for k in vertices:
        for i in vertices:
            for j in vertices:
                if dist[nIndex[i]][nIndex[k]] + dist[nIndex[k]][nIndex[j]] < dist[nIndex[i]][nIndex[j]]:
                    dist[nIndex[i]][nIndex[j]] = dist[nIndex[i]][nIndex[k]] + dist[nIndex[k]][nIndex[j]]

    return dist, nIndex

def floyd_warshall_t5(graph):
    '''
    Similar to regular floyd warshall, but compares based on the time of day (AM or PM) and Weekday or Weekend
    '''
    vertices = list(graph.keys())

    # Initialize the distance matrix and the dictionary for intermediate vertices
    dist = []
    nIndex = {}
    
    # Create the nIndex using a regular for loop
    for index, key in enumerate(vertices):
        nIndex[key] = index
    
    for i in vertices:
        row = []
        for j in vertices: 
            if j in graph[i]:
                row.append(graph[i][j]) 
            else:
                row.append([float('inf')] * 4)
        dist.append(row)

    for k in vertices:
        for i in vertices:
            for j in vertices:
                for h in range(4):
                    if dist[nIndex[i]][nIndex[k]][h] + dist[nIndex[k]][nIndex[j]][h] < dist[nIndex[i]][nIndex[j]][h]:
                        #print("update")
                        dist[nIndex[i]][nIndex[j]][h] = dist[nIndex[i]][nIndex[k]][h] + dist[nIndex[k]][nIndex[j]][h]

    return dist, nIndex
'''

CODE FOR TASKS

'''
def match_passenger_to_driver_t1(drivers, passengers):
    '''
    T1 of case study

    Parameters:
    - drivers matrix: time, lat, long
    - passengers matrix: time, lat, long

    Description:
    Assigns drivers to passengers solely based on wait time. 
    '''
    start_time = time.time()
    # Convert dates to datetime objects and initialize priority queues
    driver_heap = [(datetime.strptime(driver[0], '%m/%d/%Y %H:%M:%S'), driver) for driver in drivers]
    passenger_heap = [(datetime.strptime(passenger[0], '%m/%d/%Y %H:%M:%S'), passenger) for passenger in passengers]

    # Convert lists to heaps
    heapq.heapify(driver_heap)
    heapq.heapify(passenger_heap)

    # Match drivers to passengers based on earliest date/time
    while driver_heap and passenger_heap:
        current_driver_time, current_driver = heapq.heappop(driver_heap)
        current_passenger_time, current_passenger = heapq.heappop(passenger_heap)
        print(f"Driver assigned to Passenger: {current_driver} -> {current_passenger}")
        drive_duration = timedelta(minutes=20)
        # requeue driver 
        heapq.heappush(driver_heap, (current_driver_time + drive_duration, current_driver))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Matching process complete. Elapsed time: {elapsed_time} seconds")

def match_passenger_to_driver_t2(drivers, passengers):
    '''
    T2 of case study

    Parameters:
    - drivers matrix: time, lat, long
    - passengers matrix: time, lat, long

    Description:
    Assigns drivers to passengers based on straight line distance between coordinate locations.
    '''
    start_time = time.time()
    # Convert dates to datetime objects and initialize priority queues
    driver_heap = [(datetime.strptime(driver[0], '%m/%d/%Y %H:%M:%S'), driver) for driver in drivers]
    passenger_heap = [(datetime.strptime(passenger[0], '%m/%d/%Y %H:%M:%S'), passenger) for passenger in passengers]

    # Convert driver list to priority queue
    heapq.heapify(driver_heap)

    while driver_heap and passenger_heap:
        current_driver_time, current_driver = heapq.heappop(driver_heap)
        dist = float('inf')
        current_passenger = 0
        for passenger in passenger_heap:
            if passenger[0] < current_driver_time:
                if euclidean_distance(float(current_driver[1]),float(current_driver[2]),float(passenger[1][1]),float(passenger[1][2])) < dist:
                    dist = euclidean_distance(float(current_driver[1]),float(current_driver[2]),float(passenger[1][1]),float(passenger[1][2]))
                    current_passenger = passenger
        if current_passenger != 0:
            passenger_heap.remove(current_passenger)
            print(f"Driver assigned to Passenger: {current_driver} -> {current_passenger[1]}")
        drive_duration = timedelta(minutes=20)
        # requeue driver 
        heapq.heappush(driver_heap, (current_driver_time + drive_duration, current_driver))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Matching process complete. Elapsed time: {elapsed_time} seconds")

def match_passenger_to_driver_t3(drivers, passengers, adjacency, clusters):
    '''
    T2 of case study

    Parameters:
    - drivers matrix: time, lat, long
    - passengers matrix: time, lat, long
    - adjacency: cluster graph
    - clusters: dicts of clusters indexed by coordinates

    Description:
    Assigns drivers to passengers based on straight line distance between coordinate locations.
    '''
    start_time = time.time()

    # Convert dates to datetime objects and initialize priority queues
    driver_heap = [(datetime.strptime(driver[0], '%m/%d/%Y %H:%M:%S'), driver) for driver in drivers]
    passenger_heap = [(datetime.strptime(passenger[0], '%m/%d/%Y %H:%M:%S'), passenger) for passenger in passengers]

    # Convert lists to heaps
    heapq.heapify(driver_heap)

    # Match drivers to passengers based on earliest date/time
    while driver_heap and passenger_heap:
        current_driver_time, current_driver = heapq.heappop(driver_heap)
        shortest_path_time = float('inf')
        current_passenger = None
        current_passenger_time = None
        for passenger_time, passenger in passenger_heap:
            if passenger_time < current_driver_time:
                s1_time = time.time()
                print(f"Start")
                start_node = find_closest_coordinates((float(current_driver[1]), float(current_driver[2])), clusters)
                end_node = find_closest_coordinates((float(passenger[1]), float(passenger[2])), clusters)
                print(f"closest coord done")

                path_time = dijkstras(adjacency, start_node, end_node)
                print(f"Dijkstras done")

                if path_time < shortest_path_time:
                    shortest_path_time = path_time
                    current_passenger = passenger
                    current_passenger_time = passenger_time
        
        if current_passenger:
            passenger_heap.remove((current_passenger_time,current_passenger))
            current_driver[1] = current_passenger[3]
            current_driver[2] = current_passenger[4]
            print(f"Driver assigned to Passenger: {current_driver} -> {current_passenger}")

        if random.random() < 0.95:
            if shortest_path_time < 10000:
                drive_duration = timedelta(hours=shortest_path_time)
            else:
                drive_duration = timedelta(minutes=2)
            heapq.heappush(driver_heap, (current_driver_time + drive_duration, current_driver))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Matching process complete. Elapsed time: {elapsed_time} seconds")

def match_passenger_to_driver_t4(drivers, passengers, adjacency_matrices, nIndex, nodes):
    start_time = time.time()

    # Convert dates to datetime objects and initialize priority queues
    driver_heap = [(datetime.strptime(driver[0], '%m/%d/%Y %H:%M:%S'), driver) for driver in drivers]
    passenger_heap = [(datetime.strptime(passenger[0], '%m/%d/%Y %H:%M:%S'), passenger) for passenger in passengers]

    # Convert lists to heaps
    heapq.heapify(driver_heap)

    count = 0

    # Match drivers to passengers based on earliest date/time
    while driver_heap and passenger_heap:
        current_driver_time, current_driver = heapq.heappop(driver_heap)
        shortest_path_time = float('inf')
        current_passenger = None
        current_passenger_time = None

        for passenger_time, passenger in passenger_heap:
            minni = timedelta(minutes=7)
            max_wait_time = current_driver_time - minni

            if passenger_time < current_driver_time and passenger_time > max_wait_time:
                start_node = find_closest_coordinates((float(current_driver[1]), float(current_driver[2])), nodes)
                end_node = find_closest_coordinates((float(passenger[1]), float(passenger[2])), nodes)
                path_time = adjacency_matrices[nIndex[start_node]][nIndex[end_node]]
                
                if path_time < shortest_path_time:
                    shortest_path_time = path_time
                    current_passenger = passenger
                    current_passenger_time = passenger_time

        if current_passenger:
            print(current_passenger)
            passenger_heap.remove((current_passenger_time, current_passenger))
            current_driver[1] = current_passenger[3]
            current_driver[2] = current_passenger[4]
            count = count + 1
            print(count)
            #print(f"Driver assigned to Passenger: {current_driver} -> {current_passenger}")

            # Introduce a 95% chance for the driver to be pushed back
        if random.random() < 0.95:
            if shortest_path_time < 10000:
                drive_duration = timedelta(hours=shortest_path_time)
            else:
                drive_duration = timedelta(minutes=2)
            heapq.heappush(driver_heap, (current_driver_time + drive_duration, current_driver))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Matching process complete. Elapsed time: {elapsed_time} seconds")

def match_passenger_to_driver_t5(drivers, passengers, adjacency_matrices, nIndex, nodes, adj, nodes1):
    start_time = time.time()
    drivers = drivers[:10]
    passengers = passengers[:100]
    # Convert dates to datetime objects and initialize priority queues
    driver_heap = [(datetime.strptime(driver[0], '%m/%d/%Y %H:%M:%S'), driver) for driver in drivers]
    passenger_heap = [(datetime.strptime(passenger[0], '%m/%d/%Y %H:%M:%S'), passenger) for passenger in passengers]

    # Convert lists to heaps
    heapq.heapify(driver_heap)

    count = 0

    # Match drivers to passengers based on earliest date/time
    while driver_heap and passenger_heap:
        current_driver_time, current_driver = heapq.heappop(driver_heap)
        shortest_path_time = float('inf')
        current_passenger = None
        current_passenger_time = None

        for passenger_time, passenger in passenger_heap:
            minni = timedelta(minutes=7)
            max_wait_time = current_driver_time - minni

            if passenger_time < current_driver_time and passenger_time > max_wait_time:

                driver_node = find_closest_coordinates((float(current_driver[1]), float(current_driver[2])), nodes)
                passenger_start_node = find_closest_coordinates((float(passenger[1]), float(passenger[2])), nodes)

                if passenger_start_node == driver_node:
                    passenger_start_node = find_closest_coordinates((float(passenger[1]), float(passenger[2])), nodes1, False)
                    driver_node = find_closest_coordinates((float(passenger[1]), float(passenger[2])), nodes1, False)
                    path_time = dijkstras(adj,driver_node,passenger_start_node)
                else:    
                    passenger_end_node = find_closest_coordinates((float(passenger[3]), float(passenger[4])), nodes)

                    ride_duration = adjacency_matrices[nIndex[passenger_start_node]][nIndex[passenger_end_node]]  
                    driver_passenger = adjacency_matrices[nIndex[driver_node]][nIndex[passenger_start_node]]

                    path_time = ride_duration - driver_passenger

                if path_time < shortest_path_time:
                    shortest_path_time = path_time
                    current_passenger = passenger
                    current_passenger_time = passenger_time

        if current_passenger:
            print(current_passenger)
            passenger_heap.remove((current_passenger_time, current_passenger))
            current_driver[1] = current_passenger[3]
            current_driver[2] = current_passenger[4]
            
            print(f"Driver assigned to Passenger: {current_driver} -> {current_passenger}")
            # Introduce a 95% chance for the driver to be pushed back
        if random.random() < 0.95:
            drive_duration = timedelta(minutes=2)
            heapq.heappush(driver_heap, (current_driver_time + drive_duration, current_driver))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Matching process complete. Elapsed time: {elapsed_time} seconds")

def match_passenger_to_driver_b1(drivers, passengers, adjacency_matrices, nIndex, nodes, adj, nodes1):
    start_time = time.time()

    driver_heap = [(datetime.strptime(driver[0], '%m/%d/%Y %H:%M:%S'), driver) for driver in drivers]
    passenger_heap = [(0, datetime.strptime(passenger[0], '%m/%d/%Y %H:%M:%S'), passenger) for passenger in passengers]

    heapq.heapify(driver_heap)

    while driver_heap and passenger_heap:
        current_driver_time, current_driver = heapq.heappop(driver_heap)
        shortest_path_time = float('inf')
        current_passenger = None
        current_passenger_time = None
        heapq.heapify(passenger_heap)
        passenger_heap = list(passenger_heap)
        max_priority = passenger_heap[0][0]

        for priority, passenger_time, passenger in passenger_heap:
            minni = timedelta(minutes=7)
            max_wait_time = current_driver_time - minni

            if passenger_time < current_driver_time and priority >= max_priority-1 and passenger_time >= max_wait_time:
                driver_node = find_closest_coordinates((float(current_driver[1]),float(current_driver[2])), nodes)
                passenger_start_node = find_closest_coordinates((float(passenger[1]),float(passenger[2])), nodes)
                passenger_end_node = find_closest_coordinates((float(passenger[3]),float(passenger[4])), nodes)

                ride_duration =adjacency_matrices[nIndex[passenger_start_node]][nIndex[passenger_end_node]]
                driver_passenger = adjacency_matrices[nIndex[driver_node]][nIndex[passenger_start_node]]

                path_time = ride_duration - driver_passenger

                if passenger_start_node == driver_node:
                    passenger_start_node = find_closest_coordinates((float(passenger[1]),float(passenger[2])), nodes1)
                    driver_node = find_closest_coordinates((float(passenger[1]),float(passenger[2])), nodes1)
                    path_time = dijkstras(adj,driver_node,passenger_start_node)
                    """ if passenger_start_node == passenger_end_node:
                        path_time = dijkstras(adj, passenger_start_node, passenger_end_node) - path_time
                    else:
                        path_time = adjacency_matrices[nIndex[passenger_start_node]][nIndex[passenger_end_node]] - path_time """
                
                if path_time < shortest_path_time:
                    shortest_path_time = path_time
                    current_passenger = passenger
                    current_passenger_time = passenger_time
                
            if passenger_time < max_wait_time:
                if random.random() < 0.33:
                    passenger_heap.remove((current_passenger_time, current_passenger))
                else:
                    drive = timedelta(minutes=7)
                    passenger[0] = str(passenger_time + drive)
                    priority = priority + 1
                    passenger_time = passenger_time + drive
                
        if current_passenger:
            passenger_heap.remove((current_passenger_time, current_passenger))
            current_driver[1] = current_passenger[3]
            current_driver[2] = current_passenger[4]
            print(f"Driver assigned to Passenger: {current_driver} -> {current_passenger}")

        if random.random() < 0.95:
            if shortest_path_time < 10000:
                drive_duration = timedelta(hours=shortest_path_time)
            else:
                drive_duration = timedelta(minutes=2)
            heapq.heappush(driver_heap, (current_driver_time + drive_duration, current_driver))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Matching process complete. Elapsed time: {elapsed_time} seconds")

def match_passenger_to_driver_b2(drivers, passengers, adjacency_matrices, nodes, nIndex, adj, nodes1):
    start_time = time.time()
    drivers = drivers[:100]
    passengers = passengers[:1000]
    # Convert dates to datetime objects and initialize priority queues
    driver_heap = [(datetime.strptime(driver[0], '%m/%d/%Y %H:%M:%S'), driver) for driver in drivers]
    passenger_heap = [(datetime.strptime(passenger[0], '%m/%d/%Y %H:%M:%S'), passenger) for passenger in passengers]

    # Convert lists to heaps
    heapq.heapify(driver_heap)

    count = 0

    # Match drivers to passengers based on earliest date/time
    while driver_heap and passenger_heap:
        current_driver_time, current_driver = heapq.heappop(driver_heap)
        shortest_path_time = float('inf')
        current_passenger = None
        current_passenger_time = None

        for passenger_time, passenger in passenger_heap:
            minni = timedelta(minutes=7)
            max_wait_time = current_driver_time - minni
            
            if passenger_time < current_driver_time and passenger_time > max_wait_time:

                # clusters
                driver_large_cluster = find_closest_coordinates((float(current_driver[1]), float(current_driver[2])), nodes)
                passenger_start_large_cluster = find_closest_coordinates((float(passenger[1]), float(passenger[2])), nodes)
                passenger_end_large_cluster = find_closest_coordinates((float(passenger[3]), float(passenger[4])), nodes)
                
                if passenger_start_large_cluster == driver_large_cluster: # in the same cluster of 900
                    # use original nodes to run dijkstras for close coordinates (within a cluster)
                    passenger_start_node = find_closest_coordinates((float(passenger[1]), float(passenger[2])), nodes1, False)
                    driver_node = find_closest_coordinates((float(current_driver[1]), float(current_driver[2])), nodes1, False)
                    driver_to_passenger = dijkstras(adj,driver_node,passenger_start_node)
                
                # not in the same cluster, read from FW array
                else:    
                    driver_to_passenger = adjacency_matrices[nIndex[driver_large_cluster]][nIndex[passenger_start_large_cluster]]

                # if they are in the same cluster, we again run dijkstras on the orignial path network
                if passenger_start_large_cluster == passenger_end_large_cluster:
                    passenger_start_node = find_closest_coordinates((float(passenger[1]), float(passenger[2])), nodes1, False)
                    passenger_end_node = find_closest_coordinates((float(passenger[3]), float(passenger[4])), nodes1, False)
                    ride_duration = dijkstras(adj, passenger_start_node, passenger_end_node)
                # if they are not in the same cluster, we can read from the adjacency 
                else:
                    ride_duration = adjacency_matrices[nIndex[passenger_start_large_cluster]][nIndex[passenger_end_large_cluster]]

                path_time = ride_duration - driver_to_passenger
                
                if path_time < shortest_path_time:
                    shortest_path_time = path_time
                    current_passenger = passenger
                    current_passenger_time = passenger_time

        if current_passenger:
            print(current_passenger)
            passenger_heap.remove((current_passenger_time, current_passenger))
            current_driver[1] = current_passenger[3]
            current_driver[2] = current_passenger[4]
            count = count + 1
            print(count)
            print(f"Driver assigned to Passenger: {current_driver} -> {current_passenger}")

            # Introduce a 95% chance for the driver to be pushed back
        if random.random() < 0.95:
            drive_duration = timedelta(minutes=2)
            heapq.heappush(driver_heap, (current_driver_time + drive_duration, current_driver))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Matching process complete. Elapsed time: {elapsed_time} seconds")

def main():
    drivers_matrix = read_csv_to_matrix('drivers.csv')
    passengers_matrix = read_csv_to_matrix('passengers.csv')
    nodes, edges = read_graph_nodes_and_edges()
    complete_adjacency = create_adjacency(nodes, edges)

    clusters = cluster_nodes(nodes, 2, 3)
    cluster_network = create_clusters_network(nodes, edges, 2, 3)
    
    clusters_sparse = cluster_nodes(nodes, 3, 3)
    cluster_network_sparse = create_clusters_network(nodes, edges, 3, 3)
    
    cluster_network_t5 = create_clusters_network(nodes, edges, 3, 3, False)
    
    '''
    T1 - T3
    ''' 
    match_passenger_to_driver_t1(drivers_matrix[1:], passengers_matrix[1:])
    match_passenger_to_driver_t2(drivers_matrix[1:], passengers_matrix[1:])
    match_passenger_to_driver_t3(drivers_matrix[1:], passengers_matrix[1:], cluster_network, clusters)
    
    '''
    T4
    ''' 
    fw_matrix, indices = floyd_warshall(cluster_network_sparse)
    match_passenger_to_driver_t4(drivers_matrix[1:], passengers_matrix[1:], fw_matrix, indices, cluster_network_sparse)
    
    '''
    T5
    '''
    fw_matrix_t5, indices_t5 = floyd_warshall_t5(cluster_network_sparse)
    match_passenger_to_driver_t5(drivers_matrix[1:], passengers_matrix[1:], fw_matrix_t5, indices_t5, cluster_network_t5, complete_adjacency, nodes)

    '''
    B1
    '''
    match_passenger_to_driver_b1(drivers_matrix[1:], passengers_matrix[1:], fw_matrix, indices, cluster_network_sparse,  complete_adjacency, nodes)

    '''
    B2
    '''
    match_passenger_to_driver_b2(drivers_matrix[1:], passengers_matrix[1:], fw_matrix, indices, cluster_network_sparse, complete_adjacency, nodes)

if __name__ == "__main__":
    main()