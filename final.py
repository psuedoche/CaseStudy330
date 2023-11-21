import csv
import json
import time
import heapq
from datetime import datetime, timedelta
import calendar
from math import sqrt
import math
import random

def read_csv_to_matrix(file_path):
    data_matrix = []

    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            data_matrix.append(row)

    return data_matrix

csv_file_path = 'drivers.csv'
drivers_matrix = read_csv_to_matrix('drivers.csv')
passengers_matrix = read_csv_to_matrix('passengers.csv')
edges_matrix = read_csv_to_matrix('edges.csv')


def read_json_to_dict(file_path):
    with open(file_path, 'r') as json_file:
        data_dict = json.load(json_file)

    return data_dict

json_file_path = 'node_data.json'
node_data_dict = read_json_to_dict(json_file_path)

count = 0
""" for node_id, coordinates in node_data_dict.items():
    print(f'Node ID: {node_id}, Coordinates: {coordinates}')
    count += 1
    if count == 5:
        break """

def match_passenger_to_driver(drivers, passengers):
    start_time = time.time()
    # Convert dates to datetime objects and initialize priority queues
    driver_heap = [(datetime.strptime(driver[0], '%m/%d/%Y %H:%M:%S'), driver) for driver in drivers]
    passenger_heap = [(datetime.strptime(passenger[0], '%m/%d/%Y %H:%M:%S'), passenger) for passenger in passengers]
    #driver_heap = driver_heap[:500]
    #passenger_heap = passenger_heap[:3000]
    # Convert lists to heaps
    heapq.heapify(driver_heap)
    heapq.heapify(passenger_heap)

    # Match drivers to passengers based on earliest date/time
    while driver_heap and passenger_heap:
        current_driver_time, current_driver = heapq.heappop(driver_heap)
        current_passenger_time, current_passenger = heapq.heappop(passenger_heap)
        # remove conditional, assign first passenger to first driver
        print(f"Driver assigned to Passenger: {current_driver} -> {current_passenger}")
        drive_duration = timedelta(minutes=20)
        # requeue driver 
        heapq.heappush(driver_heap, (current_driver_time + drive_duration, current_driver))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Matching process complete. Elapsed time: {elapsed_time} seconds")
 #04/25/2014 07:00:00
#match_passenger_to_driver(drivers_matrix[1:], passengers_matrix[1:])


def euclidean_distance(lat1, lon1, lat2, lon2):
    # Calculate the Euclidean distance between two sets of coordinates
    dx = lon2 - lon1
    dy = lat2 - lat1
    distance = sqrt(dx**2 + dy**2)
    return distance

def match_passenger_to_driver_t2(drivers, passengers):
    start_time = time.time()
    # Convert dates to datetime objects and initialize priority queues
    driver_heap = [(datetime.strptime(driver[0], '%m/%d/%Y %H:%M:%S'), driver) for driver in drivers]
    passenger_heap = [(datetime.strptime(passenger[0], '%m/%d/%Y %H:%M:%S'), passenger) for passenger in passengers]
    #driver_heap = driver_heap[:300]
    #passenger_heap = passenger_heap[:3000]
    # Convert lists to heaps
    heapq.heapify(driver_heap)
    # heapq.heapify(passenger_heap)

    # Match drivers to passengers based on earliest date/time
    while driver_heap and passenger_heap:
        current_driver_time, current_driver = heapq.heappop(driver_heap)
        dist = float('inf')
        current_passenger = 0
        #count = -1
        #current_passenger_ind = 0
        for passenger in passenger_heap:
            #count = count + 1
            if passenger[0] < current_driver_time:
                if euclidean_distance(float(current_driver[1]),float(current_driver[2]),float(passenger[1][1]),float(passenger[1][2])) < dist:
                    dist = euclidean_distance(float(current_driver[1]),float(current_driver[2]),float(passenger[1][1]),float(passenger[1][2]))
                    current_passenger = passenger
                    #current_passenger_ind = count
        if current_passenger != 0:
            passenger_heap.remove(current_passenger)
        # remove conditional, assign first passenger to first driver
            print(f"Driver assigned to Passenger: {current_driver} -> {current_passenger[1]}")
        drive_duration = timedelta(minutes=20)
        # requeue driver 
        heapq.heappush(driver_heap, (current_driver_time + drive_duration, current_driver))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Matching process complete. Elapsed time: {elapsed_time} seconds")
 #04/25/2014 07:00:00
#match_passenger_to_driver_t2(drivers_matrix[1:], passengers_matrix[1:])


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
nodes, edges = read_graph_nodes_and_edges()

adjacency_dict = {}
# Populate the adjacency list with edges and weights
for edge, attributes in edges.items():
    source_node, target_node = edge
    length = attributes["length"]
    speeds = attributes['weekday_hours']
    average_speed = sum(speeds)/len(speeds)
    estimated_time = length / average_speed


    # Add source to target with weight
    if source_node in adjacency_dict:
        adjacency_dict[source_node][target_node] = estimated_time
    else:
        adjacency_dict[source_node] = {target_node: estimated_time}

    # Ensure target node is in the dictionary, even if it has no outgoing edges
    if target_node not in adjacency_dict:
        adjacency_dict[target_node] = {}

# Ensure all nodes are in the dictionary, even if they have no edges
for node in nodes:
    if node[0] not in adjacency_dict:
        adjacency_dict[node[0]] = {}

def cluster_nodes(nodes, decimal_places=3):
    clusters = {}
    for node_id, coordinates in nodes.items():
        truncated_lat = round(coordinates['lat'], 2)
        truncated_lon = round(coordinates['lon'], 3)

        # Use a tuple (truncated_lat, truncated_lon) as the cluster identifier
        cluster_key = (truncated_lat, truncated_lon)

        if cluster_key not in clusters:
            clusters[cluster_key] = []

        clusters[cluster_key].append(node_id)

    return clusters

clusters = cluster_nodes(nodes)

def update_edges_between_clusters(clusters, nodes):
    adjacency_dict = {}
    for edge, attributes in edges.items():
        source_node, target_node = edge
        source_node = nodes[str(source_node)]
        target_node = nodes[str(target_node)]

        cluster_key1 = (round(source_node['lat'], 2),round(source_node['lon'], 3))
        cluster_key2 = (round(target_node['lat'], 2),round(target_node['lon'], 3))

        #print(cluster_key1,cluster_key2)
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
    return adjacency_dict

adjacency = update_edges_between_clusters(clusters, nodes)

for key1, nested_dict in adjacency.items():
        # Iterate over each key-value pair in the second level of the dictionary
        for key2, value in nested_dict.items():            
            # Calculate the average of the nested lists
            avg_list = sum(value) / len(value) 
            
            # Update the nested dictionary with the average list
            adjacency[key1][key2] = avg_list

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

def average_edge_distance(known_coordinates, graph):
    total_distance = 0
    count = 0

    for node1, neighbors in graph.items():
        for node2 in neighbors:
            distance = euclidean_distance(float(node1[0]), float(node1[1]), float(node2[0]), float(node2[1]))
            total_distance += distance
            count += 1

    if count == 0:
        return 0
    
    average_distance = total_distance / count
    return average_distance

def find_closest_coordinates(target_coordinates, known_coordinates):
    min_distance = float('inf')
    closest_coordinates = None

    for key, value in known_coordinates.items():
        distance = euclidean_distance(float(target_coordinates[0]), float(target_coordinates[1]), float(key[0]), float(key[1]))
        if distance < min_distance:
            min_distance = distance
            closest_coordinates = key

    return closest_coordinates


def match_passenger_to_driver_t3(drivers, passengers, adjacency, clusters):
    start_time = time.time()

    # Convert dates to datetime objects and initialize priority queues
    driver_heap = [(datetime.strptime(driver[0], '%m/%d/%Y %H:%M:%S'), driver) for driver in drivers]
    passenger_heap = [(datetime.strptime(passenger[0], '%m/%d/%Y %H:%M:%S'), passenger) for passenger in passengers]

    #driver_heap = driver_heap[:100]
    #passenger_heap = passenger_heap[:3000]
    # Convert lists to heaps
    heapq.heapify(driver_heap)

    count  = 0
    #avg_dist = average_edge_distance(clusters, adjacency)
    # Match drivers to passengers based on earliest date/time
    while driver_heap and passenger_heap:
        current_driver_time, current_driver = heapq.heappop(driver_heap)
        shortest_path_time = float('inf')
        current_passenger = None
        current_passenger_time = None
        count1 = 0
        for passenger_time, passenger in passenger_heap:
            if passenger_time < current_driver_time:
                s1_time = time.time()
                #print(f"Start")
                start_node = find_closest_coordinates((float(current_driver[1]), float(current_driver[2])), clusters)
                end_node = find_closest_coordinates((float(passenger[1]), float(passenger[2])), clusters)
                #print(f"closest coord done")
                e1_time = time.time()
                d1_time = e1_time - s1_time
                """ print(d1_time)
                print(f"--\n--\n--\n")
                print(f"Start") """
                s_time = time.time()
                path_time = dijkstras(adjacency, start_node, end_node)
                #print(f"Dijkstras done")
                count1 = count1 + 1
                e_time = time.time()
                d_time = e_time - s_time
                """ print(d_time)
                print(count1)
                print(f"--\n--\n--\n") """
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

match_passenger_to_driver_t3(drivers_matrix[1:], passengers_matrix[1:], adjacency, clusters)

def cluster_nodes2(nodes, decimal_places=3):
    clusters = {}
    for node_id, coordinates in nodes.items():
        truncated_lat = round(coordinates['lat'], 2)
        truncated_lon = round(coordinates['lon'], 2)
        cluster_key = (truncated_lat, truncated_lon)

        if cluster_key not in clusters:
            clusters[cluster_key] = []

        clusters[cluster_key].append(node_id)

    return clusters

clusters2 = cluster_nodes2(nodes)
#print(clusters2)
def update_edges_between_clusters2(clusters2, nodes):
    adjacency_dict = {}
    for edge, attributes in edges.items():
        source_node, target_node = edge
        source_node = nodes[str(source_node)]
        target_node = nodes[str(target_node)]

        cluster_key1 = (round(source_node['lat'], 2),round(source_node['lon'], 2))
        cluster_key2 = (round(target_node['lat'], 2),round(target_node['lon'], 2))
        length = attributes["length"]
        speeds = attributes['weekday_hours']
        average_speed = sum(speeds)/len(speeds)
        estimated_time = length / average_speed


        if cluster_key1 in adjacency_dict:
            if cluster_key2 in adjacency_dict[cluster_key1]:
                adjacency_dict[cluster_key1][cluster_key2].append(estimated_time)
            else:
                adjacency_dict[cluster_key1][cluster_key2] = [estimated_time]
      
        if cluster_key2 not in adjacency_dict:
            adjacency_dict[cluster_key2] = {}
    return adjacency_dict

adjacency2 = update_edges_between_clusters2(clusters2, nodes)

for key1, nested_dict in adjacency2.items():
        for key2, value in nested_dict.items():            
            avg_list = sum(value) / len(value) 
            adjacency2[key1][key2] = avg_list

""" print(len(adjacency2))
for key, value in list(adjacency2.items())[:5]:
    print(key, value) """


def floyd_warshall(graph):
    vertices = list(graph.keys())
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
        print("FLOOOOOOOOYYYYDDDD MAYYYWEATHHHHHHER")
        for i in vertices:
            for j in vertices:
                if dist[nIndex[i]][nIndex[k]] + dist[nIndex[k]][nIndex[j]] < dist[nIndex[i]][nIndex[j]]:
                    dist[nIndex[i]][nIndex[j]] = dist[nIndex[i]][nIndex[k]] + dist[nIndex[k]][nIndex[j]]

    return dist, nIndex

print(f"at this step")
ftime = time.time()
adjacency1 , indices = floyd_warshall(adjacency2)
et = time.time()
elapsed_time = et - ftime
print(f"Matching process complete. Elapsed time: {elapsed_time} seconds")


def match_passenger_to_driver_t4(drivers, passengers, adjacency_matrices, nodes, nIndex):
    start_time = time.time()

    driver_heap = [(datetime.strptime(driver[0], '%m/%d/%Y %H:%M:%S'), driver) for driver in drivers]
    passenger_heap = [(datetime.strptime(passenger[0], '%m/%d/%Y %H:%M:%S'), passenger) for passenger in passengers]

    heapq.heapify(driver_heap)

    while driver_heap and passenger_heap:
        current_driver_time, current_driver = heapq.heappop(driver_heap)
        shortest_path_time = float('inf')
        current_passenger = None
        current_passenger_time = None

        for passenger_time, passenger in passenger_heap:
            #minni = timedelta(minutes=7)
            #max_wait_time = current_driver_time - minni

            if passenger_time < current_driver_time: #and passenger_time > max_wait_time:
                start_node = find_closest_coordinates((float(current_driver[1]), float(current_driver[2])), nodes)
                end_node = find_closest_coordinates((float(passenger[1]), float(passenger[2])), nodes)

                path_time = adjacency_matrices[nIndex[start_node]][nIndex[end_node]]
                
                if path_time < shortest_path_time:
                    shortest_path_time = path_time
                    current_passenger = passenger
                    current_passenger_time = passenger_time

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

match_passenger_to_driver_t4(drivers_matrix[1:], passengers_matrix[1:], adjacency1, clusters2, indices)

def match_passenger_to_driver_t5(drivers, passengers, adjacency_matrices, nodes, nIndex, nodes1, adj):
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

match_passenger_to_driver_t5(drivers_matrix[1:], passengers_matrix[1:], adjacency1, clusters2, indices, clusters, adjacency)