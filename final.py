import csv
import json
import time
import heapq
from datetime import datetime, timedelta
from math import sqrt

import heapq

class PriorityQueue:
    def __init__(self):
        self.dict = {}  
        self.heap = []     

    def pop(self):
        if len(self.heap) != 0:
            priority, node = heapq.heappop(self.heap)
            del self.dict[node]
            return node

    def push(self, node, priority):
        self.dict[node] = priority
        heapq.heappush(self.heap, (priority, node))

    def decrease_key(self, node, new_priority):
        if new_priority < self.dict[node] and node in self.dict:
            self.dict[node] = new_priority
            self.heap = heapq.heapify([(p, e) for e, p in self.dict.items()])


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
for node_id, coordinates in node_data_dict.items():
    print(f'Node ID: {node_id}, Coordinates: {coordinates}')
    count += 1
    if count == 5:
        break

def match_passenger_to_driver_t1(drivers, passengers):
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
        # remove conditional, assign first passenger to first driver
        print(f"Driver assigned to Passenger: {current_driver} -> {current_passenger}")
        drive_duration = timedelta(minutes=20)
        # requeue driver 
        heapq.heappush(driver_heap, (current_driver_time + drive_duration, current_driver))
        """ if current_passenger_time < current_driver_time:
            print(f"Driver assigned to Passenger: {current_driver} -> {current_passenger}")
        else:
            # If not, put the passenger back in the heap for later matching
            heapq.heappush(passenger_heap, (current_passenger_time, current_passenger)) """
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Matching process complete. Elapsed time: {elapsed_time} seconds")
 #04/25/2014 07:00:00
match_passenger_to_driver_t1(drivers_matrix[1:], passengers_matrix[1:])

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
match_passenger_to_driver_t2(drivers_matrix[1:], passengers_matrix[1:])

def match_passenger_to_driver_t2(drivers, passengers):
    start_time = time.time()
    # Convert dates to datetime objects and initialize priority queues
    driver_heap = [(datetime.strptime(driver[0], '%m/%d/%Y %H:%M:%S'), driver) for driver in drivers]
    passenger_heap = [(datetime.strptime(passenger[0], '%m/%d/%Y %H:%M:%S'), passenger) for passenger in passengers]

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
match_passenger_to_driver_t2(drivers_matrix[1:], passengers_matrix[1:])