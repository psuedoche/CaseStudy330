# Overview

Case study for 330 

# Data Structures

## adjacency.json

- adjacency.json contains an adjacency list dictionary that represents the connections between different nodes, along with associated attributes for each connection.

The dictionary has the following structure:
~~~
{ 
    start_node_id <ID of the start node, int>: {
        end_node_id <ID of the end node, int>: {
            'day_type': <type of day, string, 'weekday'/'weekend'>,
            'hour': <hour of the day, int, 0-23>,
            'length': <length of the connection, int, converted to miles>,
            'max_speed': <maximum speed, float, converted to miles per hour>,
            'time': <length/max_speed, float, converted to hours>
        },
        ...
    },
    ...
}
~~~
## node_data.json

node_data.json contains a dictionary that provides longitude and latitude coordinates for all nodes.

The dictionary has the following structure:
~~~
{

    node_id <ID of the node, int>: {
        'lon': <longitude coordinate, float>,
        'lat': <latitude coordinate, float>
    },
    ...

}
~~~

## drivers.csv

- Date/Time - Time the driver starts looking for a passenger. Format MM/DD/YYYY HH:MM:SS
- Source Lat - Latitude driver is located at
- Source Lon - Longitude driver is located at

## passengers.csv

- Date/Time - Time the passenger starts looking for a ride. Format MM/DD/YYYY HH:MM:SS
- Source Lat - Start latitude of passenger
- Source Lon - Start longitude of passenger
- Dest Lat - Destination latitude of passenger
- Dest Lon - Destination longitude of passenger

## edges.csv
The edges.csv file contains one row for every road segment. There is a header row naming the columns. The edges are symmetric in the sense that if there is a row for A --> B then there should be a row for B --> A, but the average speeds may differ depending on the direction.

For each row / road segment:
- Column 0: "start_id" contains the node id of the source
- Column 1: "end_id" contains the node id of the destination
- Column 2: "length" contains the distance along the segment in miles
- Columns 3 - 26: "weekday_0" ... "weekday_23" contain the average speed along the road segment in miles per hour on weekdays at different hours. 0 refers to midnight to 1 am...23 refers to the hour before midnight.
- Columns 27 - 50: "weekend_0" ... "weekend_23" contain the average speed along the road segment in miles per hour on weekends at different hours. 0 refers to midnight to 1 am...23 refers to the hour before midnight.


