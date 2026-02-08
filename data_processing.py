'''
Here we will define all of the functions and tools that we need in order to build the graphs
from the citylines dataset, and perform the data processing and graph generation.
'''
# ------ Necessary packages ---------
import numpy as np 
import pandas as pd
import networkx as nx
import time
# -----------------------------------


# -------- Data import and pre-processing --------
# We need to read all the csv files:
print('Reading data')
# Folder path
folder_path = '/Users/test/Desktop/Master semester 2/Complex networks/Projects/Public transport/data/'

# Cities data
cities_path = 'cities.csv'
cities_df = pd.read_csv(folder_path + cities_path)

# Systems data
systems_path = 'modes.csv'
systems_df = pd.read_csv(folder_path + systems_path)

# Lines data
lines_path = 'lines.csv'
lines_df = pd.read_csv(folder_path + lines_path)

# Sections data
sections_path = 'sections.csv'
sections_df = pd.read_csv(folder_path + sections_path)

# Section lines data
section_lines_path = 'section_lines.csv'
section_lines_df = pd.read_csv(folder_path + section_lines_path)

# Stations data
stations_path = 'stations.csv'
stations_df = pd.read_csv(folder_path + stations_path)

# Station lines
station_lines_path = 'station_lines.csv'
station_lines_df = pd.read_csv(folder_path + station_lines_path)

# Transport modes
transport_modes_path = 'modes.csv'
transport_modes_df = pd.read_csv(folder_path + transport_modes_path)

# Define a single dictionary with access to all the data
all_data = {'cities': cities_df, 
            'systems': systems_df, 'lines': lines_df,
            'sections': sections_df, 'section_lines': section_lines_df,
            'stations': stations_df, 'station_lines': station_lines_df,
            'transport_modes': transport_modes_df}

# -------------------------------------------



# ------------ Data functions ---------------
# Function to retrieve all the data relative to a city, given its id
def data_from_city_id(city_id, datasets=all_data):
    '''
    Params:
        city_id : int
            Identifier of a city present in the citylines dataset.
    Output:
        Returns a dictionary with the relevant data from the given
        city.
    '''
    # Init results
    res = dict()
    
    # Get city data
    city_data = datasets['cities'].loc[datasets['cities']['id'] == city_id].iloc[0]
    res['city'] = city_data

    # Get systems data
    sys_df = datasets['systems']
    sys_data = sys_df.loc[sys_df.city_id == city_id]
    res['systems'] = sys_data

    # Get lines data
    lines_data = datasets['lines'].loc[datasets['lines'].city_id == city_id]
    res['lines'] = lines_data

    # Get sections data
    sect_df = datasets['sections']
    sect_data = sect_df.loc[sect_df.city_id == city_id]
    res['sections'] = sect_data

    # Get section lines data
    sect_lines_df = datasets['section_lines']
    sect_lines_data = sect_lines_df.loc[sect_lines_df.city_id == city_id]
    res['section_lines'] = sect_lines_data

    # Get stations data
    stat_df = datasets['stations']
    stat_data = stat_df.loc[stat_df.city_id == city_id]
    res['stations'] = stat_data

    # Get station lines data
    stat_lines_df = datasets['station_lines']
    stat_lines_data = stat_lines_df.loc[stat_lines_df.city_id == city_id]
    res['station_lines'] = stat_lines_data

    # Get all transport modes
    modes_df = datasets['transport_modes']
    res['modes'] = modes_df
    
    return res


# Function that checks if either station data or section data is missing
def check_missing_data(city_data):
    '''
    Params:
        city_data : dict
            Data of a given city, with at least "stations" and
            "sections" keys.
    Output:
        Returns True if either the stations data or the sections
        data is missing for the given city, and returns False
        otherwise.
    '''
    no_stations = city_data['stations'].shape[0] == 0
    no_sections = city_data['sections'].shape[0] == 0
    if no_stations or no_sections:
        return True
        
    return False


# Function to extract latitude and longitude from geometry data
def get_point_location(point):
    '''
    Params:
        point : str
            Point location of a station as defined in the 
            geometry feature of the citylines dataset.
    Output:
        Returns a tuple (latitude, longitude) with the 
        location of the given point.
    '''
    # Remove geometry notation
    lat_long = point.replace('POINT(', '')
    lat_long = lat_long.replace(')', '')
    # Split data
    lat_long = lat_long.split()
    # Convert to numerical and return 
    lat = float(lat_long[0])
    long = float(lat_long[1])
    return lat, long


# Function to extract latitude and longitude values from section 
# geometry
def get_line_locations(line):
    '''
    Params:
        line : str
            Line geometry of a section, as defined in the
            geometry feature of the citylines dataset.
    Output:
        Returns a tuple (latitudes, longitudes), where 
        latitudes and longitudes are arrays that contain
        the locations of the given line.
    '''
    # Remove geometry notation
    lat_long = line.replace('LINESTRING', '')
    lat_long = lat_long.replace('(', '')
    lat_long = lat_long.replace(')', '')
    # Get different locations
    lat_long = lat_long.split(',')
    # Convert to numerical and return, if possible
    try:
        lat = [float(l.split()[0]) for l in lat_long]
        long = [float(l.split()[1]) for l in lat_long]
    except ValueError:
        lat = []
        long = []
    return lat, long 


# Function to get the closest distance between a point and a
# section
def get_dist_section_point(x, section_x):
    '''
    Params:
        x : numpy array
            Array of shape (2,) with the latitude and
            longitude of a point.
        section_x : numpy array
            Array of shape (2,N) with the latitude and
            longitude values of a section.
    Output:
        Return the smallest distance between x and any
        of the points in the section.
    '''
    x = np.reshape(x, (x.shape[0], 1))
    dist = np.sqrt(np.sum(np.square(x - section_x), axis=0))
    min_dist = np.min(dist)
    return min_dist



# Function that builds the node and edge lists from the data
# of a given city
def build_node_edge_lists(city_data, to_year=2025, verbose=False):
    '''
    Params:
        city_data : dict
            Data of a given city.
        to_year : int (optional)
            Reference year. Every station closed before this 
            year is not considered as a node. Default value
            is 2025.
        verbose : bool (optional)
            Whether to print out information while running,
            this is turned off by default.
    Output:
        Returns a tuple (node_list, edge_list), where both
        node_list and edge_list are dictionaries containing
        the following data:
        - node_list:
            - nodeID: Unique identifier of the node.
            - nodeLabel: Name of the station.
            - latitude
            - longitude
            - mode: Transport mode of the station.
            - year: Opening year.
        - edge_list:
            - nodeID_from, nodeID_to: Station identifiers.
            - mode: Transport mode of the line.
            - line: Name of the system + name of the line.
            - year: Opening year.
    '''
    # Init empty node list and edge list
    node_list = {'nodeID':[], 'nodeLabel':[], 
                 'latitude':[], 'longitude': [],
                 'mode':[], 'year': []}
    edge_list = {'nodeID_from': [], 'nodeID_to': [], 
                 'mode': [], 'line': [], 'year': []}

    # Init global id counter
    global_id = 0

    # Get unique modes of transport
    modes = city_data['lines']['transport_mode_id'].unique()
    mode_names = dict()
    for m in modes:
        m_df = city_data['modes']
        mode_names[m] = m_df.loc[m_df['id'] == m, 'name'].iloc[0]

    # Get systems data
    s_df = city_data['systems']

    # Build network considering one mode at a time
    for m in modes:
        # Get lines corresponding to the mode
        l_df = city_data['lines']
        l_df = l_df.loc[l_df['transport_mode_id'] == m].copy()

        # Drop unnecessary data
        l_df.drop(columns=['url_name', 'color'], inplace=True)

        # Operate for each line
        for i in range(l_df.shape[0]):
            l = l_df.iloc[i]
            if verbose:
                print('\nLine:')
                print(l)

            # Get name of system + name of line
            line_name = s_df.loc[s_df['id'] == l['system_id'], 'name'].iloc[0]
            line_name = str(line_name) + ' - ' + l['name']

            # Get station-lines data
            st_l = city_data['station_lines']
            st_l = st_l.loc[st_l['line_id'] == l.id].copy()
            st_l.drop(columns=['city_id', 'created_at',
                               'deprecated_line_group',
                               'updated_at', 'toyear', 
                               'fromyear'],
                      inplace=True)
            
            # Get corresponding nodes for the line
            nodes_l = city_data['stations']
            nodes_l = nodes_l.loc[nodes_l.id.isin(st_l.station_id)].copy()
            nodes_l.drop(columns=['buildstart', 'city_id'], 
                         inplace=True)
            
            # Keep stations that have not been closed up to the ref year
            nodes_l = nodes_l.loc[nodes_l.closure >= to_year].copy()

            # Get latitude and longitude for the nodes
            f_lat = lambda s: get_point_location(s)[0]
            f_long = lambda s: get_point_location(s)[1]
            nodes_l['latitude'] = nodes_l.geometry.apply(f_lat)
            nodes_l['longitude'] = nodes_l.geometry.apply(f_long)
            nodes_l.drop(columns=['geometry', 'closure'], inplace=True)
            if verbose:
                print('\nNodes:')
                print(nodes_l.head())

            # Get section-lines data
            sc_l = city_data['section_lines']
            sc_l = sc_l.loc[sc_l.line_id == l.id].copy()
            sc_l.drop(columns=['created_at', 'updated_at', 'city_id',
                               'fromyear', 'toyear', 
                               'deprecated_line_group'], 
                      inplace=True)

            # Get edges data from sections
            edges_l = city_data['sections']
            edges_l = edges_l.loc[edges_l.id.isin(sc_l.section_id)].copy()
            edges_l.drop(columns=['city_id', 'buildstart'], inplace=True)

            # Keep sections that have not been closed up to the ref year
            edges_l = edges_l.loc[edges_l.closure >= to_year].copy()

            # Get latitude and longitude values
            f_lat = lambda s: get_line_locations(s)[0]
            f_long = lambda s: get_line_locations(s)[1]
            edges_l['latitude'] = edges_l.geometry.apply(f_lat)
            edges_l['longitude'] = edges_l.geometry.apply(f_long)
            edges_l.drop(columns=['closure', 'geometry'], inplace=True)

            # Get section positions as arrays
            sect_pos = [
                np.array([edges_l.iloc[e]['latitude'], edges_l.iloc[e]['longitude']])\
                for e in range(edges_l.shape[0])
            ]
            
            if verbose:
                print('\nEdges:')
                print(edges_l.head())

            # Build nodes
            node_ids = []
            for n in range(nodes_l.shape[0]):
                nl = nodes_l.iloc[n]
                # ID
                node_list['nodeID'].append(global_id)
                node_ids.append(global_id)
                # Name
                node_list['nodeLabel'].append(nl['name'])
                # Location
                node_list['latitude'].append(nl['latitude'])
                node_list['longitude'].append(nl['longitude'])
                # Mode and year
                node_list['mode'].append(mode_names[m])
                node_list['year'].append(nl['opening'])

                # Update global id
                global_id += 1

            # Keep track of node IDs for when building edges
            nodes_l['nodeID'] = node_ids

            # Only build edges if there is sections data and 
            # if there is more than one node
            if edges_l.shape[0] == 0 or nodes_l.shape[0] < 2:
                continue
            
            # Build edges with a reasonable assumption:
            # Each node attemps to connect to its two closest
            # neighbours in the same line, that connection is 
            # accepted unless:
            # - Those two neighbours are closer to each other 
            #   than to the reference node.
            #   -> Only attempt connetion to closest node.
            # - The connections are already established.
            #   -> Avoid repeated connections.
            # Given the small spatial extension of cities, we
            # can apply Euclidean distance
            
            edge_arr = []
            for n1 in range(nodes_l.shape[0]):
                nl1 = nodes_l.iloc[n1]
                pos1 = np.array([nl1['latitude'],
                                 nl1['longitude']])

                # If there are only two nodes in total, then attempt
                # connection between them by default
                if n1 == 0 and nodes_l.shape[0] == 2:
                    edge_arr.append([0,1])
                    continue
                elif n1 != 0 and nodes_l.shape[0] == 2:
                    continue
                
                # Get all distances
                dist_vals = []
                for n2 in range(nodes_l.shape[0]):
                    # Ignore self-interaction
                    if n2 == n1:
                        dist_vals.append(1e7)
                    else:
                        # Get distance
                        nl2 = nodes_l.iloc[n2]
                        pos2 = np.array([nl2['latitude'],
                                         nl2['longitude']])
                        dist_vals.append(np.linalg.norm(pos2 - pos1))
                
                # Retrieve two closest nodes
                dA, dB = np.partition(dist_vals, 1)[0:2] 
                nA, nB = dist_vals.index(dA), dist_vals.index(dB)

                # Check for conditions:
                # Link A
                A_link_repeated = ([n1, nA] in edge_arr) or ([nA, n1] in edge_arr)
                if not A_link_repeated:
                    edge_arr.append([n1, nA])
                # Link B
                B_link_repeated = ([n1, nB] in edge_arr) or ([nB, n1] in edge_arr)
                if not B_link_repeated:
                    # Distance condition
                    posA = np.array([nodes_l.iloc[nA]['latitude'],
                                     nodes_l.iloc[nA]['longitude']])
                    posB = np.array([nodes_l.iloc[nB]['latitude'],
                                     nodes_l.iloc[nB]['longitude']])
                    dist_AB = np.linalg.norm(posA - posB)
                    if dist_AB > dB:
                        edge_arr.append([n1, nB])


            # Attempt additional edges from a second criterion:
            # - Two nodes are at the ends of a given section 
            # - There is no link between them
            # - They are not communicated through other nodes 
            # -> Then connect their two closest neighbors
            for s in range(edges_l.shape[0]):
                sl = edges_l.iloc[s]

                if len(sl['latitude']) <= 1:
                    continue

                # Get endpoints 
                sl1 = np.array([sl['latitude'][0], 
                                sl['longitude'][0]])
                sl2 = np.array([sl['latitude'][-1], 
                                sl['longitude'][-1]])

                # Get points closest to the section
                dist1 = []
                dist2 = []
                for n in range(nodes_l.shape[0]):
                    npos = np.array([nodes_l.iloc[n]['latitude'],
                                     nodes_l.iloc[n]['longitude']])
                    dist1.append(np.linalg.norm(sl1 - npos))
                    dist2.append(np.linalg.norm(sl2 - npos))

                n1 = np.argmin(dist1)
                n2 = np.argmin(dist2)

                # Check that the nodes are not equal
                if nodes_l.iloc[n1]['id'] == nodes_l.iloc[n2]['id']:
                    continue

                # Check that the nodes are not communicated 
                # Get nodes in indirect contact from n1
                n1_neighs = []
                for link in edge_arr:
                    if link[0] == n1:
                        n1_neighs.append(link[1])
                    elif link[1] == n1:
                        n1_neighs.append(link[0])

                for link in edge_arr:
                    if link[0] in n1_neighs:
                        n1_neighs.append(link[1])
                    elif link[1] in n1_neighs:
                        n1_neighs.append(link[0])

                n2_neighs = []
                for link in edge_arr:
                    if link[0] == n2:
                        n2_neighs.append(link[1])
                    elif link[1] == n2:
                        n2_neighs.append(link[0])

                for link in edge_arr:
                    if link[0] in n2_neighs:
                        n2_neighs.append(link[1])
                    elif link[1] in n2_neighs:
                        n2_neighs.append(link[0])

                common_neighs = [neigh1 in n2_neighs for neigh1 in n1_neighs]
                if True in common_neighs:
                    continue

                # Attempt link between their two closest neighbors:
                # Get distances between neighbor sets
                neighs_dist = []
                for nn1 in n1_neighs:
                    nn1_pos = np.array([nodes_l.iloc[nn1]['latitude'],
                                        nodes_l.iloc[nn1]['longitude']])
                    nn_dist = []
                    for nn2 in n2_neighs:
                        nn2_pos = np.array([nodes_l.iloc[nn2]['latitude'],
                                            nodes_l.iloc[nn2]['longitude']])
                        nn_dist.append(np.linalg.norm(nn1_pos - nn2_pos))
                    neighs_dist.append(nn_dist)
                neighs_dist = np.array(neighs_dist)

                # Get closest negihbors from the two sets
                nn1_closest, nn2_closest = np.where(neighs_dist == np.min(neighs_dist))
                nn_closest = [n1_neighs[nn1_closest[0]], n2_neighs[nn2_closest[0]]]
                        
                link_repeated = (nn_closest in edge_arr) or ([nn_closest[1], nn_closest[0]] in edge_arr)
                if not link_repeated:
                    edge_arr.append(nn_closest)


            # To avoid spurious connections between sections, we will
            # perform a rewiring of the edges. We will delete a link
            # between any two nodes if a shorter link can be done with
            # one of their neighbors, or in between neighbors.
            rewiring_done = False
            while not rewiring_done:
                old_edge_arr = edge_arr.copy()
                to_remove = []
                for oi in range(len(old_edge_arr)):
                    old_e = old_edge_arr[oi]

                    # Get participating nodes and their positions
                    n1, n2 = old_e[0], old_e[1]

                    nl1 = nodes_l.iloc[n1]
                    pos1 = np.array([nl1['latitude'],
                                    nl1['longitude']])

                    nl2 = nodes_l.iloc[n2]
                    pos2 = np.array([nl2['latitude'],
                                    nl2['longitude']])

                    # Get reference distance between them
                    ref_dist = np.linalg.norm(pos2 - pos1)

                    # Get neighbors
                    n_neighs = []
                    for link in edge_arr:
                        if link[0] in [n1,n2] and not link[1] in [n1, n2]:
                            n_neighs.append(link[1])
                        elif link[1] in [n1, n2] and not link[0] in [n1, n2]:
                            n_neighs.append(link[0])

                    # Get all possible distances among reference
                    # nodes and the neighbors
                    neighs_dist = []
                    for n_ref in [n1, n2]:
                        n_ref_dist = []
                        # Get reference position
                        if n_ref == n1:
                            n_ref_pos = pos1 
                        else:
                            n_ref_pos = pos2

                        for nn in n_neighs:
                            # Avoid self-interactions
                            if nn in [n1, n2]:
                                continue
                            nn_pos = np.array([nodes_l.iloc[nn]['latitude'],
                                                nodes_l.iloc[nn]['longitude']])
                        
                            n_ref_dist.append(np.linalg.norm(n_ref_pos - nn_pos))
                        neighs_dist.append(n_ref_dist)
                    neighs_dist = np.array(neighs_dist)

                    # Proceed with rewiring only if there 
                    # where neighbors present, and if there
                    # is a link shorter than the reference
                    if neighs_dist.shape[1] == 0:
                        continue
                    if np.min(neighs_dist) > ref_dist:
                        continue

                    # Get closest nodes that are not already connected
                    try_dist = neighs_dist[neighs_dist < ref_dist].flatten()
                    for td in try_dist:
                        nn1_closest, nn2_closest = np.where(neighs_dist == td)
                        nn_closest = [[n1, n2][nn1_closest[0]], n_neighs[nn2_closest[0]]]
                                
                        link_repeated = (nn_closest in edge_arr) or ([nn_closest[1], nn_closest[0]] in edge_arr)
                        if not link_repeated:
                            edge_arr.append(nn_closest)
                            # Save which links to remove, in reverse order to avoid problems
                            to_remove.insert(0, oi)

                            break
                
                # After rewiring, remove redundant edges
                for oi in to_remove:
                    edge_arr.pop(oi)

                # Check if no more links were rewired and finish process
                if len(to_remove) == 0:
                    rewiring_done = True

            # Now, we can store the edge data
            for e in edge_arr:
                # Nodes participating
                n1 = nodes_l.iloc[e[0]]
                n2 = nodes_l.iloc[e[1]]

                # Add node relevant data
                edge_list['nodeID_from'].append(n1['nodeID']) 
                edge_list['nodeID_to'].append(n2['nodeID'])
                edge_list['mode'].append(mode_names[m])
                edge_list['line'].append(line_name)

                # Finally, the most difficult attribute
                # to retrieve is the year. We do this 
                # by finding the opening year of the 
                # corresponding sections:
                # Get closest sections
                pos1 = np.array([n1['latitude'], n1['longitude']])
                dist_vals1 = [
                    get_dist_section_point(pos1, s_pos)\
                    for s_pos in sect_pos\
                    if s_pos.shape[1] > 0
                ]
                try:
                    pos1_sec_open = edges_l.iloc[np.argmin(dist_vals1)]['opening']
                    open_year1 = int(pos1_sec_open)
                except ValueError:
                    open_year1 = -1

                pos2 = np.array([n2['latitude'], n2['longitude']])
                dist_vals2 = [
                    get_dist_section_point(pos2, s_pos)\
                    for s_pos in sect_pos\
                    if s_pos.shape[1] > 0
                ]
                try:
                    pos2_sec_open = edges_l.iloc[np.argmin(dist_vals2)]['opening']
                    open_year2 = int(pos2_sec_open)
                except ValueError:
                    open_year2 = -1
                
                # If no opening year is available from sections, use those
                # from the stations
                if open_year1 == -1 and open_year2 == -1:
                    open_year = max(n1['opening'], n2['opening'])
                else:
                    open_year = max(open_year1, open_year2)

                edge_list['year'].append(open_year)

    
    # We need to consider nodes without lines
    nodes_nl = city_data['stations']
    nodes_nl = nodes_nl.loc[~nodes_nl.id.isin(city_data['station_lines']['station_id'])]

    # Keep stations that have not been closed up to the ref year
    nodes_nl = nodes_nl.loc[nodes_nl.closure >= to_year].copy()

    # Get latitude and longitude for the nodes
    f_lat = lambda s: get_point_location(s)[0]
    f_long = lambda s: get_point_location(s)[1]
    nodes_nl['latitude'] = nodes_nl.geometry.apply(f_lat)
    nodes_nl['longitude'] = nodes_nl.geometry.apply(f_long)
    nodes_nl.drop(columns=['geometry', 'closure'], inplace=True)
    
    for n in range(nodes_nl.shape[0]):
        nl = nodes_nl.iloc[n]
        # ID
        node_list['nodeID'].append(global_id)
        # Name
        node_list['nodeLabel'].append(nl['name'])
        # Location
        node_list['latitude'].append(nl['latitude'])
        node_list['longitude'].append(nl['longitude'])
        # Mode and year
        node_list['mode'].append('None')
        node_list['year'].append(nl['opening'])

        # Update global id
        global_id += 1
    
    return node_list, edge_list
# -------------------------------------------



# ------------ MAIN PROCESSING --------------
# Run the main processing loop
# We only consider cities with stations and sections data, 
# otherwise, the city files are built as empty
folder_path = '/Users/test/Desktop/Master semester 2/Complex networks/Projects/Public transport/data'
for c_count, city_id in enumerate(cities_df['id']):
    print('Processing city %d out of %d' % (c_count+1, cities_df.shape[0]))
    # Get name of the city
    city_name = cities_df.loc[cities_df['id'] == city_id, 'name'].iloc[0]
    
    # Check if the name of the city is repeated
    name_counts = cities_df.loc[cities_df['name'] == city_name].shape[0]
    if name_counts > 1:
        # If repeated, add name of the country
        country_name = cities_df.loc[cities_df['id'] == city_id, 'country'].iloc[0]

        # Check if the city is repeated in the same country
        country_counts = cities_df.loc[
            (cities_df['name'] == city_name) &\
            (cities_df['country'] == country_name)].shape[0]
        if country_counts > 1:
            city_name = city_name + '_' + str(city_id)

        city_name = city_name + '_' + country_name

    # Remove possible spaces from name
    city_name = city_name.replace(' ', '_')

    # Get city data
    city_data = data_from_city_id(city_id)

    # If either station data or section data is missing, then skip city
    missing_data = check_missing_data(city_data)
    if missing_data:
        # Write empty city files
        city_nodes = pd.DataFrame(
                {'nodeID':[], 'nodeLabel':[], 
                 'latitude':[], 'longitude': [],
                 'mode':[], 'year': []})
        city_edges = pd.DataFrame(
                {'nodeID_from': [], 'nodeID_to': [], 
                 'mode': [], 'line': [], 'year': []})

    else:
        # Build node and edge lists
        node_list, edge_list = build_node_edge_lists(city_data)
        city_nodes = pd.DataFrame(node_list)
        city_edges = pd.DataFrame(edge_list)

    # Write files
    city_nodes.to_csv(folder_path + city_name + '_nodes.csv',
                     index=False)
    city_edges.to_csv(folder_path + city_name + '_edges.csv', 
                     index=False)

print('Process completed')
exit()