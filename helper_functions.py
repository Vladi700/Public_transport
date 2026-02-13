import numpy as np
import networkx as nx
import pandas as pd

def injection_schedule(time_bins, time_share, tick_seconds, bin_seconds = 60 * 60, total_expected=900_000):
    ticks_per_bin = bin_seconds / tick_seconds
    total_ticks = ticks_per_bin * len(time_bins)
    schedule = {}
    for bin_idx in range(len(time_bins)):
        people_per_bin = total_expected * time_share[bin_idx]
        people_per_tick = people_per_bin / ticks_per_bin

        start_tick = int(bin_idx * ticks_per_bin)
        end_tick = int((bin_idx + 1) * ticks_per_bin)
        
        # Assign to each tick in this bin
        for tick in range(start_tick, end_tick):
            schedule[tick] = people_per_tick
    
    return schedule

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def build_edges_mst(
    df_line,
    min_stations_for_loop=5,
    min_radius_km=0.5,
    max_radius_coeff_var=0.35,
    min_perim_diam_ratio=1.8,
):

    g = df_line.dropna(subset=["latitude", "longitude"]).copy()
    if len(g) < 2:
        return pd.DataFrame(
            columns=["nodeID_from", "nodeID_to", "line_id", "mode", "year", "dist_km"]
        )

    line_id = g["line_id"].iloc[0]

    lats = g["latitude"].to_numpy()
    lons = g["longitude"].to_numpy()
    ids  = g["nodeID"].astype(int).to_numpy()

    H = nx.Graph()
    for nid, lat, lon, mode, year in zip(
        ids, lats, lons, g["mode"], g["year"]
    ):
        H.add_node(
            int(nid),
            lat=float(lat),
            lon=float(lon),
            mode=mode,
            year=year
        )

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = int(ids[i]), int(ids[j])
            ra = H.nodes[a]
            rb = H.nodes[b]
            w = _haversine_km(ra["lat"], ra["lon"], rb["lat"], rb["lon"])
            H.add_edge(a, b, weight=w)

    T = nx.minimum_spanning_tree(H, weight="weight")

    center_lat = float(lats.mean())
    center_lon = float(lons.mean())

    radii = np.array([
        _haversine_km(lat, lon, center_lat, center_lon)
        for lat, lon in zip(lats, lons)
    ])
    r_mean = float(radii.mean())
    r_std  = float(radii.std(ddof=0))

    n = len(ids)
    max_dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = _haversine_km(lats[i], lons[i], lats[j], lons[j])
            if d > max_dist:
                max_dist = d
    diameter_km = max_dist

    perimeter_mst_km = float(sum(d["weight"] for _, _, d in T.edges(data=True)))

    is_loop = False
    if (
        len(g) >= min_stations_for_loop and
        diameter_km > 0 and
        r_mean > 0
    ):
        radius_coeff_var  = r_std / r_mean
        perim_diam_ratio  = perimeter_mst_km / diameter_km

        if (
            r_mean >= min_radius_km and
            radius_coeff_var <= max_radius_coeff_var and
            perim_diam_ratio >= min_perim_diam_ratio
        ):
            is_loop = True

    if is_loop:
        degrees = dict(T.degree())
        leaves = [n for n, d in degrees.items() if d == 1]

        if len(leaves) >= 2:
            best_pair = None
            best_w = float("inf")
            for i in range(len(leaves)):
                for j in range(i + 1, len(leaves)):
                    u = leaves[i]
                    v = leaves[j]
                    ru = H.nodes[u]
                    rv = H.nodes[v]
                    w = _haversine_km(ru["lat"], ru["lon"], rv["lat"], rv["lon"])
                    if w < best_w:
                        best_w = w
                        best_pair = (u, v)

            if best_pair is not None:
                u, v = best_pair
                T.add_edge(u, v, weight=best_w)

    edges = []
    for a, b, d in T.edges(data=True):
        na = H.nodes[a]
        edges.append(
            (int(a), int(b), line_id, na["mode"], na["year"], float(d["weight"]))
        )

    edges_df = pd.DataFrame(
        edges,
        columns=["nodeID_from", "nodeID_to", "line_id", "mode", "year", "dist_km"]
    ).drop_duplicates()

    return edges_df

def make_graph(df_nodes, edges):
    G = nx.Graph()
    for _, r in df_nodes.iterrows():
        G.add_node(
            int(r.nodeID),
            label=r.nodeLabel,
            pos=(float(r.longitude), float(r.latitude)),
            mode = r['mode']
        )
    for _, e in edges.iterrows():
        G.add_edge(
            int(e.nodeID_from),
            int(e.nodeID_to),
            line_id=e.line_id,
            dist_km=e.dist_km,
            mode = e['mode']
        )

    return G

def create_multilayer_graph(G, max_dist_km=0.7):

    def get_label(n):
        d = G.nodes[n]
        return d.get("label")

    def get_lonlat(n):
        d = G.nodes[n]
        return d.get("pos")
    
    def get_mode(n):
        d = G.nodes[n]
        return d.get("mode")

    #Make a dictionary mapping (label, mode) to list of nodes
    label_mode_to_nodes = {}
    for n in G.nodes:
        lbl = get_label(n)
        mode = get_mode(n)
        key = (lbl, mode)
        label_mode_to_nodes.setdefault(key, []).append(n)

    #Create multilayer graph H where each node is a (label, mode) pair and edges connect nodes with same label and different modes if they are within max_dist_km
    H = nx.Graph()
    old_to_new = {}
    for (lbl, mode), nodes in label_mode_to_nodes.items():
        if len(nodes) == 1:
            n = nodes[0]
            new_name = (lbl, mode)

            attrs = G.nodes[n].copy()
            attrs["label"] = new_name
            attrs["mode"] = mode
            lon, lat = get_lonlat(n)
            attrs["pos"] = (lon, lat)
            attrs["original_node"] = [n]
            H.add_node(new_name, **attrs)
            old_to_new[n] = new_name
            continue
        
        #Cluster nodes with same label and mode based on distance
        C = nx.Graph()
        for n in nodes:
            C.add_node(n)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                lon1, lat1 = get_lonlat(a)
                lon2, lat2 = get_lonlat(b)
                d = _haversine_km(lat1, lon1, lat2, lon2)
                if d <= max_dist_km:
                    C.add_edge(a, b, dist_km=d)
        
        components = list(nx.connected_components(C))
        for idx, comp in enumerate(components):
            comp = list(comp)
            if len(components) == 1:
                new_name = (lbl, mode)
            else:
                new_name = (f"{lbl}_{idx}", mode)
            lons = []
            lats = []
            years = []

            for n in comp:
                lon, lat = get_lonlat(n)
                lons.append(lon)
                lats.append(lat)
                d = G.nodes[n]
                if "year" in d and pd.notna(d["year"]):
                    years.append(d["year"])
                lon_avg = float(np.mean(lons))
                lat_avg = float(np.mean(lats))

                attrs = {}
                attrs.update(G.nodes[comp[0]])
                attrs["label"] = new_name
                attrs["mode"] = mode
                attrs["pos"] = (lon_avg, lat_avg)
                attrs["lon"] = lon_avg
                attrs["lat"] = lat_avg
                attrs["year_min"] = min(years) if years else None
                attrs["orig_nodes"] = comp
                H.add_node(new_name, **attrs)
                for n in comp:
                    old_to_new[n] = new_name

        #Add edges to H based on edges in G
    for u, v, data in G.edges(data=True):
        u_new = old_to_new[u]
        v_new = old_to_new[v]
        if u_new is None or v_new is None:
            continue
        if u_new == v_new:
            continue
        if H.has_edge(u_new, v_new):
            if "line_id" in data:
               if "line_id" in data:
                existing = H[u_new][v_new].get("line_ids", set())
                existing.add(data["line_id"])
                H[u_new][v_new]["line_ids"] = existing
        else:
            H.add_edge(u_new, v_new, **data)

    # Add transfer edges between the same station with different modes
    # (these represent transfer points in the network)
    label_to_mode_nodes = {}
    for node in H.nodes():
        label = node[0]  # station name
        if label not in label_to_mode_nodes:
            label_to_mode_nodes[label] = []
        label_to_mode_nodes[label].append(node)

    for label, mode_nodes in label_to_mode_nodes.items():
        if len(mode_nodes) > 1:
            # Create edges between different modes at the same station
            for i in range(len(mode_nodes)):
                for j in range(i + 1, len(mode_nodes)):
                    node1 = mode_nodes[i]
                    node2 = mode_nodes[j]
                    if not H.has_edge(node1, node2):
                        # Transfer edge with zero distance
                        H.add_edge(node1, node2, 
                                  transfer=True, 
                                  dist_km=0,
                                  edge_type="transfer")

    return H, label_to_mode_nodes
