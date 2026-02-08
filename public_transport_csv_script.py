import pandas as pd
import networkx as nx
import numpy as np
import os 
import re

#Functions

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s-]+", "_", s)
    return s 

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def build_edges_mst(df_line):
    g = df_line.dropna(subset=["latitude", "longitude"])
    if len(g) < 2:
        return pd.DataFrame(columns=["nodeID_from","nodeID_to","line_id","mode","year","dist_km"])

    H = nx.Graph()
    for _, r in g.iterrows():
        H.add_node(
            int(r.nodeID),
            lat=float(r.latitude),
            lon=float(r.longitude),
            mode=r.mode,
            year=r.year,
        )

    ids = g["nodeID"].astype(int).tolist()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            ra, rb = H.nodes[a], H.nodes[b]
            w = _haversine_km(ra["lat"], ra["lon"], rb["lat"], rb["lon"])
            H.add_edge(a, b, weight=w)

    T = nx.minimum_spanning_tree(H, weight="weight")

    edges = []
    line_id = g["line_id"].iloc[0]
    for a, b, d in T.edges(data=True):
        na = H.nodes[a]
        edges.append((int(a), int(b), line_id, na["mode"], na["year"], float(d["weight"])))

    edges_df = pd.DataFrame(
        edges,
        columns=["nodeID_from","nodeID_to","line_id","mode","year","dist_km"]
    )

    return edges_df

def remap_ids(nodes_df, edges_df):
    unique_ids = sorted(nodes_df["nodeID"].unique())
    mapping = {old: new for new, old in enumerate(unique_ids, start=1)}
    nodes_new = nodes_df.copy()
    nodes_new["nodeID"] = nodes_new["nodeID"].map(mapping)

    edges_new = edges_df.copy()
    edges_new["nodeID_from"] = edges_new["nodeID_from"].map(mapping)
    edges_new["nodeID_to"]   = edges_new["nodeID_to"].map(mapping)

    return nodes_new, edges_new, mapping

# Import data
cities = pd.read_csv("data/cities.csv")
lines = pd.read_csv("data/lines.csv")
section_lines = pd.read_csv("data/section_lines.csv")
sections = pd.read_csv("data/sections.csv")
stations = pd.read_csv("data/stations.csv")
station_lines = pd.read_csv("data/station_lines.csv")
transport_modes = pd.read_csv("data/transport_modes.csv")


stations_r = stations.rename(columns={"id": "station_id"})
lines_r    = lines.rename(columns={"id": "line_id"})
modes_r    = transport_modes.rename(columns={"id": "transport_mode_id",
                                           "name": "mode"})
cities_r   = cities.rename(columns={"id": "city_id", "name": "city"})

df_geom = (
    station_lines[["station_id", "line_id", "city_id"]]
      .merge(stations_r[["station_id", "city_id", "name", "geometry", "opening"]],
             on=["station_id", "city_id"], how="inner")
      .merge(lines_r[["line_id", "city_id", "transport_mode_id"]],
             on=["line_id", "city_id"], how="inner")
      .merge(modes_r[["transport_mode_id", "mode"]],
             on="transport_mode_id", how="left")
      .merge(cities_r[["city_id", "city"]], on="city_id", how="left")
      [["station_id", "name", "geometry", "mode", "opening", "city"]]
      .drop_duplicates()
      .sort_values(["city", "station_id"])
      .reset_index(drop=True)
)
df_geom = df_geom[["city", "station_id", "name", "geometry", "mode", "opening"]]
m = df_geom["geometry"].str.extract(
    r'POINT\s*\(\s*(?P<lon>-?\d+(?:\.\d+)?)\s+(?P<lat>-?\d+(?:\.\d+)?)\s*\)',
    flags=re.I
)

df_geom["latitude"]  = pd.to_numeric(m["lat"], errors="coerce")
df_geom["longitude"] = pd.to_numeric(m["lon"], errors="coerce")

df = (
    df_geom.rename(columns={"station_id": "nodeID",
                       "name": "nodeLabel",
                       "opening": "year"})
      [["city", "nodeID", "nodeLabel", "latitude", "longitude", "mode", "year"]]
)
os.makedirs("cities", exist_ok=True)
for city in df['city'].unique():
    city_df = df[df['city'] == city].reset_index(drop=True)
    city_df = city_df.merge(station_lines[['station_id','line_id']],
                     left_on='nodeID',
                     right_on='station_id', how='left').drop(columns=['station_id'])
    edges_all = pd.DataFrame(columns=["nodeID_from","nodeID_to","line_id","mode","year"])
    for line_id, grp in city_df.groupby("line_id", dropna=False):
        edges = build_edges_mst(grp)
        edges_all = pd.concat([edges_all, edges], ignore_index=True)
        edges_df = edges_all.drop(columns =['mode','dist_km'])
        edges_df = edges_df.merge(df[['nodeID','mode']], left_on='nodeID_from', right_on='nodeID', how='left').drop(columns=['nodeID'])
        nodes_new, edges_new, mapping = remap_ids(city_df, edges_df)
    fname_nodes = f"{slugify(str(city))}_nodes.csv"
    fname_edges = f"{slugify(str(city))}_edges.csv"
    nodes_new[nodes_new['city'] == city].drop(columns=['city']).to_csv(os.path.join("cities", fname_nodes), index=False)
    edges_new.to_csv(os.path.join("cities", fname_edges), index=False)    


     



