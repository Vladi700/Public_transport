import networkx as nx
import numpy as np
from collections import defaultdict, deque

from mesa import Model
from mesa.datacollection import DataCollector



class Cohort:
    __slots__ = ("mass", "node", "dest", "route_index") #attributes
    def __init__(self, mass, node, dest, route_index=0):
        self.mass = float(mass)
        self.node = node
        self.dest = dest
        self.route_index = route_index

class QueueModel(Model):
    def __init__(self, 
                 H: nx.Graph,
                 entry_nodes,
                 dest_nodes,
                 schedules,
                 total_target,
                 pos,
                 sigma_xy,
                 center=None,
                 weight='dist_km',
                 tick_seconds=180,
                 cohort_mass=float(20),
                 cap_attr = 'cap_ppl_per_tick',
                 seed=1,
                 alpha=1,
                 beta=1,
                 eps=1e-6,
                 congestion_threshold=100.0,
                 num_alternative_routes=3):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        self.H = H.copy()

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

        self.pos = pos
        self.sigma_xy = sigma_xy
        self.center = np.asarray(center, dtype=float)
        
        
        self.node_queue_sum = defaultdict(float)
        self.edge_flow_sum =defaultdict(float)
        self.dest_nodes = list(dest_nodes)
        self.dest_set = set(self.dest_nodes)
        self.entry_nodes = list(entry_nodes)
        self.cohort_mass = float(cohort_mass)
        self.weight = weight
        self.cap_attr = cap_attr
        self.tick_seconds = tick_seconds
        self.running = True
        self.schedules = schedules   
        self.L_dest = self._precompute_entry_dest_lengths()
        self.entry_p = self._gaussian_entry_probs(self.sigma_xy)
        self.node_queue_max = defaultdict(float)
        self.line_nodes = defaultdict(set)
        for u, v, d in self.H.edges(data=True):
            lid = d.get("line_id")
            if lid is not None:
                self.line_nodes[lid].add(u)
                self.line_nodes[lid].add(v)
        
        self.congestion_threshold = float(congestion_threshold)
        self.congested_lines = set()
        self.line_max_queues = defaultdict(float)
        self.num_alternative_routes = int(num_alternative_routes)

        self.alternative_routes = self._precompute_alternative_routes()

        self.tick = 0
        self.in_system = 0.0
        self.completed = 0.0
        self.injected = 0.0
        self.moved_to = 0.0
        self.injected_target = total_target

        self.waiting = {n: deque() for n in self.H.nodes}
        self.entry_split = {n: 1.0 / len(self.entry_nodes) for n in self.entry_nodes}

        

        self.next_hop = self._compute_next_hop()
        self.closest_dest_of_entry = self._closest_destination()
        def per_line_queue_mass(m):
            out = {}
            for lid, nodes in m.line_nodes.items():
                q = 0.0
                for n in nodes:
                    q += sum(c.mass for c in m.waiting[n])
                out[lid] = q
            return out

        self.datacollector = DataCollector(
        model_reporters={
            "tick": lambda m: m.tick,
            "in_system": lambda m: m.in_system,
            "completed": lambda m: m.completed,
            "total_queued": lambda m: sum(sum(c.mass for c in q) for q in m.waiting.values()),
            "max_node_queue": lambda m: max((sum(c.mass for c in q) for q in m.waiting.values()), default=0.0),
            "median_nonzero_queue": lambda m: (
                float(np.median([sum(c.mass for c in q) for q in m.waiting.values() if sum(c.mass for c in q) > 0]))
                if any(sum(c.mass for c in q) > 0 for q in m.waiting.values())
                else 0.0
            ),
            "global_max_node_queue": lambda m: max(m.node_queue_max.values(), default=0.0),
            "line_queue_mass": per_line_queue_mass,
            "congested_lines": lambda m: list(m.congested_lines)
        }
    )



    def _find_center(self):
        xy = np.array([self.pos[n] for n in self.entry_nodes], dtype=float)
        return xy.mean(axis=0)
    
    def _gaussian_entry_probs(self, sigma_xy: float):
        xy = np.array([self.pos[n] for n in self.entry_nodes], dtype=float)
        r2 = np.sum((xy - self.center)**2, axis=1)
        w = np.exp(-0.5 * r2 / (sigma_xy**2))
        Z = w.sum()
        if not np.isfinite(Z) or Z <= 0:
            return np.ones(len(self.entry_nodes), dtype=float) / len(self.entry_nodes)
        return w / Z

    def dest_probs(self, e, tick, alpha=1.0, beta=1.0, eps=1e-6):
    # candidate destinations reachable from e
            Lmap = self.L_dest.get(e)
            
            dests = np.array(list(Lmap.keys()), dtype=object)
            L    = np.array([Lmap[d] for d in dests], dtype=float)

            lam  = np.array([float(self.schedules[d].get(tick, 0.0)) for d in dests], dtype=float)

            # score = alpha*log(lam+eps) - beta*L
            score = alpha * np.log(lam + eps) - beta * L

            # stable softmax
            score -= score.max()
            w = np.exp(score)
            Z = w.sum()
            if Z <= 0:
                return dests, np.ones_like(w) / len(w)
            p = w / Z
            return dests, p
    
    def lambda_d(self, d, tick):
        return float(self.schedules[d].get(tick))
    
    def _precompute_entry_dest_lengths(self):
        lengths = {}
        for e in self.entry_nodes:
            dist_map = nx.single_source_dijkstra_path_length(self.H, e, weight=self.weight)
            lengths[e] = {d: float(dist_map[d]) for d in self.dest_nodes if d in dist_map}
        return lengths

    def _closest_destination(self):
        closest = {}
        for e in self.entry_nodes:
            best_d = None
            best_L = float("inf")
            for d in self.dest_nodes:
                L = nx.shortest_path_length(self.H, e, d, weight=self.weight)
                if L < best_L:
                    best_L = L
                    best_d = d
                closest[e] = best_d
        return closest
    
    def _compute_next_hop(self):
        tables = {}
        for d in self.dest_nodes:
            nh = {}
            for u in self.H.nodes:
                if u == d:
                    nh[u] = None
                    continue
                p = nx.shortest_path(self.H, u, d, weight=self.weight)
                nh[u] = p[1]
            tables[d] = nh
        return tables 
    
    def _precompute_alternative_routes(self):
        routes = {}
        for dest in self.dest_nodes:
            for node in self.H.nodes:
                if node == dest:
                    continue
                try:
                    paths = list(nx.shortest_simple_paths(
                        self.H, 
                        node, 
                        dest, 
                        weight=self.weight
                    ))
                    for route_idx, path in enumerate(paths[:self.num_alternative_routes]):
                        if len(path) > 1:
                            routes[(node, dest, route_idx)] = path[1]
                        else:
                            routes[(node, dest, route_idx)] = None

                        for route_idx in range(len(paths), self.num_alternative_routes):
                            routes[(node, dest, route_idx)] = None
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # No path exists
                    for route_idx in range(self.num_alternative_routes):
                        routes[(node, dest, route_idx)] = None
                    
        return routes
    
    def get_next_hop(self, node, dest, route_index=0):
        key = (node, dest, route_index)
        if key in self.alternative_routes:
            return self.alternative_routes[key]
        if route_index > 0:
            return self.get_next_hop(node, dest, 0)
        return None
    
    def get_line_for_edge(self, u, v):
        data = self.H.get_edge_data(u, v, default={})
        return data.get("line_id")
    
    def is_line_congested(self, line_id):
        return line_id in self.congested_lines
    
    def get_line_queue_mass(self, line_id):
        q = 0.0
        for n in self.line_nodes[line_id]:
            q += sum(c.mass for c in self.waiting[n])
        return q
    
    def update_congestion_status(self):
        for line_id in self.line_nodes.keys():
            current_q = self.get_line_queue_mass(line_id)
            if current_q > self.line_max_queues[line_id]:
                self.line_max_queues[line_id] = current_q
            if current_q > self.congestion_threshold:
                self.congested_lines.add(line_id)
            elif line_id in self.congested_lines and current_q <= self.congestion_threshold:
                self.congested_lines.remove(line_id)

    def find_uncongested_route(self, node, dest, route_index=0):
        for alt_idx in range(route_index, self.num_alternative_routes):
            next_node = self.get_next_hop(node, dest, alt_idx)
            if next_node is None:
                continue
            line_id = self.get_line_for_edge(node, next_node)
            if not self.is_line_congested(line_id):
                return next_node, alt_idx
            next_node = self.get_next_hop(node, dest, route_index)
            return next_node, route_index

    def edge_capacity(self, u, v):
        data = self.H.get_edge_data(u, v, default={})
        return data.get(self.cap_attr, float('inf'))

    def queue_mass_at_node(self, node):
        return sum(c.mass for c in self.waiting[node])

    def step(self):
        self.update_congestion_status()

        total_expected = sum(float(self.schedules[d].get(self.tick, 0.0)) for d in self.dest_nodes)
        expected_cohorts = total_expected / self.cohort_mass
        k_total = self.rng.poisson(expected_cohorts) if expected_cohorts > 0 else 0
        for _ in range(k_total):
            e = self.rng.choice(list(self.entry_nodes), p=self.entry_p)  

            dests, p = self.dest_probs(e, self.tick)           
            if dests is None:
                continue
            d = self.rng.choice(dests, p=p)

            c = Cohort(self.cohort_mass, e, d)
            self.waiting[e].append(c)
            self.in_system += c.mass
            self.injected  += c.mass
                    
        moved_to = defaultdict(deque) #que of cohorts arriving this tick at v
        for u in list(self.H.nodes):
            if u in self.dest_set:
                continue

            q = self.waiting[u]
            if not q:
                continue

            used_by_v = defaultdict(float)
            while q:
                c = q[0]
                dest = c.dest
                route_idx = c.route_index
                v, new_route_idx = self.find_uncongested_route(u, dest, route_idx)

                if v is None:
                    break

                cap = float(self.edge_capacity(u, v))
                used = used_by_v[v]
                if used >= cap:
                    break
                sendable = cap - used

                if c.mass <= sendable:
                    q.popleft()
                    used_by_v[v] += c.mass
                    sent = c.mass
                    self.edge_flow_sum[(u, v)] += sent
                    c.node = v
                    c.route_index = new_route_idx
                    moved_to[v].append(c)
                else:
                    if sendable <=0:
                        break
                    q.popleft()
                    sent = sendable
                    self.edge_flow_sum[(u, v)] += sent
                    moved_cohort = Cohort(sendable, v, dest, new_route_idx)
                    moved_to[v].append(moved_cohort)
                    used_by_v[v] += sendable

                    remaining = c.mass - sendable
                    c.mass = remaining
                    c.route_index = new_route_idx
                    q.appendleft(c)   # remainder stays, still FIFO
                    break
        for v, dq in moved_to.items():
            self.waiting[v].extend(dq)

        for d in self.dest_set:
            sink_q = self.waiting[d]
            while sink_q:
                c = sink_q.popleft()
                self.completed += c.mass
                self.in_system -= c.mass
        for n, q in self.waiting.items():
            qm = sum(c.mass for c in q)
            self.node_queue_sum[n] += sum(c.mass for c in q)
            if qm > self.node_queue_max[n]:
                self.node_queue_max[n] = qm
        self.datacollector.collect(self)
        self.tick += 1
        if self.in_system <= 1e-9 and self.tick > 0:
            self.running = False


        assert abs(self.completed + self.in_system - self.injected) < 1e-6