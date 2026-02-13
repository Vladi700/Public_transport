import networkx as nx
import numpy as np
from collections import defaultdict, deque

from mesa import Model
from mesa.datacollection import DataCollector



class Cohort:
    __slots__ = ("mass", "node", "dest") #attributes
    def __init__(self, mass, node, dest):
        self.mass = float(mass)
        self.node = node
        self.dest = dest

class QueueModel(Model):
    def __init__(self, 
                 H: nx.Graph,
                 entry_nodes,
                 dest_nodes,
                 injection_schedule,
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
                 eps=1e-6):
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
        self.injection_schedule = injection_schedule
        self.injection_schedule = injection_schedule
        self.L_dest = self._precompute_entry_dest_lengths()
        self.entry_p = self._gaussian_entry_probs(self.sigma_xy)
        self.node_queue_max = defaultdict(float)
        self.line_nodes = defaultdict(set)
        for u, v, d in self.H.edges(data=True):
            lid = d.get("line_id")
            if lid is not None:
                self.line_nodes[lid].add(u)
                self.line_nodes[lid].add(v)

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
            "line_queue_mass": per_line_queue_mass
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

    
    def _precompute_entry_dest_lengths(self):
        lengths = {}
        for e in self.entry_nodes:
            dist_map = nx.single_source_dijkstra_path_length(self.H, e, weight=self.weight)
            lengths[e] = {d: dist_map.get(d, float('inf')) for d in self.dest_nodes}
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
    
    
    def edge_capacity(self, u, v):
        data = self.H.get_edge_data(u, v, default={})
        return data.get(self.cap_attr, float('inf'))

    def queue_mass_at_node(self, node):
        return sum(c.mass for c in self.waiting[node])

    def step(self):
        while self.running == True:
            print(f"Tick {self.tick}: in_system={self.in_system:.2f}, completed={self.completed:.2f}, injected={self.injected:.2f}")
            total_expected = float(self.injection_schedule.get(self.tick, 0.0))
            expected_cohorts = total_expected / self.cohort_mass
            k_total = self.rng.poisson(expected_cohorts) if expected_cohorts > 0 else 0
            for _ in range(k_total):
                e_idx = self.rng.choice(len(self.entry_nodes), p=self.entry_p)
                e = self.entry_nodes[e_idx]  
                if e not in self.L_dest or not self.L_dest[e]:
                    continue
                dests = sorted(self.dest_nodes, key=lambda d: self.L_dest[e].get(d, float('inf')))        
                if not dests:
                    continue
                d = dests[0]

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

                    v = self.next_hop.get(dest, {}).get(u)
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
                        moved_to[v].append(c)
                    else:
                        if sendable <=0:
                            break
                        q.popleft()
                        sent = sendable
                        self.edge_flow_sum[(u, v)] += sent
                        moved_to[v].append(Cohort(sendable, v, dest))
                        used_by_v[v] += sendable

                        remaining = c.mass - sendable
                        c.mass = remaining
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