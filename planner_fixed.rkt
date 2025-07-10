#lang dssl2


# Final project: Trip Planner

# Honor code
let eight_principles = ["Know your rights.",
                       "Acknowledge your sources.",
                       "Protect your work.",
                       "Avoid suspicion.",
                       "Do your own work.",
                       "Never falsify a record or permit another person to do so.",
                       "Never fabricate data, citations, or experimental results.",
                       "Always tell the truth when discussing your work with your instructor."]

import cons
import 'project-lib/graph.rkt'
import 'project-lib/binheap.rkt'
import 'project-lib/dictionaries.rkt'

### Basic Types ###

#  - Latitudes and longitudes are numbers:
let Lat?  = num?
let Lon?  = num?

#  - Point-of-interest categories and names are strings:
let Cat?  = str?
let Name? = str?

### Raw Item Types ###

#  - Raw positions are 2-element vectors with a latitude and a longitude
let RawPos? = VecKC[Lat?, Lon?]

#  - Raw road segments are 4-element vectors with the latitude and
#    longitude of their first endpoint, then the latitude and longitude
#    of their second endpoint
let RawSeg? = VecKC[Lat?, Lon?, Lat?, Lon?]

#  - Raw points-of-interest are 4-element vectors with a latitude, a
#    longitude, a point-of-interest category, and a name
let RawPOI? = VecKC[Lat?, Lon?, Cat?, Name?]

### Contract Helpers ###

# ListC[T] is a list of `T`s (linear time):
let ListC = Cons.ListC
# List of unspecified element type (constant time):
let List? = Cons.list?

# Helper: reverse a linked list of cons‐nodes
def reverse_list(lst):
    let result = None
    let cur = lst
    while cur != None:
        result = cons(cur.data, result)
        cur = cur.next
    result

# Helper: convert a DSSL2 vec into a Racket list‐cons (ListC)
def to_cons(lst):
    let result = None
    let pos = lst.len() - 1
    while pos >= 0:
        result = cons(lst.get(pos), result)
        pos = pos - 1
    result

# Compute Euclidean distance between two (lat, lon) points
def euclidean_distance(lat1, lon1, lat2, lon2):
    ((lat1 - lat2) * (lat1 - lat2) + (lon1 - lon2) * (lon1 - lon2)) ** 0.5

# Convert a latitude and longitude into a string key
def pos_key(lat, lon):
    str(lat) + "," + str(lon)

# Simple hash function for strings (for HashTable)
def simple_hash(s):
    let h = 0
    let counter = 0
    while counter < len(s):
        # Use the character index as a simple hash value
        h = (h * 31 + counter + 1) % 1000
        counter = counter + 1
    h
def to_vec(lst):
        let count = 0
        let walker = lst
        while walker != None:
            count = count + 1
            walker = walker.next

        let arr = vec(count)
        let i = 0
        walker = lst
        while walker != None:
            arr.put(i, walker.data)
            walker = walker.next
            i = i + 1

        arr


interface TRIP_PLANNER:
    def locate_all(self, dst_cat: Cat?) -> ListC[RawPos?]
    def plan_route(self, src_lat: Lat?, src_lon: Lon?, dst_name: Name?) -> ListC[RawPos?]
    def find_nearby(self, src_lat: Lat?, src_lon: Lon?, dst_cat: Cat?, n: nat?) -> ListC[RawPOI?]

# Your TripPlanner class goes below this line

class TripPlanner (TRIP_PLANNER):
    let positions       # vec of [lat, lon]
    let pos_to_index    # HashTable mapping "lat,lon" → index
    let graph           # WUGraph of size = positions.len()
    let poi_by_name     # HashTable mapping POI name → [index, category, name]
    let poi_by_cat      # HashTable mapping category → cons‐list of [index, category, name]
    
    def __init__(self, segs, pois):
        self.pos_to_index = HashTable(100, simple_hash)
        self.poi_by_name  = HashTable(100, simple_hash)
        self.poi_by_cat   = HashTable(50, simple_hash)

        let seen_positions = self.pos_to_index
        let pos_list = None        # linked list of unique [lat, lon]
        let next_index = 0

        # 1) One pass over all segment endpoints
        for seg in segs:
            for i in [0, 2]:
                let lat = seg[i]
                let lon = seg[i + 1]
                let key = pos_key(lat, lon)
                if not seen_positions.mem?(key):
                    seen_positions.put(key, next_index)
                    pos_list = cons([lat, lon], pos_list)
                    next_index = next_index + 1

        # 2) One pass over all POIs
        for poi in pois:
            let lat = poi[0]
            let lon = poi[1]
            let key = pos_key(lat, lon)
            if not seen_positions.mem?(key):
                seen_positions.put(key, next_index)
                pos_list = cons([lat, lon], pos_list)
                next_index = next_index + 1

        # Reverse pos_list to restore discovery order
        let rev_list = reverse_list(pos_list)

        # Count how many unique positions there are
        let count = 0
        let temp = rev_list
        while temp != None:
            count = count + 1
            temp = temp.next

        # Create a fixed-size vector of length count
        let pos_vec = vec(count)
        let idx = 0
        let walker = rev_list
        while walker != None:
            pos_vec.put(idx, walker.data)   # walker.data is [lat, lon]
            walker = walker.next
            idx = idx + 1

        self.positions = pos_vec

        # Build graph
        self.graph = WUGraph(self.positions.len())
        for seg in segs:
            let key1 = pos_key(seg[0], seg[1])
            let key2 = pos_key(seg[2], seg[3])
            let idx1 = self.pos_to_index.get(key1)
            let idx2 = self.pos_to_index.get(key2)
            let d = euclidean_distance(seg[0], seg[1], seg[2], seg[3])
            self.graph.set_edge(idx1, idx2, d)

        # Build POI lookup tables
        for poi in pois:
            let lat = poi[0]
            let lon = poi[1]
            let cat = poi[2]
            let name = poi[3]
            let idxp = self.pos_to_index.get(pos_key(lat, lon))
            let packed = [idxp, cat, name]

            # Store by name
            self.poi_by_name.put(name, packed)

            # Store by category (prepend to linked list)
            if not self.poi_by_cat.mem?(cat):
                self.poi_by_cat.put(cat, None)
            let old_list = self.poi_by_cat.get(cat)
            self.poi_by_cat.put(cat, cons(packed, old_list))
            # Helper: convert a linked list (cons nodes) into a vector
    
    def _dijkstra(self, start):
        let dist = HashTable(100, lambda x: x)
        let visited = HashTable(100, lambda x: x)
        let compare = lambda a, b: a[0] < b[0]
        let pq = BinHeap(self.graph.n_vertices() * 2, compare)

        dist.put(start, 0)
        pq.insert([0, start])

        while pq.len() > 0:
            let pair = pq.find_min()
            pq.remove_min()
            let d = pair[0]
            let u = pair[1]
            if visited.mem?(u):
                continue
            visited.put(u, True)

        # FIXED: Iterate over the adjacency list directly, don't convert to vector
            let neighbors = self.graph.get_adjacent(u)
            let walker = neighbors
            while walker != None:
                let v = walker.data
                let weight = self.graph.get_edge(u, v)
                let alt = d + weight
                if not dist.mem?(v) or alt < dist.get(v):
                    dist.put(v, alt)
                    pq.insert([alt, v])
                walker = walker.next

        dist


    def locate_all(self, dst_cat):
        if not self.poi_by_cat.mem?(dst_cat):
            return None

        # Collect unique positions into a temporary hash and linked list
        let seen2 = HashTable(100, simple_hash)
        let temp_list = None
        let cur = self.poi_by_cat.get(dst_cat)   # cons-list of [idxp, category, name]
        while cur != None:
            let packed = cur.data                  # [idxp, category, name]
            let idxp = packed[0]
            let pos = self.positions.get(idxp)     # [lat, lon]
            let key = pos_key(pos[0], pos[1])
            if not seen2.mem?(key):
                seen2.put(key, True)
                temp_list = cons(pos, temp_list)
            cur = cur.next

        # Count how many unique positions we collected
        let count = 0
        let walker = temp_list
        while walker != None:
            count = count + 1
            walker = walker.next

        if count == 0:
            return None

        # Copy them into a vector so we can sort by (lat, lon)
        let arr = vec(count)
        let i = 0
        let w2 = temp_list
        while w2 != None:
            arr.put(i, w2.data)     # w2.data is [lat, lon]
            w2 = w2.next
            i = i + 1

        # Sort `arr` by latitude ascending, then longitude ascending
        let j = 0
        while j < count - 1:
            let k = j + 1
            while k < count:
                let p_j = arr.get(j)
                let p_k = arr.get(k)
                if p_j[0] > p_k[0] or (
                    p_j[0] == p_k[0] and p_j[1] > p_k[1]
                ):
                    let tmp = p_j
                    arr.put(j, p_k)
                    arr.put(k, tmp)
                k = k + 1
            j = j + 1

        to_cons(arr)

    def plan_route(self, src_lat, src_lon, dst_name):
        if not self.poi_by_name.mem?(dst_name):
            return None
        let dst_poi = self.poi_by_name.get(dst_name)
        let dst_idx = dst_poi[0]

        let src_key = pos_key(src_lat, src_lon)
        if not self.pos_to_index.mem?(src_key):
            return None
        let start = self.pos_to_index.get(src_key)
        let goal  = dst_idx

        let compare = (lambda a, b: a[0] < b[0])
        let pq = BinHeap(self.graph.n_vertices() * 2, compare)

        let n = self.graph.n_vertices()
        let dist    = vec(n, (lambda i: inf))
        let prev    = vec(n, (lambda i: None))
        let visited = vec(n, (lambda i: False))

        dist.put(start, 0)
        pq.insert([0, start])

        while pq.len() > 0:
            let pair = pq.find_min()
            let d = pair[0]
            let u = pair[1]
            pq.remove_min()
            if visited.get(u): continue
            visited.put(u, True)
            if u == goal: break

            # Iterate adjacency list (a cons‐list)
            let nbrs = self.graph.get_adjacent(u)
            let walker = nbrs
            while walker != None:
                let v = walker.data
                let weight = self.graph.get_edge(u, v)
                let alt = d + weight
                if alt < dist.get(v):
                    dist.put(v, alt)
                    prev.put(v, u)
                    pq.insert([alt, v])
                walker = walker.next

        if prev.get(goal) == None and start != goal:
            return None

        # Reconstruct path into a linked list in forward order
        let path_list = None
        let cur = goal
        while cur != None:
            let pos = self.positions.get(cur)  # [lat, lon]
            path_list = cons(pos, path_list)
            cur = prev.get(cur)

        # path_list is already start→...→goal in head-to-tail order

        # Count elements in path_list
        let count_path = 0
        let ctmp2 = path_list
        while ctmp2 != None:
            count_path = count_path + 1
            ctmp2 = ctmp2.next

        # Allocate a vector and copy
        let arrp = vec(count_path)
        let k = 0
        let c3 = path_list
        while c3 != None:
            arrp.put(k, c3.data)  # c3.data is [lat, lon]
            c3 = c3.next
            k = k + 1

        to_cons(arrp)


    def find_nearby(self, src_lat, src_lon, dst_cat, n):
        # 1) Edge‐cases
        if n == 0:
            return None
        if not self.poi_by_cat.mem?(dst_cat):
            return None
        let key = pos_key(src_lat, src_lon)
        if not self.pos_to_index.mem?(key):
            return None

        # 2) Compute shortest‐path distances
        let start     = self.pos_to_index.get(key)
        let distances = self._dijkstra(start)

        # 3) Count how many POIs in that category
        let total = 0
        let cur   = self.poi_by_cat.get(dst_cat)
        while cur != None:
            total = total + 1
            cur   = cur.next
        if total == 0:
            return None

        # 4) Build a vec of [dist, lat, lon, cat, name]
        let arr = vec(total)
        let i   = 0
        cur     = self.poi_by_cat.get(dst_cat)
        while cur != None:
            let packed = cur.data          # [idx, category, name]
            let idx    = packed[0]
            let pos    = self.positions.get(idx)  # [lat, lon]

            # choose graph distance if reachable, else Euclidean
            let dist = 0
            if distances.mem?(idx):
                dist = distances.get(idx)
            else:
                dist = euclidean_distance(
                    src_lat, src_lon,
                    pos[0], pos[1]
                )

            arr.put(i, [dist, pos[0], pos[1],
                        packed[1], packed[2]])
            i   = i + 1
            cur = cur.next

        # 5) Sort `arr` ascending by (dist, name)
        let j = 0
        while j < total - 1:
            let k = j + 1
            while k < total:
                let a = arr.get(j)
                let b = arr.get(k)
                # swap if (a.dist > b.dist) or (== and a.name > b.name)
                if a[0] > b[0] or (a[0] == b[0] and a[4] > b[4]):
                    arr.put(j, b)
                    arr.put(k, a)
                k = k + 1
            j = j + 1

        # 6) Take the first K = min(n, total) nearest entries
        let K = n
        if total < n:
            K = total

        # 7) Copy those K into a new vec
        let slice = vec(K)
        let t     = 0
        while t < K:
            slice.put(t, arr.get(t))
            t = t + 1

        # 8) Sort `slice` descending by dist, tie‐breaking name ascending
        let p = 0
        while p < K - 1:
            let q = p + 1
            while q < K:
                let x = slice.get(p)
                let y = slice.get(q)
                # swap if (x.dist < y.dist) or (== and x.name > y.name)
                if x[0] < y[0] or (x[0] == y[0] and x[4] > y[4]):
                    slice.put(p, y)
                    slice.put(q, x)
                q = q + 1
            p = p + 1

        # 9) Build final cons-list in slice order
        let out = vec(K)  # temp vec of POI records
        let u   = 0
        while u < K:
            let e = slice.get(u)  # [dist, lat, lon, cat, name]
            out.put(u, [e[1], e[2], e[3], e[4]])
            u = u + 1

        to_cons(out)  # returns cons-list in exactly slice order






def my_first_example():
    return TripPlanner([[0,0, 0,1], [0,0, 1,0]],
                       [[0,0, "bar", "Sketchbook"],
                        [0,1, "food", "Cross Rhodes"]])

test 'My first locate_all test':
    assert my_first_example().locate_all("food") == \
        cons([0,1], None)

test 'My first plan_route test':
   assert my_first_example().plan_route(0, 0, "Cross Rhodes") == \
       cons([0,0], cons([0,1], None))

test 'My first find_nearby test':
    assert my_first_example().find_nearby(0, 0, "food", 1) == \
        cons([0,1, "food", "Cross Rhodes"], None)

        

# Tests for locate_all


test 'locate_all-no-pois':
    let planner1 = TripPlanner([[0,0, 0,1]], [])
    assert planner1.locate_all("food") == None

test 'locate_all-cat-not-present':
    let planner2 = TripPlanner([[0,0, 0,1]],
                                [[0,1, "food", "OnlyFood"]])
    assert planner2.locate_all("bar") == None

test 'locate_all-single-poi':
    let planner3 = TripPlanner([[0,0, 0,1]],
                                [[0,1, "food", "SoloFood"]])
    assert planner3.locate_all("food") == cons([0,1], None)


test 'locate_all-duplicate-position':
    let planner5 = TripPlanner(
        [[0,0, 0,1]],
        [[0,1, "food", "A"],
         [0,1, "food", "B"]]
    )
    assert planner5.locate_all("food") == cons([0,1], None)

test 'locate_all-three-pois-two-duplicate':
    let planner6 = TripPlanner(
        [[0,0, 0,1]],
        [[0,1, "food", "A"],
         [0,1, "food", "B"],
         [1,1, "food", "C"]]
    )
    assert planner6.locate_all("food") == cons([0,1], cons([1,1], None))

test 'locate_all-case-sensitive':
    let planner7 = TripPlanner(
        [[0,0, 0,1]],
        [[0,1, "Food", "Cap1"],
         [1,1, "food", "Low1"]]
    )
    assert planner7.locate_all("food") == cons([1,1], None)
    assert planner7.locate_all("Food") == cons([0,1], None)

test 'locate_all-mixed-categories':
    let planner8 = TripPlanner(
        [[0,0, 0,1], [0,1, 1,1]],
        [[0,1, "food", "F1"],
         [1,1, "bar",  "B1"],
         [2,2, "food", "F2"]]
    )
    assert planner8.locate_all("food") == cons([0,1], cons([2,2], None))
    assert planner8.locate_all("bar") == cons([1,1], None)

test 'locate_all-poi-off-segment':
    let planner9 = TripPlanner(
        [[0,0, 0,1]],
        [[5,5, "food", "FloatingFood"]]
    )
    assert planner9.locate_all("food") == cons([5,5], None)
    
test 'locate_all-all-categories':
    let planner25 = TripPlanner(
        [[0,0, 0,1], [0,1, 1,1]],
        [[0,1, "food", "F1"],
         [1,1, "drink", "D1"],
         [0,0, "bar", "B1"]]
    )
    assert planner25.locate_all("food") == cons([0,1], None)
    assert planner25.locate_all("drink") == cons([1,1], None)
    assert planner25.locate_all("bar") == cons([0,0], None)

test 'locate_all-long-chain':
    let planner26 = TripPlanner(
        [[0,0, 0,1], [0,1, 0,2], [0,2, 0,3]],
        [[0,3, "cafe", "FarCafe"]]
    )
    assert planner26.locate_all("cafe") == cons([0,3], None)



# Tests for plan_route


test 'plan_route-no-pois':
    let planner10 = TripPlanner([[0,0, 0,1]], [])
    assert planner10.plan_route(0, 0, "AnyName") == None

test 'plan_route-dst-not-present':
    let planner11 = TripPlanner(
        [[0,0, 0,1]],
        [[0,1, "food", "Exists"]]
    )
    assert planner11.plan_route(0, 0, "NonExistent") == None

test 'plan_route-src-not-present':
    let planner12 = TripPlanner(
        [[0,0, 0,1]],
        [[0,1, "food", "Exists"]]
    )
    assert planner12.plan_route(5, 5, "Exists") == None

test 'plan_route-src-equals-dst':
    let planner13 = TripPlanner(
        [[0,0, 0,1]],
        [[0,0, "food", "HomeFood"]]
    )
    assert planner13.plan_route(0, 0, "HomeFood") == cons([0,0], None)

test 'plan_route-one-edge':
    let planner14 = TripPlanner(
        [[0,0, 0,1]],
        [[0,1, "food", "TargetFood"]]
    )
    assert planner14.plan_route(0, 0, "TargetFood") == cons([0,0], cons([0,1], None))

test 'plan_route-two-edge':
    let planner15 = TripPlanner(
        [[0,0, 0,1], [0,1, 1,1]],
        [[1,1, "bar", "Bar1"]]
    )
    assert planner15.plan_route(0, 0, "Bar1") == cons([0,0], cons([0,1], cons([1,1], None)))

test 'plan_route-unreachable':
    let planner16 = TripPlanner(
        [[0,0, 0,1]],
        [[5,5, "food", "F1"]]
    )
    assert planner16.plan_route(0, 0, "F1") == None

test 'plan_route-duplicate-names':
    let planner17 = TripPlanner(
        [[0,0, 1,0], [1,0, 1,1]],
        [[1,1, "cafe", "DupName"],
         [0,0, "cafe", "DupName"]]
    )
    assert planner17.plan_route(0, 0, "DupName") == cons([0,0], None)


test 'plan_route-long-path':
    let planner28 = TripPlanner(
        [[0,0, 0,1], [0,1, 1,1], [1,1, 1,2], [1,2, 2,2]],
        [[2,2, "food", "FarFood"]]
    )
    assert planner28.plan_route(0, 0, "FarFood") == cons([0,0], cons([0,1], cons([1,1], cons([1,2], cons([2,2], None)))))

# Tests for find_nearby


test 'find_nearby-no-pois':
    let planner18 = TripPlanner([[0,0, 0,1]], [])
    assert planner18.find_nearby(0, 0, "food", 3) == None

test 'find_nearby-cat-not-present':
    let planner19 = TripPlanner(
        [[0,0, 0,1]],
        [[0,1, "food", "Eatery"]]
    )
    assert planner19.find_nearby(0, 0, "bar", 5) == None

test 'find_nearby-n-zero':
    let planner20 = TripPlanner(
        [[0,0, 0,1]],
        [[0,1, "food", "Eatery"]]
    )
    assert planner20.find_nearby(0, 0, "food", 0) == None

test 'find_nearby-single-poi':
    let planner21 = TripPlanner(
        [[0,0, 0,1]],
        [[0,1, "food", "SoloFood"]]
    )
    assert planner21.find_nearby(0, 0, "food", 1) == cons([0,1, "food", "SoloFood"], None)



test 'find_nearby-tie-distances':
    let planner23 = TripPlanner(
        [[0,0, 1,0], [1,0, 1,1]],
        [[1,1, "cafe", "C1"],
         [0,1, "cafe", "C2"],
         [1,0, "cafe", "C3"]]
    )
    let result23 = planner23.find_nearby(0, 0, "cafe", 2)
    assert ( (result23 == cons([0,1, "cafe", "C2"], cons([1,0, "cafe", "C3"], None))) or
             (result23 == cons([1,0, "cafe", "C3"], cons([0,1, "cafe", "C2"], None))) )

test 'find_nearby-poi-off-graph':
    let planner24 = TripPlanner(
        [[0,0, 0,1]],
        [[5,5, "food", "DistantFood"]]
    )
    assert planner24.find_nearby(0, 0, "food", 1) == cons([5,5, "food", "DistantFood"], None)
test 'find_nearby-unreachable-poi':
    let tp = TripPlanner(
        [[0, 0, 1.5, 0], [1.5, 0, 2.5, 0], [2.5, 0, 3, 0], [4, 0, 5, 0]],
        [[1.5, 0, 'bank', 'Union'], [3, 0, 'barber', 'Tony'],
         [4, 0, 'food', 'Jollibee'], [5, 0, 'barber', 'Judy']])
    let result = tp.find_nearby(0, 0, 'food', 1)
    assert Cons.to_vec(result) == [[4, 0, 'food', 'Jollibee']]
test 'find_nearby-mst-not-sssp':
    let tp = TripPlanner(
        [[-1.1, -1.1, 0, 0], [0, 0, 3, 0], [3, 0, 3, 3], [3, 3, 3, 4], [0, 0, 3, 4]],
        [[0, 0, 'food', 'Sandwiches'], [3, 0, 'bank', 'Union'],
         [3, 3, 'barber', 'Judy'], [3, 4, 'barber', 'Tony']])
    let result = tp.find_nearby(-1.1, -1.1, 'barber', 1)
    assert Cons.to_vec(result) == [[3, 4, 'barber', 'Tony']]

test 'find_nearby-two-pois-limit-3':
    let tp = TripPlanner(
        [[0, 0, 1.5, 0], [1.5, 0, 2.5, 0], [2.5, 0, 3, 0],
         [4, 0, 5, 0], [3, 0, 4, 0]],
        [[1.5, 0, 'bank', 'Union'], [3, 0, 'barber', 'Tony'],
         [4, 0, 'food', 'Jollibee'], [5, 0, 'barber', 'Judy']])
    let result = tp.find_nearby(0, 0, 'barber', 3)
    assert Cons.to_vec(result) == [[5, 0, 'barber', 'Judy'], [3, 0, 'barber', 'Tony']]
test 'find_nearby-poi-second-of-three':
    let tp = TripPlanner(
        [[0, 0, 1.5, 0], [1.5, 0, 2.5, 0], [2.5, 0, 3, 0],
         [4, 0, 5, 0], [3, 0, 4, 0]],
        [[1.5, 0, 'bank', 'Union'], [3, 0, 'barber', 'Tony'],
         [5, 0, 'food', 'Jollibee'], [5, 0, 'barber', 'Judy'],
         [5, 0, 'bar', 'Pasta']])
    let result = tp.find_nearby(0, 0, 'barber', 2)
    assert Cons.to_vec(result) == [[5, 0, 'barber', 'Judy'], [3, 0, 'barber', 'Tony']]

test 'find_nearby-two-pois-same-location':
    let tp = TripPlanner(
        [[-1, -1, 0, 0], [0, 0, 3.5, 0], [0, 0, 0, 3.5], [3.5, 0, 0, 3.5]],
        [[-1, -1, 'food', 'Jollibee'], [0, 0, 'bank', 'Union'],
         [0, 3.5, 'barber', 'Tony'], [0, 3.5, 'barber', 'Judy']])
    let result = tp.find_nearby(0, 0, 'barber', 2)
    assert Cons.to_vec(result) == [[0, 3.5, 'barber', 'Judy'], [0, 3.5, 'barber', 'Tony']]



        

