
from classes import globe
def update_edge(curr:str):
    globe.edges.append((globe.prev, curr))
    globe.prev = curr

