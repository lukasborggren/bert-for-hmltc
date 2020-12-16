import os
import networkx as nx
import matplotlib.pyplot as plt

inf = "data/hierarchy.txt"
out = os.path.splitext(inf)[0] + ".png"

g = nx.Graph()

for line in open(inf).readlines():
    adj = list(map(str, line.split("\t")))
    u = adj[0]
    for v in adj[1:]:
        g.add_edge(u, v)

# nx.drawing.nx_pydot.write_dot(g, out_dot)
nx.draw(g, with_labels=True, font_size=7)
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()

# (graph,) = pydot.graph_from_dot_file(out_dot)
# g.draw(out_png)
