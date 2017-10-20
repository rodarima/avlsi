import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Problem description
NON_FIXED = list('abcd')
FIXED = list('uvw')
FIXED_POS = {'u':(0,4), 'v':(4,3), 'w':(2,0)}
NON_FIXED_EDGES = 'ab ac bd'
FIXED_EDGES = 'ub vc vd wa'

# Output graph
OUT_FILE = 'graph.pdf'

#----------------------------------------------------------------------

# Compute the sets from the compact representation:
# 'ab cd' -> [('a', 'b'), ('c', 'd')]
edges_n = [tuple(list(e)) for e in NON_FIXED_EDGES.split()]
edges_f = [tuple(list(e)) for e in FIXED_EDGES.split()]

# Build the graph
g = nx.Graph()
g.add_nodes_from(NON_FIXED)
for v in FIXED:
	g.add_node(v, pos=FIXED_POS[v])
g.add_edges_from(edges_n)
g.add_edges_from(edges_f)

# Eq. system of NxN
N = len(NON_FIXED)

# Compute bx and by
bx = np.zeros(N)
by = np.zeros(N)
for i in range(len(NON_FIXED)):
	v = NON_FIXED[i]
	for w in g.neighbors(v):
		if w in FIXED_POS:
			bx[i] += FIXED_POS[w][0]
			by[i] += FIXED_POS[w][1]

# Compute the Laplacian matrix
A = nx.laplacian_matrix(g).todense()
# And remove the fixed points
RN = range(N)
An = A[RN,:][:,RN]

# Solve the system to get the positions
X = np.linalg.solve(An, bx)
Y = np.linalg.solve(An, by)

print('X = {}'.format(X))
print('Y = {}'.format(Y))

# Add non-fixed computed node positions
pos = FIXED_POS.copy()
for i in range(len(NON_FIXED)):
	v = NON_FIXED[i]
	pos[v] = (X[i], Y[i])

# Mark fixed nodes with black color
nodelist = NON_FIXED + FIXED
node_color = ['darkgrey'] * len(NON_FIXED) + ['black'] * len(FIXED)

plt.figure(figsize=(4, 4))

# Plot the graph with the node positions and corresponding colors
nx.draw_networkx(g, pos, edge_color='black', node_color=node_color,
	font_color='white', width=2, with_labels = True)

plt.grid()
plt.tight_layout()
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.axis('equal')
ax = plt.gca()
ax.set_axisbelow(True)
#plt.show()

# Save in pdf
plt.savefig(OUT_FILE)

