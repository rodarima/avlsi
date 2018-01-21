import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import scipy.sparse

STD_SCALE = True

def laplacian_layout(G, norm=False, dim=2, bad=False):
	A = nx.to_scipy_sparse_matrix(G, format='csr')
	A = np.array(A.todense())
	n, m = A.shape

	diags = A.sum(axis=1).flatten()
	D = np.diag(diags)
	L = D - A
	B = np.eye(n)
	if norm == False:
		layout = eig_layout(G, L, B)
	else:
		if bad:
			with scipy.errstate(divide='ignore'):
				#diags_sqrt = 1.0 / scipy.power(diags, 1)
				diags_sqrt = 1.0 / scipy.sqrt(diags)
			diags_sqrt[scipy.isinf(diags_sqrt)] = 0
			DH = np.diag(diags_sqrt)
			L = np.dot(DH, np.dot(L, DH))
			eigenvalues, eigenvectors = scipy.linalg.eigh(L)
			index = np.argsort(eigenvalues)[1:dim + 1]
			pos = np.real(eigenvectors[:, index])

			pos = np.dot(DH, pos)
			pos = dict(zip(G, pos))
			layout = pos
		else:
			B = D
			layout = eig_layout(G, L, B)

		
	#print(L)
	#print(B)

	return layout

def eig_layout(G, L, B, dim=2):

	#eigenvalues, eigenvectors = np.linalg.eigh(L)
	eigenvalues, eigenvectors = scipy.linalg.eigh(L, B)
	index = np.argsort(eigenvalues)[1:dim + 1]
	pos = np.real(eigenvectors[:, index])
	# Scale to std = 1
	#pos /= np.std(pos, axis=0)
	#print(pos)
	#pos = rescale_layout(pos, scale) + center
	pos = dict(zip(G, pos))
	return pos

def show_layout(G, pos):
	nx.draw_networkx(G, pos=pos)
	plt.xlim(-1,1)
	plt.ylim(-1,1)
	plt.show()

def save_graph(G, fn, title=None, pos=None):

	with_labels = False
	#if G.number_of_nodes() > 30: with_labels = None
	#labels = None
	labels = dict(G.degree)
	node_size=20
	el = []
	node_color = np.array([G.degree[v] for v in G.nodes()])
	color_map = plt.cm.jet

	g = nx.draw_networkx_nodes(G, pos=pos, node_size=node_size, alpha=1,
		with_labels=with_labels, cmap=color_map, node_color=node_color)

	nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.1)

	#plt.xlim(-2,2)
	#plt.ylim(-2,2)
	cbar = plt.colorbar(g)
	cbar.ax.set_ylabel('Node degree')
	plt.title(title)
	plt.tight_layout()
	plt.savefig(fn)
	plt.close()

def wire_length(G, pos):
	length = 0
	for u, v in G.edges:
		length += np.linalg.norm(pos[u] - pos[v])

	return length

def scale_layout(G, pos):
	A = np.array([list(e) for e in pos.values()])
	std = np.std(A, axis=0)
	var = np.var(A, axis=0)
	for u in G:
		pos[u] /= std

def scale_unit(G, pos):
	# Test the wirelength using -1, 1 bounds
	A = np.array([list(e) for e in pos.values()])
	vmax = np.min(A, axis=0)
	vmin = np.max(A, axis=0)
	for u in G:
		pos[u] = (pos[u] - vmin) / (vmax - vmin)

def process_graph(G, name):

	figdir = 'fig/{}/'.format(name)
	laplacian = laplacian_layout(G)
	norm = laplacian_layout(G, norm=True)
	spring = nx.layout.spring_layout(G)

	if STD_SCALE:
		scale_layout(G, laplacian)
		scale_layout(G, norm)
		scale_layout(G, spring)
	else:
		scale_unit(G, laplacian)
		scale_unit(G, norm)
		scale_unit(G, spring)

	print("laplacian wire-length                : {:.2f}".format(wire_length(G, laplacian)))
	print("normalized laplacian wire-length     : {:.2f}".format(wire_length(G, norm)))
	print("spring wire-length                   : {:.2f}".format(wire_length(G, spring)))

	wire_laplacian = wire_length(G, laplacian)
	wire_norm = wire_length(G, norm)
	wire_spring = wire_length(G, spring)

	laplacian_title = 'Graph {}, method 1 laplacian, wirelength {:.2f}'.format(
		name, wire_laplacian)
	norm_title = 'Graph {}, method 2 normalized, wirelength {:.2f}'.format(
		name, wire_norm)
	spring_title = 'Graph {}, method 3 force-directed, wirelength {:.2f}'.format(
		name, wire_spring)

	save_graph(G, figdir + 'laplacian.pdf', title=laplacian_title, pos=laplacian)
	save_graph(G, figdir + 'norm.pdf', title=norm_title, pos=norm)
	save_graph(G, figdir + 'spring.pdf', title=spring_title, pos=spring)


def main():
	graphs = {}
	np.random.seed(1)

	# Generate some graphs

	G = nx.Graph()
	G.add_edges_from([(1,2),(2,3),(2,4),(3,5),(3,6),(4,5),(5,6),(5,7)])
	graphs['A'] = G
	#G = nx.Graph()
	#G.add_edges_from([(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,3),(3,4),(4,5),(5,6),(6,7),(7,2)])
	#graphs['new'] = G
	graphs['B'] = nx.erdos_renyi_graph(100, 0.1, seed=2)
	#graphs['BIG'] = nx.erdos_renyi_graph(1000, 0.1, seed=2)
	graphs['C'] = nx.read_edgelist('uart.edges')
		
	# More graphs
	#G = nx.complete_graph(N)
	#G = nx.star_graph(N)
	#G.add_edges_from([(1,2),(2,3),(3,1),(3,4)])
	#G.add_edges_from([(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,3),(3,4),(4,5),(5,6),(6,7),(7,2)])

	for name, G in graphs.items():
		n = G.number_of_nodes()
		print('Graph ' + name + ' with {} nodes'.format(n))
		process_graph(G, name)

main()
