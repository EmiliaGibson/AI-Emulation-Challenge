! pip install torch_geometric

import numpy as np
import torch
from torch_geometric.data import HeteroData
import pandas as pd
from pathlib import Path
from collections import defaultdict



def create_icosahedron():
    '''
    define (coords of) vertices and faces of regular icosahedron
    '''
    
    phi = (1.0 + np.sqrt(5.0)) / 2.0 #golden ration

    vertices = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],

        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],

        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1],
    ], dtype=float)

    # Project to unit sphere
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

    # vertices on each face  (vertices numbered from from 0 to 11)
    faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],

        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],

        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],

        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ], dtype=int)

    return vertices, faces


def subdivide(vertices, faces):
    '''
    One subdivision step of icosahedron mesh refinement
    '''
    new_vertices =  np.zeros((len(vertices)+int(len(faces)*3/2),3))
    new_vertices[:len(vertices)] = vertices
    edge_to_midpoint = {}

    def get_midpoint(i, j, count):

        # Make edge key independent of orientation
        key = tuple(sorted((i, j)))
        if key not in edge_to_midpoint:
            midpoint = (
                vertices[i] +
                vertices[j]
            )

            # Project midpoint back onto unit sphere
            midpoint /= np.linalg.norm(midpoint)

            edge_to_midpoint[key] = len(vertices) + count #new vertex number
            new_vertices[len(vertices) + count] = midpoint
            count+=1

        return edge_to_midpoint[key], count

    new_faces =  np.zeros((len(faces)*4,3),dtype=int)
    
    
    
    count1 = 0
    count2 = 0

    for a, b, c in faces:
        ab, count2 = get_midpoint(a, b, count2)
        bc, count2 = get_midpoint(b, c, count2)
        ca, count2 = get_midpoint(c, a, count2)
        new_faces[4*count1:4*count1+4] = np.array([
            [a,  ab, ca],
            [b,  bc, ab],
            [c,  ca, bc],
            [ab, bc, ca],
        ])
            
        count1+=1


    return new_vertices, new_faces

def create_icosphere(refinement=3):
    '''
    Construct refined icosphere of order refinement
    '''
  
    vertices, faces = create_icosahedron()
    
    for _ in range(refinement):
        vertices, faces = subdivide(vertices, faces)
    return vertices, faces


def spherical_coordinates(vertices):
    '''
    Convert Euclidean coords to Lat\Lon coords
    '''
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    longitude = np.arctan2(y, x)
    latitude = np.arcsin(z)

    return latitude, longitude

def create_mesh_node_features(vertices):

    latitude, longitude = spherical_coordinates(vertices)

    features = np.stack([
        np.cos(latitude),
        np.sin(longitude),
        np.cos(longitude),
    ], axis=1)

    return features



def receiver_rotation(latitude, longitude):
    """
    Receiver-dependent local rotation: 
    - maps the receiver's position to (1, 0, 0).
    - Return R_y(latitude) R_z(-longitude).
    """

    cl = np.cos(-longitude)
    sl = np.sin(-longitude)

    cp = np.cos(latitude)
    sp = np.sin(latitude)

    Rz = np.array([
        [ cl, -sl, 0.0],
        [ sl,  cl, 0.0],
        [0.0, 0.0,1.0],
    ])

    Ry = np.array([
        [ cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ])

    return  Ry @ Rz

def create_mesh_edges(vertices, faces):

    latitude, longitude = spherical_coordinates(vertices)

    # Get unique undirected edges
    undirected_edges = set()

    for a, b, c in faces:
        undirected_edges.add(tuple(sorted((a, b))))
        undirected_edges.add(tuple(sorted((b, c))))
        undirected_edges.add(tuple(sorted((c, a))))

    edge_list = []
    edge_attributes = []

    for a, b in undirected_edges:

        # Add both directions
        for sender, receiver in [(a, b), (b, a)]:
            x_sender = vertices[sender]
            x_receiver = vertices[receiver]

            # Receiver coordinate system
            R = receiver_rotation(
                latitude[receiver],
                longitude[receiver]
            )

            # Vector difference in receiver coordinates
            delta = R @ (x_sender - x_receiver)

            # Euclidean edge length
            length = np.linalg.norm(x_sender - x_receiver)

            edge_list.append([sender, receiver])

            edge_attributes.append([
                length,
                delta[0],
                delta[1],
                delta[2],
            ])

    edge_index = np.asarray(edge_list).T
    edge_attr = np.asarray(edge_attributes)

    return edge_index, edge_attr


def create_mesh_graph(refinement=3):
    '''
    Build PyTorch Geometric graph
    '''
    vertices, faces = create_icosphere(refinement)

    node_features = create_mesh_node_features(vertices)

    edge_index, edge_attr = create_mesh_edges(
        vertices,
        faces
    )

    data = HeteroData(mesh_nodes={
        'x' : torch.tensor(
            node_features,
            dtype=torch.float32
        ),
        'pos' : torch.tensor(
            vertices,
            dtype=torch.float32
        )},
        mesh_nodes__mesh_edges__mesh_nodes={
        'edge_index': torch.tensor(
                        edge_index,
                        dtype=torch.long
        ),
        'edge_attr' : torch.tensor(
            edge_attr,
            dtype=torch.float32
        )}
        ,
    )

    return data, faces


def site_index_to_lat_lon(idx):
    """Converts site_index to (lat, lon)."""
    lat = np.zeros(len(idx))
    lon = np.zeros(len(idx))
    
    LAT_ORIGIN = 90.0  # Top edge (North)
    LON_ORIGIN = -180.0  # Left edge (West)
    SPACING = 0.5  # 0.5-degree spacing
    
    flat_pos = np.flatnonzero(mask_bool.ravel()) 
    
    for n in range(len(idx)):
        pos = flat_pos[idx[n]]
        i,j = np.unravel_index(pos, mask_bool.shape)
    
        # Subtract for latitude because moving down rows goes South
        lat[n] = LAT_ORIGIN - (i * SPACING)
        lon[n] = LON_ORIGIN + (j * SPACING)



    return lat, lon


def create_grid_node_features(site_idx):
    ''' 
    Place holder for grid node features:
    grid node features defined based off longitude and latitude
    '''
    grid_lat, grid_lon = site_index_to_lat_lon(site_idx)
    
    #to be changed
    grid_features = np.column_stack([ 
        np.cos(grid_lat),
        np.sin(grid_lon),
        np.cos(grid_lon)
    ])
    return grid_features

def create_grid_to_mesh_edges(mesh_graph, site_idx,refinement):
    # max_distance between grid_nodes and mesh_vertices for thresholding
    thresholds = [0.6,0.6,0.6,0.6,0.68]
    threshold = thresholds[refinement]
    max_length = threshold*max(np.sort(mesh_graph['mesh_nodes', 'mesh_edges', 'mesh_nodes'].edge_attr[:,0])[:3840])  #refinement 3 has 3840 edges

    grid_features = create_grid_node_features(site_idx)

    mesh_vertices = mesh_graph['mesh_nodes'].pos.numpy()
    #convert training sites to lat-lon
    grid_lat, grid_lon = site_index_to_lat_lon(site_idx)

    # Grid points -> Cartesian coordinates on unit sphere
    grid_xyz = np.column_stack([
        np.cos(grid_lat) * np.cos(grid_lon),
        np.cos(grid_lat) * np.sin(grid_lon),
        np.sin(grid_lat)
    ])

    # Find grid-mesh pairs within threshold
    dot = grid_xyz @ mesh_vertices.T
    dot = np.clip(dot, -1.0, 1.0)
    #angular_distance
    angular_distance = np.arccos(dot)
    distance = 2*np.sin(angular_distance/2)
    grid_idx, mesh_idx = np.where(distance < max_length)


    grid_features = torch.tensor(grid_features).to(torch.float32)
    
    mesh_idx_tensor = torch.tensor(
        mesh_idx,
        dtype=torch.long
    )
    
    grid_idx_tensor = torch.tensor(
        grid_idx, #+mesh_graph.num_nodes 
        dtype=torch.long
    )

    grid_to_mesh_edges = torch.column_stack((grid_idx_tensor,mesh_idx_tensor))
    
    G2M_attributes = []
    latitude, longitude = spherical_coordinates(mesh_vertices)
    for edge in grid_to_mesh_edges:
        x_sender = grid_xyz[edge[0]]
        x_receiver = mesh_vertices[edge[1]]
        
        # Receiver coordinate system
        
        R = receiver_rotation(
                    latitude[edge[1]],
                    longitude[edge[1]]
                )
        # Vector difference in receiver coordinates
        delta = R @ (x_sender - x_receiver)
        # Euclidean edge length
        length = np.linalg.norm(x_sender - x_receiver)
    
        G2M_attributes.append([
                    length,
                    delta[0],
                    delta[1],
                    delta[2],
                ])
    G2M_attributes = np.asarray(G2M_attributes)

    # max_length = np.max(G2M_attributes[:,0])
    # G2M_attributes = G2M_attributes/max_length
    grid_to_mesh_edges = grid_to_mesh_edges.numpy().T

    return grid_to_mesh_edges, G2M_attributes, grid_features, grid_xyz


def create_grid_to_mesh_graph(mesh_graph, site_idx,refinement):
    grid_to_mesh_edges, G2M_attributes, grid_node_features, grid_positions = create_grid_to_mesh_edges(mesh_graph, 
                                                                           site_idx,refinement)
    data = mesh_graph    
    data['grid_nodes'].x = grid_node_features
    data['grid_nodes'].pos =  torch.tensor(grid_positions).to(torch.float32)
    
    data['grid_nodes','G2M_edges','mesh_nodes'].edge_index =  torch.tensor(grid_to_mesh_edges).to(torch.long)
    data['grid_nodes','G2M_edges','mesh_nodes'].edge_attr = torch.tensor(G2M_attributes).to(torch.float32)

    return data

#### CREATE mesh-to-grid edges
def point_in_spherical_triangle(point, a, b, c, tol=1e-12):
    s1 = np.dot(np.cross(a, b), point)
    s2 = np.dot(np.cross(b, c), point)
    s3 = np.dot(np.cross(c, a), point)

    return (
        (s1 >= -tol and s2 >= -tol and s3 >= -tol)
        or
        (s1 <= tol and s2 <= tol and s3 <= tol)
    )


def find_containing_face(point, grid2mesh_graph, faces, candidate_faces):

    for face_idx in candidate_faces:

        a_idx, b_idx, c_idx = faces[face_idx]

        if point_in_spherical_triangle(
            point,
            grid2mesh_graph['mesh_nodes'].pos[a_idx].numpy(),
            grid2mesh_graph['mesh_nodes'].pos[b_idx].numpy(),
            grid2mesh_graph['mesh_nodes'].pos[c_idx].numpy()
        ):
            return face_idx

    return None





def get_M2G_edgelist(faces,grid2mesh_graph, refinement):
    vertex_to_faces = defaultdict(list)

    for face_idx, face in enumerate(faces):
        for vertex_idx in face:
            vertex_to_faces[vertex_idx].append(face_idx) 
            
    grid_idx = np.repeat(np.arange(grid2mesh_graph['grid_nodes'].num_nodes), 3)

    mesh_idx = np.zeros(len(grid_idx))
    for grid_node in range(grid2mesh_graph['grid_nodes'].num_nodes):
        candidate_mesh_vertices = grid2mesh_graph['grid_nodes', 'G2M_edges', 'mesh_nodes'].edge_index[1][grid2mesh_graph['grid_nodes', 'G2M_edges', 'mesh_nodes'].edge_index[0]==grid_node].numpy()
        candidate_faces = []
        for vertex in candidate_mesh_vertices:
            candidate_faces+=vertex_to_faces[vertex]
        face_idx = find_containing_face(grid2mesh_graph['grid_nodes'].pos[grid_node].numpy(), grid2mesh_graph, faces, candidate_faces)
        mesh_idx[3*grid_node:3*grid_node+3] = faces[face_idx]
    M2G_edges = np.column_stack((np.asarray(mesh_idx),grid_idx))
    return M2G_edges


def create_M2G_attr(faces,grid2mesh_graph,site_idx,refinement):
    M2G_edges = get_M2G_edgelist(faces,grid2mesh_graph, refinement)
    M2G_edges_tensor = torch.tensor(
        M2G_edges,
        dtype=torch.long
    )

    grid_lat, grid_lon = site_index_to_lat_lon(site_idx)

    M2G_attr = []
    for edge in M2G_edges_tensor:
        x_sender = grid2mesh_graph['mesh_nodes'].pos[edge[0]].numpy()
        x_receiver = grid2mesh_graph['grid_nodes'].pos[edge[1]].numpy()
        
        # Receiver coordinate system
        R = receiver_rotation(
                    grid_lat[edge[1]],
                    grid_lon[edge[1]]
                )
        # Vector difference in receiver coordinates
        delta = R @ (x_sender - x_receiver)
        # Euclidean edge length
        length = np.linalg.norm(x_sender - x_receiver)
    
        M2G_attr.append([
                    length,
                    delta[0],
                    delta[1],
                    delta[2],
                ])
    M2G_attr = np.asarray(M2G_attr)

    # max_length = np.max(M2G_attr[:,0])
    # M2G_attr = M2G_attr/max_length

    return M2G_edges.T, M2G_attr


def create_mesh_to_grid_graph(faces,grid2mesh_graph,site_idx, refinement):
    
    M2G_edges, M2G_attributes = create_M2G_attr(faces,grid2mesh_graph,site_idx, refinement)
    data = grid2mesh_graph

    data['mesh_nodes','M2G_edges','grid_nodes'].edge_index =  torch.tensor(M2G_edges).to(torch.long)
    data['mesh_nodes','M2G_edges','grid_nodes'].edge_attr = torch.tensor(M2G_attributes).to(torch.float32)


    return data




#--------------------------------------------
#        Build graph
# ---------------------------------------------
#get training sites
DATA_DIR = Path('/kaggle/input/datasets/zhihaow/carbonglobe/')
STAT_DIR = DATA_DIR / 'data_stats'

#landsites mask
mask2d = np.load(STAT_DIR / 'mask2d.npy')                # (360, 720)
mask_bool = mask2d.astype(bool)


train_idx = pd.read_csv(DATA_DIR / 'train.csv', header=None)[0].values
test_idx = pd.read_csv(DATA_DIR / 'test.csv', header=None)[0].values

COMP_DIR = Path('/kaggle/input/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge/public')
competition_test_idx  = pd.read_csv(COMP_DIR / 'sites_ssp.csv', header=None,skiprows=1)[0].values



mesh_graph, mesh_faces = create_mesh_graph(refinement=3)
data = create_grid_to_mesh_graph(mesh_graph, test_idx[competition_test_idx],refinement=3)

full_graph = create_mesh_to_grid_graph(mesh_faces,data,test_idx[competition_test_idx],refinement=3)
full_graph

torch.save(full_graph,'full_graph.pt')
