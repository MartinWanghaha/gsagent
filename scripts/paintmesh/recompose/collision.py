"""Accelerated exact triangle checks without changing mesh geometry."""
import hashlib
import numpy as np

_closed_cache={}


def is_watertight(vertices,faces):
    """Closed manifold plus MeshLab's indexed triangle self-intersection test.

    Open3D 0.18 IsWatertight scans every triangle pair when testing self
    intersections; generated objects can contain hundreds of thousands of faces.
    """
    import open3d as o3d
    vertices=np.ascontiguousarray(vertices,dtype=np.float64)
    faces=np.ascontiguousarray(faces,dtype=np.int32)
    key=hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest()
    if key in _closed_cache:return _closed_cache[key]
    m=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
    closed=m.is_edge_manifold(False) and m.is_vertex_manifold()
    if closed:
        import pymeshlab
        ms=pymeshlab.MeshSet();ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices,face_matrix=faces))
        ms.compute_selection_by_self_intersections_per_face()
        closed=ms.current_mesh().selected_face_number()==0
    _closed_cache[key]=bool(closed)
    return bool(closed)


def meshes_intersect(vertices_a,faces_a,vertices_b,faces_b):
    """FCL BVH triangle collision. Containment must be checked separately."""
    import fcl
    def model(vertices,faces):
        m=fcl.BVHModel();m.beginModel()
        m.addSubModel(np.ascontiguousarray(vertices,dtype=np.float64),np.ascontiguousarray(faces,dtype=np.int32))
        m.endModel();return fcl.CollisionObject(m)
    a=model(vertices_a,faces_a);b=model(vertices_b,faces_b)
    result=fcl.CollisionResult()
    fcl.collide(a,b,fcl.CollisionRequest(num_max_contacts=1,enable_contact=False),result)
    return bool(result.is_collision)
