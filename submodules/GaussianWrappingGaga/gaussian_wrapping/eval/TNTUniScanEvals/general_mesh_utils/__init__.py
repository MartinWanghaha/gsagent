from general_mesh_utils.mesh_utils import ScalableMeshRenderer, MeshRasterizer, Meshes
from general_mesh_utils.geometry_utils import (
    depths_to_points as depth_to_view_points, 
    is_in_view_frustum,
    transform_points_view_to_world
)
from general_mesh_utils.cameras import Camera, cameraList_from_camInfos
from general_mesh_utils.camera_info_dataset import sceneLoadTypeCallbacks

def _frustum_cull_mesh(
    mesh:Meshes,
    viewpoint_cam:Camera,
) -> Meshes:
    """Cull the mesh based on the view frustum of the given viewpoint.

    Args:
        mesh (Meshes): The mesh to cull.
        viewpoint_cam (Camera): The viewpoint camera.

    Returns:
        Meshes: The culled mesh.
    """
    faces_mask = is_in_view_frustum(mesh.verts, viewpoint_cam)[mesh.faces].any(axis=1)
    return Meshes(
        verts=mesh.verts, 
        faces=mesh.faces[faces_mask], 
        verts_colors=mesh.verts_colors
    )

def frustum_cull_mesh(mesh: Meshes, camera: Camera) -> Meshes:
    return _frustum_cull_mesh(mesh, camera)

from general_mesh_utils.arguments import ModelParams