from typing import Optional, Tuple, Optional, Callable, List, Union
import torch
import numpy as np
from scene.cameras import Camera
from scene.mesh import Meshes
from utils.geometry_utils import (
    depths_to_points,
    transform_points_view_to_world,
)
from utils.tetmesh import marching_tetrahedra
from utils.tetmesh_gggs import marching_tetrahedra as marching_tetrahedra_with_valid # We borrow this MT function with valids input from GGGS
# LINK: https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting
from utils.sh_utils import eval_sh
from tqdm import tqdm


@torch.no_grad()
def compute_isosurface_value_from_depth(
    cameras: List[Camera],
    depth: Union[torch.Tensor, List[torch.Tensor]],
    sdf_function: Callable,
    n_depth_points: int = 1_000_000,
    reduction: str = "median",
) -> float:
    """
    Compute the best isosurface value from depth points.

    Args:
        cameras (List[Camera]): List of V cameras.
        depth (Union[torch.Tensor, List[torch.Tensor]]): Depth map. Shape: (V, 1, H, W) or List of V tensors with shape (1, H, W).
        sdf_function (Callable): SDF function. Takes a tensor of P points and returns the SDF values with shape (P,).
        n_depth_points (int, optional): Number of depth points to sample. Defaults to 1_000_000.
        reduction (str, optional): Reduction method to compute the best isosurface value. Defaults to "median".
            Can be "mean", "median".

    Returns:
        float: The best isosurface value.
    """
    assert reduction in ["mean", "median"]

    depth_points = []
    n_depth_points_per_cam = n_depth_points // len(cameras)
    
    for i_cam in tqdm(range(len(cameras)), desc="Sampling depth points for computing isosurface value"):
        # Convert depth to points in view space
        points_i = depths_to_points(view=cameras[i_cam], depthmap1=depth[i_cam].cuda())  # (3, H, W)
        points_i = points_i.permute(1, 2, 0).view(-1, 3)  # (H*W, 3)
        
        # Sample random points
        points_i = points_i[torch.randperm(points_i.shape[0])[:n_depth_points_per_cam]]  # (n_depth_points_per_cam, 3)
        
        # Convert points to world space
        points_i = transform_points_view_to_world(
            points=points_i.unsqueeze(0),
            cameras=[cameras[i_cam]]
        ).squeeze(0)  # (n_depth_points_per_cam, 3)
        
        depth_points.append(points_i)
        
    depth_points = torch.cat(depth_points, dim=0)  # (n_depth_points, 3)
    
    # Evaluate SDF at depth points
    depth_points_sdf = sdf_function(depth_points)  # (n_depth_points,)
    
    if reduction == "mean":
        isosurface_value = depth_points_sdf.mean().item()
    elif reduction == "median":
        isosurface_value = depth_points_sdf.median().item()
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")
    
    return isosurface_value
    

def semantic_edge_mask(
    pivots_labels: torch.Tensor,
    pivots_label_confidence: torch.Tensor,
    verts_idx: torch.Tensor,
    confidence_threshold: float = 0.5,
    background_label: int = 0,
) -> torch.Tensor:
    """Mark MT edges that confidently bridge two different instances.

    An edge is semantically bad when both endpoint pivots carry confident,
    different, non-background instance labels. Edges touching the background
    or an unlabeled pivot are never filtered: the object-background contact
    is a real surface, and low-confidence labels must not delete geometry.

    Args:
        pivots_labels: (N_pivots,) long instance labels, -1 = unlabeled.
        pivots_label_confidence: (N_pivots,) in [0, 1].
        verts_idx: (N_verts, 2) pivot indices of each crossing MT edge.
        confidence_threshold: minimum label confidence on both endpoints.
        background_label: label treated as background.

    Returns:
        (N_verts,) bool tensor, True = semantically bad edge.
    """
    labels = pivots_labels[verts_idx]  # (N_verts, 2)
    confidence = pivots_label_confidence[verts_idx]  # (N_verts, 2)
    both_confident = (confidence >= confidence_threshold).all(dim=-1)
    both_foreground = ((labels != background_label) & (labels >= 0)).all(dim=-1)
    different = labels[:, 0] != labels[:, 1]
    return different & both_confident & both_foreground


def extract_mesh(
    delaunay_tets:torch.Tensor,
    pivots:torch.Tensor,
    pivots_sdf:torch.Tensor,
    pivots_colors:Optional[torch.Tensor]=None,
    pivots_scale:Optional[torch.Tensor]=None,
    filter_large_edges:bool=False,
    collapse_large_edges:bool=False,
    return_details:bool=False,
    sdf_sh:Optional[torch.Tensor]=None,
    mtet_on_cpu:bool=False,
    valid:Optional[torch.Tensor]=None,
    pivots_semantics:Optional[torch.Tensor]=None,
    pivots_labels:Optional[torch.Tensor]=None,
    pivots_label_confidence:Optional[torch.Tensor]=None,
    filter_semantic_edges:bool=False,
    semantic_edge_confidence:float=0.5,
    semantic_edge_length_ratio:float=0.5,
) -> Meshes:
    """
    Extract a mesh from a set of pivots, their SDF values and the Delaunay triangulation.

    Args:
        delaunay_tets (torch.Tensor): The Delaunay tetrahedra. Shape: (N_tets, 4).
        pivots (torch.Tensor): The pivots. Shape: (N_pivots, 3).
        pivots_sdf (torch.Tensor): The SDF values for the pivots. Shape: (N_pivots,).
        pivots_colors (torch.Tensor): The colors for the pivots. Shape: (N_pivots, 3).
        pivots_scale (torch.Tensor): The scales for the pivots. Shape: (N_pivots,).
        filter_large_edges (bool, optional): If True, filter out large edges. Defaults to True.
        collapse_large_edges (bool, optional): If True, collapse large edges onto the pivot with the smallest SDF value. Defaults to False.
        pivots_semantics (torch.Tensor, optional): Per-pivot semantic embeddings, shape (N_pivots, C).
            Interpolated onto mesh vertices with the same SDF weights as positions.
        pivots_labels (torch.Tensor, optional): Per-pivot instance labels, shape (N_pivots,).
        pivots_label_confidence (torch.Tensor, optional): Per-pivot label confidences, shape (N_pivots,).
        filter_semantic_edges (bool, optional): If True, drop faces whose edges confidently
            bridge two different instances (kills cross-object bridging).
        semantic_edge_confidence (float, optional): Confidence threshold for semantic edge filtering.

    Returns:
        Meshes: The extracted mesh.
    """
    # If filtering or collapsing large edges, the pivot scales must be provided
    if filter_large_edges or collapse_large_edges:
        assert pivots_scale is not None
        
    if pivots_scale is None:
        pivots_scale = torch.ones_like(pivots_sdf)

    # Applying Marching Tetrahedra
    if valid is None:
        verts_list, scale_list, faces_list, verts_idx_list = marching_tetrahedra(
            vertices=pivots[None].cpu() if mtet_on_cpu else pivots[None],
            tets=delaunay_tets.cpu() if mtet_on_cpu else delaunay_tets,
            sdf=pivots_sdf.reshape(1, -1).cpu() if mtet_on_cpu else pivots_sdf.reshape(1, -1),
            scales=pivots_scale[None].cpu() if mtet_on_cpu else pivots_scale[None],
        )
    else:
        verts_list, scale_list, faces_list, verts_idx_list = marching_tetrahedra_with_valid(
            vertices=pivots[None].cpu() if mtet_on_cpu else pivots[None],
            tets=delaunay_tets.cpu() if mtet_on_cpu else delaunay_tets,
            sdf=pivots_sdf.reshape(1, -1).cpu() if mtet_on_cpu else pivots_sdf.reshape(1, -1),
            scales=pivots_scale[None].cpu() if mtet_on_cpu else pivots_scale[None],
            valids=valid[None].cpu() if mtet_on_cpu else valid[None],
        )
        
    end_points, end_sdf = verts_list[0]  # (N_verts, 2, 3) and (N_verts, 2, 1)
    if not mtet_on_cpu:
        end_points = end_points.cuda()
        end_sdf = end_sdf.cuda()
    end_scales = scale_list[0].cuda()  # (N_verts, 2, 1)
    need_verts_idx = (
        (pivots_colors is not None)
        or (pivots_semantics is not None)
        or (pivots_labels is not None)
    )
    verts_idx = verts_idx_list[0].cuda() if need_verts_idx else None  # (N_verts, 2)
    if pivots_colors is not None:
        verts_colors = pivots_colors[verts_idx]  # (N_verts, 2, 3)
    else:
        verts_colors = None
    
    # If spherical harmonics are provided, we use them to interpolate the SDF values along edges
    if sdf_sh is not None:
        # Add zero as the 0-degree SH component
        sdf_harmonics = torch.cat(
            [
                torch.zeros(sdf_sh.shape[0], 1, device=sdf_sh.device),  # (N_voronoi_points, 1)
                sdf_sh,  # (N_voronoi_points, N_sh-1)
            ],
            dim=-1,
        )  # (N_voronoi_points, N_sh)
        
        # Compute edge directions
        edge_dir = end_points[:, 1, :] - end_points[:, 0, :]  # (N_verts, 3)
        edge_dir_normalized = edge_dir / edge_dir.norm(dim=-1, keepdim=True)  # (N_verts, 3)
        edge_dir_normalized = torch.cat(
            [
                edge_dir_normalized.unsqueeze(1),  # (N_verts, 1, 3)
                -edge_dir_normalized.unsqueeze(1),  # (N_verts, 1, 3)
            ], 
            dim=1,
        )  # (N_verts, 2, 3)
        
        sh_deg = int(np.sqrt(sdf_harmonics.shape[-1])) - 1
        
        # We apply exponential to the SDF factors computed from spherical harmonics.
        # This ensures that:
        # - The SDF factors are always positive, so they do not change the sign of the SDF values
        # - The SDF factors are initialized to 1, as the SH coefficients are initialized to 0
        # - It is easier to encode high variance accross edge directions in the SDF values
        sdf_factors = torch.exp(
            eval_sh(
                deg=sh_deg,
                sh=sdf_harmonics[verts_idx_list[0].cuda()].unsqueeze(-2),  # (N_verts, 2, 1, N_sh)
                dirs=edge_dir_normalized,  # (N_verts, 2, 3)
            )  # (N_verts, 2, 1)
        )  # (N_verts, 2, 1)
        
        # Linear interpolation along edges, adjusted with SDF factors
        end_sdf = end_sdf * sdf_factors  # (N_verts, 2, 1)
    
    # Normalizing the SDF values to get the weights for the interpolation
    norm_sdf = end_sdf.abs() / end_sdf.abs().sum(dim=1, keepdim=True)  # (N_verts, 2, 1)
    verts = end_points[:, 0, :] * norm_sdf[:, 1, :] + end_points[:, 1, :] * norm_sdf[:, 0, :]        
    faces = faces_list[0].cuda()  # (N_faces, 3)
    
    # If colors are provided, we interpolate them along edges
    if pivots_colors is not None:
        verts_colors = verts_colors[:, 0, :] * norm_sdf[:, 1, :] + verts_colors[:, 1, :] * norm_sdf[:, 0, :]

    # If semantic embeddings are provided, interpolate them with the same
    # SDF weights as the vertex positions
    if pivots_semantics is not None:
        end_semantics = pivots_semantics[verts_idx]  # (N_verts, 2, C)
        verts_semantics = (
            end_semantics[:, 0, :] * norm_sdf[:, 1, :]
            + end_semantics[:, 1, :] * norm_sdf[:, 0, :]
        )
    else:
        verts_semantics = None

    # If labels are provided, a vertex inherits the label of the endpoint
    # closest to the surface (labels are categorical, never interpolated)
    if pivots_labels is not None:
        closest_end = end_sdf.abs().argmin(dim=1).flatten()  # (N_verts,)
        vertex_range = torch.arange(verts_idx.shape[0], device=verts_idx.device)
        closest_pivot = verts_idx[vertex_range, closest_end]  # (N_verts,)
        verts_labels = pivots_labels[closest_pivot]
        verts_label_confidence = (
            pivots_label_confidence[closest_pivot]
            if pivots_label_confidence is not None
            else torch.ones_like(verts_labels, dtype=torch.float32)
        )
    else:
        verts_labels = None
        verts_label_confidence = None

    # Filtering and collapsing edges based on the distance between the pivots
    if filter_large_edges or collapse_large_edges or filter_semantic_edges:
        dmtet_distance = torch.norm(end_points[:, 0, :] - end_points[:, 1, :], dim=-1)
        dmtet_scale = end_scales[:, 0, 0] + end_scales[:, 1, 0]
        dmtet_vertex_mask = (dmtet_distance <= dmtet_scale)

    #    > Filtering for large edges, inspired by GOF.
    #      If the edge between two pivots is larger than the sum
    #      of the scales of the two corresponding Gaussians,
    #      The pivots should probably not be connected.
    if filter_large_edges:
        dmtet_face_mask = dmtet_vertex_mask[faces].all(axis=1)
        faces = faces[dmtet_face_mask]

    #    > Semantic edge filtering: an MT edge whose two pivots confidently
    #      belong to two different instances crosses an object boundary in
    #      free space; the surface it generates is a bridge artifact.
    #      Real contact surfaces between touching instances also produce
    #      label-crossing edges, but those edges are SHORT (the instances
    #      are adjacent); bridge edges span free space and are long. The
    #      length gate keeps contact geometry and removes only bridges.
    if filter_semantic_edges and (pivots_labels is not None):
        bad_semantic_verts = semantic_edge_mask(
            pivots_labels=pivots_labels,
            pivots_label_confidence=(
                pivots_label_confidence
                if pivots_label_confidence is not None
                else torch.ones_like(pivots_labels, dtype=torch.float32)
            ),
            verts_idx=verts_idx,
            confidence_threshold=semantic_edge_confidence,
        )
        bad_semantic_verts = bad_semantic_verts & (
            dmtet_distance > semantic_edge_length_ratio * dmtet_scale
        )
        semantic_face_mask = ~(bad_semantic_verts[faces].any(axis=1))
        faces = faces[semantic_face_mask]
    
    #    > The following option collapses big edges
    #      onto the pivot with the smallest SDF value.
    if collapse_large_edges:
        min_end_points = end_points[
            np.arange(end_points.shape[0]), 
            end_sdf.argmin(dim=1).flatten().cpu().numpy()
        ]  # TODO: Do the computation only for filtered vertices
        verts = torch.where(dmtet_vertex_mask[:, None], verts, min_end_points)

    # Build Mesh and return details if requested
    output_mesh = Meshes(verts=verts, faces=faces, verts_colors=verts_colors)
    if return_details:
        details = {
            'end_points': end_points,
            'end_sdf': end_sdf,
            'end_scales': end_scales,
            'end_idx': verts_idx_list[0].cuda(),
            'verts_semantics': verts_semantics,
            'verts_labels': verts_labels,
            'verts_label_confidence': verts_label_confidence,
        }
        return output_mesh, details
    else:
        return output_mesh