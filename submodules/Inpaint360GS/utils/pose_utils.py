import numpy as np
from typing import Tuple
import torch
# from icecream import ic
from utils.stepfun import sample_np
from utils.graphics_utils import getWorld2View2


def normalize(x):
    return x / np.linalg.norm(x)

def get_focal(camera):
    focal = camera.FoVx
    return focal

def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def viewmatrix(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    #  （PCA）
    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return poses_recentered, transform

def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def generate_virtual_radius(views, target_object_radius=None):

    """
    Automatically calculates the optimal radius for a virtual camera path based on 
    the target object's size and the camera's Field of View (FOV).

    The function ensures the target object is perfectly framed (occupying ~50% of the screen)
    to provide enough surrounding context for 3D Inpainting.

    Args:
        views (list): List of training camera objects containing R, T, and FOV.
        target_object_radius (float): The 3D radius of the object to be inpainted.

    Returns:
        circle_radius (The ratio of the ideal path distance to the original scene scale).
    """
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)            #  world to camera pose
        tmp_view[:, 1:3] *= -1                       
        poses.append(tmp_view)
    poses = np.stack(poses, 0)     # (79, 4, 4)
    poses, transform = transform_poses_pca(poses)

    center = focus_point_fn(poses)

    offset = np.array([center[0] , center[1],  0 ])                   
    # Calculate scaling for ellipse axes based on input camera positions.
    sc_base = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)

    scale_factor = np.linalg.norm(transform[:3, :3], axis=0).mean()
    pca_target_r = target_object_radius * scale_factor
    
    fov_x = views[0].FoVx 
    fov_y = views[0].FoVy 
    
    # Use Pinhole Camera Model Trigonometry: Distance = Radius / tan(FOV / 2)
    dist_x = pca_target_r / np.tan(fov_x / 2.0)
    dist_y = pca_target_r / np.tan(fov_y / 2.0)

    # Dividing by 0.7 ensures the object occupies roughly 70% of the viewport.
    # This provides sufficient background pixels for high-quality inpainting.
    ideal_distance_pca = max(dist_x, dist_y) / 0.7
    
    circle_radius = ideal_distance_pca / np.max(sc_base)
    
    print(f"---------------------------------------")
    print(f"generated circle_radius : {circle_radius:.4f}")
    print(f"---------------------------------------")
            
    return circle_radius

def generate_ellipse_path(views, n_frames=240, const_speed=True, z_variation=0., z_phase=0., 
                          is_circle=False, circle_radius=1.0, ellipse_radius=1.0, gaussians=None, 
                          object_centered=False):

    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)            #  world to camera pose
        tmp_view[:, 1:3] *= -1                       
        poses.append(tmp_view)
    poses = np.stack(poses, 0)     # (79, 4, 4)
    poses, transform = transform_poses_pca(poses)

    if object_centered:
        xyz = gaussians.get_xyz.cpu().numpy()
        xyz_homo = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)  # (N, 4)

        xyz_transformed = (transform @ xyz_homo.T).T[:, :3]  # shape (N, 3)
        center = xyz_transformed.mean(axis=0)
    else:
        # Calculate the focal point for the path (cameras point toward this).
        center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0] , center[1],  0 ])                   
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0) * ellipse_radius   # training dataset camera pose， ellipse axes

    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    if is_circle:
        r = np.max(sc) * circle_radius 
        def get_positions(theta):
            return np.stack([
                center[0] + r * np.cos(theta),
                center[1] + r * np.sin(theta),
                z_variation * (z_low[2] + (z_high - z_low)[2] *
                               (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
            ], -1)
    else:   
        def get_positions(theta):
            # Interpolate between bounds with trig functions to get ellipse in x-y.
            # Optionally also interpolate in z to change camera height along path.
            return np.stack([
                (low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5)),
                (low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5)),
                z_variation * (z_low[2] + (z_high - z_low)[2] *
                            (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
            ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    if const_speed:
        # Resample theta angles so that the velocity is closer to constant.
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        theta = sample_np(None, theta, np.log(lengths), n_frames + 1)
        positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    
    render_poses = []
    for p in positions:
        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(p - center, up, p)
        render_pose = np.linalg.inv(transform) @ render_pose
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    
    return render_poses


def generate_virtual_path(views, n_frames=30, path_type="circle", circle_radius=1.0,
                          max_elevation_deg=85.0):
    """Peer circular/hemispherical paths, preserving the legacy PCA similarity.

    Returns world-to-camera matrices and generation provenance. Default circle
    calls the untouched legacy implementation, including its speed sampling.
    """
    import hashlib
    import json
    from utils.virtual_camera_manifest import (
        check_camera_request, hemisphere_spiral_layout,
        HEMISPHERE_ALGORITHM_REVISION,
    )

    check_camera_request(None, path_type, n_frames, max_elevation_deg)
    if not np.isfinite(circle_radius) or circle_radius <= 0:
        raise ValueError("circle radius ratio must be finite and positive")
    poses = []
    source = []
    for view in views:
        w2c = np.eye(4)
        w2c[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        source.append(dict(matrix=w2c.tolist(), FoVx=float(view.FoVx), FoVy=float(view.FoVy)))
        c2w = np.linalg.inv(w2c)
        c2w[:, 1:3] *= -1
        poses.append(c2w)
    if len(poses) < 3 or not np.isfinite(poses).all():
        raise ValueError("virtual path requires at least three finite training poses")
    pca, transform = transform_poses_pca(np.stack(poses))
    focus = focus_point_fn(pca)
    center = np.array([focus[0], focus[1], 0.])
    radius = np.max(np.percentile(np.abs(pca[:, :3, 3] - center), 90, axis=0)) * circle_radius
    if not np.isfinite(focus).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError("degenerate camera focus/radius")
    legacy = generate_ellipse_path(views, n_frames=n_frames,
        is_circle=True, circle_radius=circle_radius)
    first = transform @ np.linalg.inv(legacy[0])
    first[:3, 1:3] *= -1
    theta0 = float(np.arctan2(first[1, 3] - center[1], first[0, 3] - center[0]))
    provenance = dict(type=path_type, algorithm_revision=1, frame_count=n_frames,
        center_pca=center.tolist(), focus_pca=focus.tolist(), radius_pca=float(radius),
        world_to_pca=transform.tolist(), start_azimuth_rad=theta0, up_axis="+z_pca",
        source_cameras_sha256=hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    if path_type == "circle":
        provenance.update(sampling="legacy_circle", orientation="legacy_up")
        return legacy, provenance

    offsets, pitch = hemisphere_spiral_layout(n_frames, max_elevation_deg)
    cs, sn = np.cos(theta0), np.sin(theta0)
    azimuth_rotation = np.array([[cs, -sn, 0.], [sn, cs, 0.], [0., 0., 1.]])
    positions = center + radius * (offsets @ azimuth_rotation.T)
    positions[0] = first[:3, 3]
    avg_up = pca[:, :3, 1].mean(0)
    axis = np.argmax(np.abs(avg_up))
    up = np.eye(3)[axis] * np.sign(avg_up[axis])
    if not np.isfinite(up).all() or np.linalg.norm(up) == 0:
        raise ValueError("training cameras do not define a scene up direction")
    # Every frame is anchored to the same scene up, never to the preceding
    # camera's right/up. This constrains absolute roll over arbitrarily many turns.
    result = []
    inverse = np.linalg.inv(transform)
    for i, position in enumerate(positions):
        backward = position - focus
        length = np.linalg.norm(backward)
        if not np.isfinite(length) or length <= 1e-12:
            raise ValueError(f"camera {i} coincides with its look-at point")
        backward /= length
        right = np.cross(up, backward)
        if np.linalg.norm(right) <= 1e-12:
            # Roll is undefined when the reference up is parallel to the view.
            # Do not silently introduce a history-dependent orientation here.
            raise ValueError(f"camera {i}: scene up is parallel to the view; lower hemisphere elevation")
        right /= np.linalg.norm(right)
        c2w = np.eye(4)
        c2w[:3, :3] = np.column_stack([right, np.cross(backward, right), backward])
        c2w[:3, 3] = position
        c2w = inverse @ c2w
        c2w[:3, 1:3] *= -1
        result.append(legacy[0] if i == 0 else np.linalg.inv(c2w))
    if not np.isfinite(result).all():
        raise ValueError("nonfinite hemisphere camera")
    provenance.update(algorithm_revision=HEMISPHERE_ALGORITHM_REVISION,
        sampling="equal_height_auto_pitch", orientation="scene_up", scene_up_pca=up.tolist(),
        max_elevation_deg=float(max_elevation_deg), pitch_rad=pitch,
        turns=float(np.deg2rad(max_elevation_deg)/pitch))
    return result, provenance


def generate_spherify_path(views):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)

    p34_to_44 = lambda p: np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    )

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        render_pose = np.eye(4)
        render_pose[:3] = p
        #render_pose[:3, 1:3] *= -1
        new_poses.append(render_pose)

    new_poses = np.stack(new_poses, 0)
    return new_poses

def get_rotation_matrix(axis, angle):
    """
    Create a rotation matrix for a given axis (x, y, or z) and angle.
    """
    axis = axis.lower()
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, cos_angle, -sin_angle],
            [0, sin_angle, cos_angle]
        ])
    elif axis == 'y':
        return np.array([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ])
    elif axis == 'z':
        return np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', 'z'.")


def circular_poses(viewpoint_cam, radius, angle=0.0):
    translate_x = radius * np.cos(angle)
    translate_y = radius * np.sin(angle)
    translate_z = 0
    translate = np.array([translate_x, translate_y, translate_z])
    viewpoint_cam.world_view_transform = torch.tensor(getWorld2View2(viewpoint_cam.R, viewpoint_cam.T, translate)).transpose(0, 1).cuda()
    viewpoint_cam.full_proj_transform = (viewpoint_cam.world_view_transform.unsqueeze(0).bmm(viewpoint_cam.projection_matrix.unsqueeze(0))).squeeze(0)
    viewpoint_cam.camera_center = viewpoint_cam.world_view_transform.inverse()[3, :3]

    return viewpoint_cam

def generate_spherical_sample_path(views, azimuthal_rots=1, polar_rots=0.75, N=10):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
        focal = get_focal(view)
    poses = np.stack(poses, 0)
    # ic(min_focal, max_focal)
    
    c2w = poses_avg(poses)  
    up = normalize(poses[:, :3, 1].sum(0))  
    rads = np.percentile(np.abs(poses[:, :3, 3]), 90, 0)
    rads = np.array(list(rads) + [1.0])
    ic(rads)
    render_poses = []
    focal_range = np.linspace(0.5, 3, N **2+1)
    index = 0
    # Modify this loop to include phi
    for theta in np.linspace(0.0, 2.0 * np.pi * azimuthal_rots, N + 1)[:-1]:
        for phi in np.linspace(0.0, np.pi * polar_rots, N + 1)[:-1]:
            # Modify these lines to use spherical coordinates for c
            c = np.dot(
                c2w[:3, :4],
                rads * np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi),
                    1.0
                ])
            )
            
            z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal_range[index], 1.0])))
            render_pose = np.eye(4)
            render_pose[:3] = viewmatrix(z, up, c)  
            render_pose[:3, 1:3] *= -1
            render_poses.append(np.linalg.inv(render_pose))
            index += 1
    return render_poses


def generate_spiral_path(views, focal=1.5, zrate= 0, rots=1, N=600):
    poses = []
    focal = 0
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
        focal += get_focal(views[0])
    poses = np.stack(poses, 0)


    c2w = poses_avg(poses)
    up = normalize(poses[:, :3, 1].sum(0))

    # Get radii for spiral path
    rads = np.percentile(np.abs(poses[:, :3, 3]), 90, 0)
    render_poses = []

    rads = np.array(list(rads) + [1.0])
    focal /= len(views)

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta),-np.sin(theta * zrate), 1.0]) * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))

        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(z, up, c)
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    return render_poses
