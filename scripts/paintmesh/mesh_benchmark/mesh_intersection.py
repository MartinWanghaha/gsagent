"""Complete triangle-surface intersections, with float64 contact classification.

FCL supplies BVH candidate face pairs. Its contact position/depth are deliberately
not used: mesh/mesh contact depths are not an object penetration distance. Each
pair is checked against both triangle planes and its actual intersection is
computed here. Open environment meshes need no global inside/outside sign.

``transverse`` means a positive-length intersection and both triangles strictly
straddle the other triangle's plane. Edge/vertex and coplanar contact are kept
separate. Boundary contact can still be part of penetration along a mesh edge;
classification alone is not a physical solid-overlap or support-contact test.
Containment without intersecting surfaces must be checked separately.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


CLASSES = ('transverse', 'boundary_touch', 'coplanar_overlap', 'point_touch',
           'degenerate', 'disjoint')


def _mesh(vertices, faces):
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    raw = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError('Finite [N,3] vertices required')
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError('Integer [M,3] triangle indices required')
    f = np.ascontiguousarray(raw, dtype=np.int64)
    if len(f) and (f.min() < 0 or f.max() >= len(v)):
        raise ValueError('Out-of-range triangle index')
    return v, f


def _fcl_model(vertices, faces):
    import fcl
    # FCL otherwise stores a copy of all input vertices for each recursive chunk.
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    model = fcl.BVHModel()
    model.beginModel()
    model.addSubModel(np.ascontiguousarray(vertices[used]),
                      np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int32))
    model.endModel()
    return fcl.CollisionObject(model), model


def _candidate_pairs(va, fa, vb, fb, cap):
    """Enumerate all colliding face pairs; a contact cap is never truncation."""
    import fcl
    found = []
    stats = dict(bvh_queries=0, contact_cap_subdivisions=0, raw_contacts=0)

    def query(ia, ib, amodel=None, bmodel=None):
        aa = _fcl_model(va, fa[ia]) if amodel is None else amodel
        bb = _fcl_model(vb, fb[ib]) if bmodel is None else bmodel
        result = fcl.CollisionResult()
        fcl.collide(aa[0], bb[0], fcl.CollisionRequest(
            num_max_contacts=cap, enable_contact=True), result)
        stats['bvh_queries'] += 1
        contacts = result.contacts
        # A single face pair is fully identified even if FCL emits/caps multiple
        # contact points for it. Otherwise discard capped results and subdivide.
        if len(contacts) >= cap and (len(ia) > 1 or len(ib) > 1):
            stats['contact_cap_subdivisions'] += 1
            if len(ib) > 1:
                mid = len(ib) // 2
                query(ia, ib[:mid], aa)
                query(ia, ib[mid:], aa)
            else:
                mid = len(ia) // 2
                query(ia[:mid], ib, bmodel=bb)
                query(ia[mid:], ib, bmodel=bb)
            return
        if result.is_collision and not contacts:
            raise RuntimeError('FCL reported collision without enabled contacts')
        stats['raw_contacts'] += len(contacts)
        found.extend((int(ia[c.b1]), int(ib[c.b2])) for c in contacts)

    if len(fa) and len(fb):
        query(np.arange(len(fa)), np.arange(len(fb)))
    pairs = (np.unique(np.asarray(found, dtype=np.int64), axis=0) if found
             else np.empty((0, 2), dtype=np.int64))
    return pairs, stats


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _coplanar_contact(a, b, normal, tolerance):
    """Clip a coplanar triangle against the other; retain contact geometry."""
    drop = int(np.argmax(np.abs(normal)))
    axes = [i for i in range(3) if i != drop]
    origin = a[0]
    aa, bb = a-origin, b-origin
    b2 = bb[:, axes]
    orientation = np.sign(_cross2(b2[1]-b2[0], b2[2]-b2[0]))
    polygon = [p.copy() for p in aa]
    for j in range(3):
        edge_start, edge_end = b2[j], b2[(j+1) % 3]
        edge = edge_end-edge_start
        epsilon = tolerance*np.linalg.norm(edge)

        def distance(p):
            return orientation*_cross2(edge, p[axes]-edge_start)

        clipped = []
        if not polygon:
            break
        previous, previous_d = polygon[-1], distance(polygon[-1])
        for current in polygon:
            current_d = distance(current)
            previous_in, current_in = previous_d >= -epsilon, current_d >= -epsilon
            if previous_in != current_in:
                denom = previous_d-current_d
                if abs(denom) > np.finfo(float).tiny:
                    clipped.append(previous+(current-previous)*np.clip(previous_d/denom, 0, 1))
            if current_in:
                clipped.append(current)
            previous, previous_d = current, current_d
        polygon = clipped
    if not polygon:
        return 'disjoint', np.full((2, 3), np.nan), np.full(3, np.nan), 0.
    polygon = np.asarray(polygon)
    distance2 = np.sum((polygon[:, None]-polygon[None, :])**2, axis=-1)
    i, j = np.unravel_index(np.argmax(distance2), distance2.shape)
    segment = polygon[[i, j]]+origin
    area = .5*np.linalg.norm(np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(0))
    kind = ('coplanar_overlap' if area > tolerance**2 else
            'boundary_touch' if distance2[i, j] > tolerance**2 else 'point_touch')
    return kind, segment, polygon.mean(0)+origin, float(area)


def _plane_interval(triangle, distances, origin, direction, tolerance):
    """Project a triangle/plane intersection onto the planes' line direction."""
    count = len(triangle)
    values = np.full((count, 6), np.nan)
    for i in range(3):
        zero = np.abs(distances[:, i]) <= tolerance
        values[zero, i] = np.einsum('ij,ij->i', triangle[zero, i]-origin[zero], direction[zero])
        j = (i+1) % 3
        cross = ((distances[:, i] < -tolerance) & (distances[:, j] > tolerance) |
                 (distances[:, i] > tolerance) & (distances[:, j] < -tolerance))
        if cross.any():
            t = distances[cross, i]/(distances[cross, i]-distances[cross, j])
            point = triangle[cross, i]+t[:, None]*(triangle[cross, j]-triangle[cross, i])
            values[cross, i+3] = np.einsum('ij,ij->i', point-origin[cross], direction[cross])
    exists = np.isfinite(values).any(1)
    lo = np.where(np.isfinite(values), values, np.inf).min(1)
    hi = np.where(np.isfinite(values), values, -np.inf).max(1)
    return lo, hi, exists


def classify_triangle_pairs(triangles_a, triangles_b, tolerance=1e-8):
    """Classify equally sized [N,3,3] arrays; no broad phase is performed."""
    a, b = np.asarray(triangles_a, dtype=np.float64), np.asarray(triangles_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (3, 3):
        raise ValueError('Matched [N,3,3] triangle arrays required')
    if tolerance <= 0 or not np.isfinite(tolerance) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Positive tolerance and finite coordinates required')
    count = len(a)
    kind = np.full(count, 'disjoint', dtype='<U20')
    segments = np.full((count, 2, 3), np.nan)
    positions = np.full((count, 3), np.nan)
    coplanar_area = np.zeros(count)
    na = np.cross(a[:, 1]-a[:, 0], a[:, 2]-a[:, 0])
    nb = np.cross(b[:, 1]-b[:, 0], b[:, 2]-b[:, 0])
    la, lb = np.linalg.norm(na, axis=1), np.linalg.norm(nb, axis=1)
    ea = np.max(np.linalg.norm(a-np.roll(a, -1, axis=1), axis=2), axis=1)
    eb = np.max(np.linalg.norm(b-np.roll(b, -1, axis=1), axis=2), axis=1)
    degenerate = (la <= tolerance*ea) | (lb <= tolerance*eb)
    kind[degenerate] = 'degenerate'
    na = np.divide(na, la[:, None], out=np.zeros_like(na), where=la[:, None] > 0)
    nb = np.divide(nb, lb[:, None], out=np.zeros_like(nb), where=lb[:, None] > 0)
    db = np.einsum('nki,ni->nk', b-a[:, :1], na)
    da = np.einsum('nki,ni->nk', a-b[:, :1], nb)
    line = np.cross(na, nb)
    line_length = np.linalg.norm(line, axis=1)
    parallel = line_length <= 1e-12
    coplanar = (~degenerate & parallel & (np.abs(db).max(1) <= tolerance)
                & (np.abs(da).max(1) <= tolerance))
    for i in np.flatnonzero(coplanar):
        kind[i], segments[i], positions[i], coplanar_area[i] = _coplanar_contact(a[i], b[i], na[i], tolerance)
    available = np.flatnonzero(~degenerate & ~parallel)
    if len(available):
        u = line[available]/line_length[available, None]
        origin = a[available, 0]
        alo, ahi, ahit = _plane_interval(a[available], da[available], origin, u, tolerance)
        blo, bhi, bhit = _plane_interval(b[available], db[available], origin, u, tolerance)
        lo, hi = np.maximum(alo, blo), np.minimum(ahi, bhi)
        valid = ahit & bhit & (hi >= lo-tolerance)
        ids = available[valid]
        if len(ids):
            direction = u[valid]
            # Solve both planes in coordinates relative to A's first vertex.
            c = np.einsum('ni,ni->n', nb[ids], b[ids, 0]-a[ids, 0])
            base = a[ids, 0]+np.cross(-c[:, None]*na[ids], line[ids])/line_length[ids, None]**2
            middle = (lo[valid]+hi[valid])*.5
            lower = np.minimum(lo[valid], middle)
            upper = np.maximum(hi[valid], middle)
            segments[ids, 0] = base+direction*lower[:, None]
            segments[ids, 1] = base+direction*upper[:, None]
            positions[ids] = base+direction*middle[:, None]
            positive_length = hi[valid]-lo[valid] > tolerance
            strict_a = (da[ids].min(1) < -tolerance) & (da[ids].max(1) > tolerance)
            strict_b = (db[ids].min(1) < -tolerance) & (db[ids].max(1) > tolerance)
            kind[ids] = np.where(positive_length,
                np.where(strict_a & strict_b, 'transverse', 'boundary_touch'), 'point_touch')
    return dict(classification=kind, segments=segments, positions=positions,
                coplanar_overlap_area=coplanar_area)


def audit_intersections(vertices_a, faces_a, vertices_b, faces_b, tolerance=1e-8,
                        *, max_contacts=131072, pair_batch=65536):
    """Return complete verified face pairs and geometric contact evidence.

    Indices in ``face_pairs`` refer to the input face arrays, not FCL's internal
    compacted vertex arrays. All arrays have matching first dimension. FCL false
    positive pairs classified disjoint are excluded and separately counted.
    ``complete`` refers to surface-pair enumeration, not solid containment.
    """
    va, fa = _mesh(vertices_a, faces_a)
    vb, fb = _mesh(vertices_b, faces_b)
    if tolerance <= 0 or not np.isfinite(tolerance) or max_contacts < 1 or pair_batch < 1:
        raise ValueError('Positive tolerance, contact cap and pair batch required')
    pairs, query_stats = _candidate_pairs(va, fa, vb, fb, int(max_contacts))
    parts = []
    for start in range(0, len(pairs), pair_batch):
        selected = pairs[start:start+pair_batch]
        parts.append(classify_triangle_pairs(va[fa[selected[:, 0]]], vb[fb[selected[:, 1]]], tolerance))
    combined = {key: np.concatenate([p[key] for p in parts]) if parts else np.empty(shape, dtype=dtype)
                for key, shape, dtype in [('classification', (0,), '<U20'),
                    ('segments', (0, 2, 3), float), ('positions', (0, 3), float),
                    ('coplanar_overlap_area', (0,), float)]}
    keep = combined['classification'] != 'disjoint'
    disjoint = int((~keep).sum())
    pairs = pairs[keep]
    combined = {key: value[keep] for key, value in combined.items()}
    crossing = combined['classification'] == 'transverse'
    counts = {key: int(np.count_nonzero(combined['classification'] == key)) for key in CLASSES}
    return dict(complete=True, tolerance=float(tolerance), face_pairs=pairs,
                **combined, counts=counts, fcl_candidate_pairs=int(len(keep)),
                rejected_disjoint_pairs=disjoint, query_statistics=query_stats,
                crossing_face_pairs=pairs[crossing], crossing_segments=combined['segments'][crossing],
                crossing_positions=combined['positions'][crossing],
                containment_checked=False, global_signed_distance_used=False,
                algorithm='FCL BVH candidate pairs + float64 triangle-plane/line and coplanar clipping',
                classification_warning='Boundary/coplanar contact may include penetrating mesh edges; transverse is a sufficient, not necessary, indication of intersecting surface interiors.')
