from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recompose.collision import is_watertight,meshes_intersect
from recompose.geometry import RayMesh
from recompose.optimize import PoseProblem
from scipy.spatial import cKDTree
import json


def tetra():
    return np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float),np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])


def test_closed_mesh_requires_no_self_intersection():
    v,f=tetra()
    assert is_watertight(v,f)
    assert not is_watertight(v,f[:-1])
    # Two individually closed components cross: topology alone is insufficient.
    assert not is_watertight(np.vstack([v,v+.2]),np.vstack([f,f+len(v)]))


def test_bvh_surface_collision_and_separation():
    v,f=tetra()
    assert meshes_intersect(v,f,v+.2,f)
    assert not meshes_intersect(v,f,v+2.,f)
    # Full enclosure has no triangle crossing; the caller's signed-distance
    # containment test handles this case separately.
    assert not meshes_intersect(v,f,v*.1+.1,f)


def test_contact_uses_triangle_distance_not_support_sample_spacing():
    p=PoseProblem.__new__(PoseProblem)
    p.samples=np.array([[0.,0.,0.]])
    p.up=np.array([0.,0.,1.]);p.d=1.;p.watertight=False
    p.support=np.array([[-.08,0,0],[.08,0,0]])
    p.support_tree=cKDTree(p.support);p.support_weight=np.ones(2)
    p.cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/recompose.json').read_text())['pose']
    p.background=RayMesh(np.array([[-1.,-1.,0],[1,-1,0],[1,1,0],[-1,1,0]]),np.array([[0,1,2],[0,2,3]]))
    _,result=p.physics(np.eye(4),np.array([0]),return_detail=True)
    assert result['contact_gap_fraction']<1e-7
    p.samples[0,2]=.02
    _,result=p.physics(np.eye(4),np.array([0]),return_detail=True)
    assert abs(result['contact_gap_fraction']-.02)<1e-7
