"""Checks for camera convention, physical scale and visibility-aware samples."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark import surfel_export as module


def frame(depth=2.,height=4,width=5):
    return dict(depth=np.full((height,width),depth,dtype='float32'),
        normal=np.broadcast_to(np.array([0.,0.,-1.],dtype='float32'),(height,width,3)).copy(),
        rgb=np.broadcast_to(np.array([31,127,229],dtype='uint8'),(height,width,3)).copy(),
        K=np.array([[100.,0,2],[0,80.,1],[0,0,1.]]),w2c=np.eye(4),
        alpha=np.ones((height,width),dtype='float32'),confidence=np.ones((height,width),dtype='float32'))


def position(surfels):return np.column_stack([surfels[k] for k in ['x','y','z']])
def normal(surfels):return np.column_stack([surfels[k] for k in ['nx','ny','nz']])


class SurfelExportTests(unittest.TestCase):
    def test_z_depth_integer_pixel_centers_and_world_normal_rotation(self):
        f=frame();angle=.7
        R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        f['w2c'][:3,:3]=R;f['w2c'][:3,3]=[.3,-.2,.8]
        points,stats=module.frame_surfels(f)
        pixels=np.stack(np.meshgrid(np.arange(5),np.arange(4),indexing='xy'),-1).reshape(-1,2)
        camera=np.c_[(pixels[:,0]-2)/100,(pixels[:,1]-1)/80,np.ones(20)]*2
        homogeneous=np.c_[camera,np.ones(20)]@np.linalg.inv(f['w2c']).T
        np.testing.assert_allclose(position(points),homogeneous[:,:3],rtol=0,atol=2e-7)
        np.testing.assert_allclose(normal(points),np.broadcast_to([0,0,-1.]@R,(20,3)),rtol=0,atol=1e-7)
        self.assertEqual(stats['pixel_center_offset'],0)
        self.assertEqual(len(points),20)

    def test_positive_fssr_value_is_pixel_tangent_footprint(self):
        f=frame();samples,_=module.frame_surfels(f)
        np.testing.assert_allclose(samples['value'],2/np.sqrt(100*80),rtol=1e-7)
        np.testing.assert_array_equal(samples['value'],samples['scale'])
        n=np.array([.5,.2,-1.]);n/=np.linalg.norm(n);f['normal'][:]=n
        samples,_=module.frame_surfels(f)
        index=7;r=np.array([0.,0.,1.]);z=2.
        du=z*(np.array([1/100,0,0])-r*n[0]/(100*np.dot(n,r)))
        dv=z*(np.array([0,1/80,0])-r*n[1]/(80*np.dot(n,r)))
        expected=np.sqrt(np.linalg.norm(np.cross(du,dv)))
        self.assertAlmostEqual(float(samples['value'][index]),expected,places=8)
        self.assertTrue((samples['value']>0).all())

    def test_soft_visibility_does_not_penalize_occlusion_or_unsupported_points(self):
        source=frame();samples,_=module.frame_surfels(source)
        for target,label in [(frame(1.),'occluded'),(frame(0.),'unsupported'),(frame(2.),'consistent')]:
            weights,stats=module.consistency_weights(samples,0,module.prepare_targets([source,target]),chunk_points=7)
            np.testing.assert_allclose(weights,1,rtol=0,atol=1e-7)
            self.assertFalse(stats['hard_multiview_point_rejection'])
            if label=='occluded':self.assertGreater(stats['occluded_observations'],0)
            if label=='unsupported':self.assertEqual(stats['points_without_other_view_evidence'],20)
        target=frame(3.)
        weights,stats=module.consistency_weights(samples,0,module.prepare_targets([source,target]),chunk_points=7)
        self.assertTrue((weights<.8).all());self.assertTrue((weights>0).all())
        self.assertEqual(len(weights),20);self.assertEqual(stats['free_space_conflicts'],20)

    def test_opposite_thin_surface_orientations_are_not_averaged(self):
        first=frame();other=frame();other['w2c'][:3,:3]=np.diag([1.,-1.,-1.]);other['w2c'][:3,3]=[0,0,4]
        a,_=module.frame_surfels(first);b,_=module.frame_surfels(other)
        np.testing.assert_allclose(normal(a),np.broadcast_to([0,0,-1],(20,3)))
        np.testing.assert_allclose(normal(b),np.broadcast_to([0,0,1],(20,3)))
        weights,stats=module.consistency_weights(a,0,module.prepare_targets([first,other]))
        np.testing.assert_allclose(weights,1)
        self.assertGreater(stats['opposite_side_observations'],0)
        self.assertFalse(stats['normal_averaging'])

    def test_base_alpha_is_not_applied_twice_and_invalid_fields_accounted(self):
        f=frame();f['alpha'][:]=.5;f['confidence'][:]=.25
        samples,stats=module.frame_surfels(f)
        # At (u=cx,v=cy) incidence=1, so confidence remains .25, not .125.
        self.assertAlmostEqual(float(samples['confidence'][7]),.25)
        f['depth'][0,0]=np.nan;f['normal'][0,1]=0
        samples,stats=module.frame_surfels(f)
        self.assertEqual(len(samples),18);self.assertEqual(stats['image_pixels'],20)
        self.assertEqual(stats['finite_positive_depth_pixels'],19)
        self.assertTrue(np.isfinite(position(samples)).all())

    def test_complete_full_frame_queue_binary_ply_and_exact_resume(self):
        import torch
        from plyfile import PlyData
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as name:
            out=Path(name);folder=out/'frames';folder.mkdir()
            records=[]
            for i in range(2):
                p=folder/f'{i}.npz';np.savez(p,**frame())
                records.append(dict(name=f'{i}.JPG',path=str(p),identity=module.identity(p)))
            source=dict(complete=True,frames=records,request=dict(train_views=2))
            prepared=module.prepare(out,source)
            with self.assertRaises(ValueError):module.finalize(out)
            self.assertEqual(module.run_worker(out,'cpu0',device='cpu'),2)
            self.assertEqual(module.run_worker(out,'cpu1',device='cpu'),0)
            result=module.finalize(out)
            self.assertTrue(result['all_source_frames']);self.assertEqual(result['frames'],2)
            self.assertEqual(result['points'],40);self.assertTrue(result['no_explicit_deduplication'])
            self.assertEqual(module.prepare(out,source)['request_id'],prepared['request_id'])
            mesh=PlyData.read(result['records'][0]['path']);v=mesh['vertex'].data
            self.assertEqual(len(v),20);self.assertFalse(mesh.text);self.assertEqual(mesh.byte_order,'<')
            self.assertEqual(v.dtype['red'].kind,'u');np.testing.assert_array_equal(v['value'],v['scale'])
            self.assertTrue((v['value']>0).all());np.testing.assert_array_equal(v['red'],np.full(20,31))
            self.assertEqual(module.progress(out)['completed_frames'],2)
            # A changed original frame invalidates its receipt, rather than
            # silently exporting different evidence into an old reconstruction.
            p=Path(records[0]['path']);np.savez(p,**frame(2.1))
            with self.assertRaises(ValueError):module.finalize(out)

    def test_exclusive_claim_prevents_two_workers_owning_same_frame(self):
        with tempfile.TemporaryDirectory() as name:
            out=Path(name);row=dict(index=0)
            with module.claim(out,row,'test') as first:
                self.assertTrue(first)
                with module.claim(out,row,'test') as second:self.assertFalse(second)
            with module.claim(out,row,'test') as after:self.assertTrue(after)

    def test_fssr_depth_ratio_guard_rejects_unsupported_extent_without_clipping(self):
        with tempfile.TemporaryDirectory() as name:
            out=Path(name);records=[]
            for i in range(2):
                data=frame()
                if i:data['w2c'][0,3]=100000.
                p=out/f'{i}.npz';np.savez(p,**data)
                records.append(dict(name=f'{i}.JPG',path=str(p),identity=module.identity(p)))
            module.prepare(out,dict(complete=True,frames=records))
            module.run_worker(out,'cpu',device='cpu')
            with self.assertRaisesRegex(ValueError,'20-level'):module.finalize(out)
            report=json.loads((out/'surfels/preflight.json').read_text())
            self.assertFalse(report['supported']);self.assertFalse(report['scale_clipping'])
            self.assertGreater(report['extent_to_minimum_footprint_ratio'],2**20)
            self.assertFalse((out/'surfels/complete.json').exists())


if __name__=='__main__':unittest.main()
