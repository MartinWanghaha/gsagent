"""CPU safety checks for isolated artifact corrections; no training data or CUDA."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from repair_low_psnr_cases import coverage_hole, sanitize_aura, prepare_aura_sdedit
import report_low_psnr_repairs as reporting
import run_inpaint360_full as full_report


class RepairSafetyTests(unittest.TestCase):
    def test_report_only_preserves_finished_status_without_launching_tools(self):
        with tempfile.TemporaryDirectory() as folder:
            job=Path(folder)
            state=dict(state='complete',finished_at='recorded time',pid=123,scenes={})
            (job/'request.json').write_text('{}')
            (job/'status.json').write_text(json.dumps(state))
            with patch.object(full_report.subprocess,'check_output',side_effect=AssertionError('must not launch training setup')):
                runner=full_report.FullJob(job,report_only=True)
            self.assertEqual(runner.status,state)
            self.assertEqual(json.loads((job/'status.json').read_text()),state)

    def test_report_correction_preserves_baseline_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            job=Path(folder);source=job/'evaluation.json'
            scores={m:float(i+1) for i,m in enumerate(full_report.METRICS)}
            source.write_text(json.dumps(dict(complete=True,inputs=dict(iteration=10000),metrics=scores)))
            record=dict(method='PaintMesh 10000',scene='cube',source=str(source),sha256=full_report.sha(source),format='paintmesh',iteration=10000)
            (job/'report_corrections.json').write_text(json.dumps(dict(records=[record])))
            original={'PaintMesh 10000':{'cube':{m:0.0 for m in scores}}}
            result,_=full_report.report_corrections(job,original)
            self.assertEqual(result['PaintMesh 10000']['cube'],scores)
            self.assertEqual(original['PaintMesh 10000']['cube']['PSNR'],0.0)
            source.write_text('{}')
            with self.assertRaisesRegex(ValueError,'source changed'):
                full_report.report_corrections(job,original)

    def test_sdedit_rerun_archives_only_equivalent_reference_mask(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'data';model=Path(folder)/'model'
            paths=[data/'images',data/'unseen_masks_dilated',model/'train/ours_object_inpaint_init/renders']
            for p in paths:p.mkdir(parents=True)
            (data/'dataset_manifest.json').write_text(json.dumps({'training_images':['IMG_0001.JPG']}))
            Image.new('RGB',(16,16)).save(paths[0]/'IMG_0001.JPG')
            Image.new('RGB',(16,16)).save(paths[2]/'00000.png')
            Image.new('L',(16,16),255).save(paths[1]/'IMG_0001.JPG')
            Image.new('L',(16,16),0).save(paths[1]/'IMG_0001.png')
            with self.assertRaisesRegex(ValueError,'Conflicting duplicate'):
                prepare_aura_sdedit(data,model)
            self.assertTrue((paths[1]/'IMG_0001.png').exists())
            Image.new('L',(16,16),255).save(paths[1]/'IMG_0001.png')
            prepare_aura_sdedit(data,model)
            prepare_aura_sdedit(data,model)  # idempotent preflight
            self.assertEqual([p.name for p in paths[1].iterdir()],['IMG_0001.JPG'])
            self.assertTrue((data/'sdedit_duplicate_mask_archive/IMG_0001.png').exists())
            proof=json.loads((data/'sdedit_input_validation.json').read_text())
            self.assertEqual(len(proof['archived_duplicates']),1)

    def test_completed_scene_metrics_survive_vendor_aggregate_overwrite(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(reporting,'BASE',Path(folder)):
            root=Path(folder)/'gaussian-grouping/output/low_psnr_repair_20261026/workspace/output/inpaint360'
            for scene,score in [('cone_red',19.2),('garden_toys',20.77)]:
                file=root/scene/'repair_results.json';file.parent.mkdir(parents=True)
                file.write_text(json.dumps({'per_scene_results':{f'/output/inpaint360/{scene}/inpaint/final':{k:score for k in reporting.NATIVE}}}))
            # The shared file has been overwritten: independent snapshots survive.
            (root/'all_scene_evaluation_results.json').write_text('{"per_scene_results": {}}')
            result=reporting.gg_repairs()
            self.assertEqual(set(result),{'cone_red','garden_toys'})
            self.assertEqual(result['cone_red']['PSNR'],19.2)

    def test_background_bytes_and_retained_attributes_are_preserved(self):
        dtype=np.dtype([(k,'f4') for k in ('x','y','z','opacity','scale_0','f_dc_0')])
        old=np.zeros(4,dtype=dtype);old['f_dc_0']=np.arange(4)
        new=np.zeros(402,dtype=dtype)
        grid=np.stack(np.meshgrid(np.linspace(0,1,20),np.linspace(0,1,20)),axis=-1).reshape(-1,2)
        new['x'][:400]=grid[:,0];new['y'][:400]=grid[:,1]
        new['x'][-2:]=[10,20];new['z'][-2:]=[20,40]
        new['opacity']=np.inf;new['scale_0']=.01;new['f_dc_0']=np.arange(402)
        corrected,stats=sanitize_aura(old,np.concatenate([old,new]))
        self.assertEqual(corrected[:4].tobytes(),old.tobytes())
        self.assertEqual(stats['removed_new_indices'],[400,401])
        for k in ('x','y','z','scale_0','f_dc_0'):
            np.testing.assert_array_equal(corrected[k][4:],new[k][:400])
        self.assertTrue(np.isfinite(corrected['opacity'][4:]).all())
        alpha=1/(1+np.exp(-corrected['opacity'][4:]))
        self.assertTrue(((alpha>0)&(alpha<1)).all())

    def test_changed_old_points_abort_before_writing(self):
        dtype=np.dtype([(k,'f4') for k in ('x','y','z','opacity')])
        old=np.zeros(4,dtype=dtype);initial=np.zeros(30,dtype=dtype)
        initial['x'][0]=1
        with self.assertRaisesRegex(ValueError,'Background changed'):
            sanitize_aura(old,initial)

    def test_hole_excludes_valid_table_and_remote_background_island(self):
        old=np.zeros((100,100),bool);old[5:95,5:95]=True
        alpha=np.ones((100,100));alpha[40:60,40:60]=0;alpha[10:20,10:20]=0
        mask=coverage_hole(old,alpha)
        self.assertTrue(mask[45,45]);self.assertFalse(mask[15,15]);self.assertFalse(mask[35,50])
        self.assertEqual(int(mask.sum()),400)
        alpha[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Invalid alpha'):
            coverage_hole(old,alpha)


if __name__=='__main__':unittest.main()
