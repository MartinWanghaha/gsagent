import unittest
import numpy as np
import torch
from lama_context_predict import canvas,optimize_features


class Front(torch.nn.Module):
    def forward(self,x):return x[:,:3]*.3,x[:,:3]*.2


class Rear(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.),requires_grad=False)
    def forward(self,z):return (z[0]+z[1])*self.weight


class Tests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.full((17,23,3),180,dtype=np.uint8)
        self.mask=np.zeros((17,23),bool);self.mask[5:12,7:16]=True
        self.ref=np.full_like(self.rgb,140)

    def test_hole_rgb_cannot_leak_to_encoder(self):
        other=self.rgb.copy();other[self.mask]=17
        a=canvas(self.rgb,self.mask,self.ref);b=canvas(other,self.mask,self.ref)
        torch.testing.assert_close(a['encoded'],b['encoded'])
        self.assertEqual(int(a['mask'][:,:,:17,a['left']:a['left']+23].sum()),int(self.mask.sum()))
        self.assertEqual(int(a['reference'].sum()),17*23)

    def test_ten_real_updates_and_frozen_network(self):
        batch=canvas(self.rgb,self.mask,self.ref);rear=Rear();before=rear.weight.detach().clone()
        _,_,trace=optimize_features(Front(),rear,batch,10,lr=.002)
        self.assertEqual(trace['optimizer_steps'],10)
        self.assertTrue(all(x>0 for x in trace['gradient_norms']))
        self.assertLess(trace['after']['known_l1'],trace['before']['known_l1'])
        self.assertLess(trace['after']['reference_l1'],trace['before']['reference_l1'])
        torch.testing.assert_close(rear.weight,before);self.assertIsNone(rear.weight.grad)

    def test_zero_steps_is_exact_forward(self):
        a,b,trace=optimize_features(Front(),Rear(),canvas(self.rgb,self.mask,self.ref),0)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(trace['optimizer_steps'],0)

    def test_padding_and_independent_context(self):
        b=canvas(self.rgb,self.mask)
        self.assertEqual(b['left'],0);self.assertEqual(tuple(b['image'].shape),(1,3,24,24))
        self.assertEqual(int(b['known'].sum()),17*23-int(self.mask.sum()))
        self.assertEqual(float(b['reference'].sum()),0)


if __name__=='__main__':unittest.main()
