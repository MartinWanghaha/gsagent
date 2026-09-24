import cv2
import torch

from ..base_estimator import BaseEstimator


class OpenCVHomographyEstimator(BaseEstimator):
    default_conf = {
        "ransac_th": 3.0,
        "options": {"method": "ransac", "max_iters": 3000, "confidence": 0.999}, # 0.999
    }

    required_data_keys = ["m_kpts0", "m_kpts1"]

    def _init(self, conf):
        self.solver = {
            "ransac": cv2.RANSAC,
            "lmeds": cv2.LMEDS,
            "rho": cv2.RHO,
            "usac": cv2.USAC_DEFAULT,
            "usac_fast": cv2.USAC_FAST,
            "usac_accurate": cv2.USAC_ACCURATE,
            "usac_prosac": cv2.USAC_PROSAC,
            "usac_magsac": cv2.USAC_MAGSAC,
        }[conf.options.method]

    def _forward(self, data, scale=None):
        pts0, pts1 = data["m_kpts0"], data["m_kpts1"]

        try:
            if scale is not None:
                M, mask = cv2.findHomography(
                    pts0.cpu().numpy(),
                    pts1.cpu().numpy(),
                    self.solver,
                    ransacReprojThreshold = 3 * scale / 480,
                    confidence=0.99999,
                    # maxIters=self.conf.options.max_iters,                
                )
            else:
                M, mask = cv2.findHomography(
                    pts0.cpu().numpy(),
                    pts1.cpu().numpy(),
                    self.solver,
                    confidence=0.99999,
                    # maxIters=self.conf.options.max_iters,                
                )

            success = M is not None
        except cv2.error:
            success = False
        if not success:
            M = torch.eye(3, device=pts0.device, dtype=pts0.dtype)
            inl = torch.zeros_like(pts0[:, 0]).bool()
        else:
            M = torch.tensor(M).to(pts0)
            inl = torch.tensor(mask).bool().to(pts0.device)

        return {
            "success": success,
            "M_0to1": M,
            "inliers": inl,
        }
