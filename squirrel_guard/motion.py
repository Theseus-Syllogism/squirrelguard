"""Cheap motion gate: only run the detector when enough pixels changed inside the region of interest."""
from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    def __init__(self, min_area_frac: float = 0.002, threshold: int = 25, roi: list[list[int]] | None = None):
        self.min_area_frac = min_area_frac
        self.threshold = threshold
        self.roi = np.array(roi, np.int32) if roi else None
        self._mask: np.ndarray | None = None
        self._bg: np.ndarray | None = None

    def _roi_mask(self, shape) -> np.ndarray:
        if self._mask is None or self._mask.shape != shape:
            if self.roi is None:
                self._mask = np.full(shape, 255, np.uint8)
            else:
                self._mask = np.zeros(shape, np.uint8)
                cv2.fillPoly(self._mask, [self.roi], 255)
        return self._mask

    def __call__(self, frame: np.ndarray) -> bool:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        if self._bg is None:
            self._bg = gray.astype(np.float32)
            return False
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self._bg))
        # slow running average so lighting changes and swaying leaves fade into the background
        cv2.accumulateWeighted(gray, self._bg, 0.05)
        _, th = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        th = cv2.bitwise_and(th, self._roi_mask(th.shape))
        changed = cv2.countNonZero(th) / float(th.size)
        return changed >= self.min_area_frac
