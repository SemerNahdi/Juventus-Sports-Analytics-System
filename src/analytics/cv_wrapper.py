"""
Robust OpenCV import wrapper.
Handles environment inconsistencies where cv2 might be partially loaded or shadowed.
"""

import sys
import os
import logging
import importlib

logger = logging.getLogger(__name__)

cv2 = None

# Try various import strategies for OpenCV
strategies = [
    lambda: importlib.import_module('cv2'),
    lambda: importlib.import_module('cv2').cv2,
    lambda: importlib.import_module('cv2.cv2'),
]

for strategy in strategies:
    try:
        candidate = strategy()
        if hasattr(candidate, 'VideoCapture'):
            cv2 = candidate
            break
    except (ImportError, AttributeError, TypeError):
        continue

# Fallback if no strategy worked but cv2 is somewhat available
if cv2 is None:
    try:
        import cv2 as _cv2
        cv2 = _cv2
    except ImportError:
        cv2 = None

# Final fix-up for missing attributes
if cv2 is not None:
    # Try to inject missing attributes from submodules if possible
    if not hasattr(cv2, 'VideoCapture'):
        # Try to find where VideoCapture is hiding
        for sub in ['cv2', 'videoio', 'highgui']:
            try:
                mod = importlib.import_module(f'cv2.{sub}')
                for attr in dir(mod):
                    if not hasattr(cv2, attr):
                        setattr(cv2, attr, getattr(mod, attr))
                if hasattr(cv2, 'VideoCapture'):
                    break
            except (ImportError, AttributeError):
                continue

    # Ensure some basics exist to prevent immediate crashes
    if not hasattr(cv2, 'CAP_PROP_FPS'): cv2.CAP_PROP_FPS = 5
    if not hasattr(cv2, 'CAP_PROP_FRAME_COUNT'): cv2.CAP_PROP_FRAME_COUNT = 7
    if not hasattr(cv2, 'CAP_PROP_FRAME_WIDTH'): cv2.CAP_PROP_FRAME_WIDTH = 3
    if not hasattr(cv2, 'CAP_PROP_FRAME_HEIGHT'): cv2.CAP_PROP_FRAME_HEIGHT = 4
    if not hasattr(cv2, 'CAP_PROP_POS_FRAMES'): cv2.CAP_PROP_POS_FRAMES = 1
    if not hasattr(cv2, 'INTER_AREA'): cv2.INTER_AREA = 3
    if not hasattr(cv2, 'LINE_AA'): cv2.LINE_AA = 16
    if not hasattr(cv2, 'FONT_HERSHEY_SIMPLEX'): cv2.FONT_HERSHEY_SIMPLEX = 0

if cv2 is None or not hasattr(cv2, 'VideoCapture'):
    msg = "CRITICAL: OpenCV VideoCapture could not be resolved."
    print(f"\n[CV2_WRAPPER] {msg}")
    logger.error(msg)
    raise RuntimeError(msg)
else:
    # Optional debug print to confirm resolution
    # print(f"[CV2_WRAPPER] Resolved cv2 with VideoCapture from: {getattr(cv2, '__file__', 'unknown')}")
    pass


# Export
__all__ = ['cv2']
