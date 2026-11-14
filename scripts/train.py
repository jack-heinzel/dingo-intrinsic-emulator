"""dingo_train with the run-B reparameterization patch loaded. Same arguments as dingo_train.

The patch only acts when inference_parameters contain emulator.box_reparam.BOX_KEYS, so
this script also trains stock models (run A) unchanged.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import emulator.box_reparam  # noqa: F401,E402  (patches dingo's set_train_transforms)
from dingo.gw.training.train_pipeline import train_local  # noqa: E402

if __name__ == "__main__":
    train_local()
