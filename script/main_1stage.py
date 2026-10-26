#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seeing Speech — stage 1: SAM pre-training (training loop and arguments in trainer.py).

  python script/main_1stage.py --exp_name <exp> --axis_ae_ckpt <axis_ae_best.pth> [...]
  see run_bash/run_vocaset.sh, run_bash/run_tfhp.sh
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # package root

from trainer import run

if __name__ == "__main__":
    run(stage=1)
