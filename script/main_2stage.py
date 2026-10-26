#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seeing Speech — stage 2: TAC fine-tuning / inference (training loop and arguments in trainer.py).

  python script/main_2stage.py --exp_name <exp> --base_pretrained <stage-1 ckpt> [...]
  python script/main_2stage.py --exp_name <exp> --base_pretrained <stage-1 ckpt> --test_only --start_epoch <N> [...]
  see run_bash/run_vocaset.sh, run_bash/run_tfhp.sh
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # package root

from trainer import run

if __name__ == "__main__":
    run(stage=2)
