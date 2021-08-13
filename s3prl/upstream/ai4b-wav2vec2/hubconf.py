# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/ai4b-wav2vec2/hubconf.py ]
#   Synopsis     [ the wav2vec 2.0 torch hubconf ]
#   Author       [ AI4Bharat / Sumanth Doddapaneni]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch

# -------------#
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


from .expert import UpstreamExpert as _UpstreamExpert


def customized_upstream(*args, **kwargs):
    """
    To enable your customized pretrained model, you only need to implement
    downstream/example/expert.py and leave this file as is. This file is
    used to register the UpstreamExpert in downstream/example/expert.py
    The following is a brief introduction of the registration mechanism.

    The root hubconf.py collects all the entries registered in this file
    (callable variables without the underscore prefix) as a centralized
    upstream factory. One can pick up this upstream from the factory via

    1.
    from hubconf import customized_upstream
    model = customized_upstream(ckpt, model_config)

    2.
    model = torch.hub.load(
        'your_s3prl_path',
        'customized_upstream',
        ckpt,
        model_config,
        source='local',
    )

    Our run_downstream.py and downstream/runner.py follows the first usage
    """
    return _UpstreamExpert(*args, **kwargs)
