"""
    Support functions for running the tests.
"""
import os
import numpy as np
from numpy.linalg import eig
from numpy.random import rand, randn, poisson


def init_config(config, default_config, name=None):
    """Initialise non-given config values with defaults"""
    if config is None:
        config = default_config
    else:
        for k in default_config.keys():
            if k not in config.keys():
                config[k] = default_config[k]
    if name and config['PRINT_CONFIG']:
        print('\n%s Config:' % name)
        for c in config.keys():
            print('%-20s : %-30s' % (c, config[c]))
    return config


def get_code_path():
    """Get base path where code is"""
    return os.path.abspath(os.path.dirname(__file__))


def initiate_classification_submission_file(classification):
    submission={}
    submission['meta']={}
    submission['meta']["use_camera"]=False
    submission['meta']["use_lidar"]=True
    submission['meta']["use_radar"]=False
    submission['meta']["use_map"]=False
    submission['meta']["use_external"]=False
    submission['meta']["classification"]=classification
    submission['results']={}
    submission['lidar_pcl'] = {}
    submission['run_time']={}
    return submission
