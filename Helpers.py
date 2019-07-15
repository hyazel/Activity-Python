#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:11:30 2019

@author: laurent.droguet
"""

import os
import glob

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

