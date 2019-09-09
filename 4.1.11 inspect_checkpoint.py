#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:27:09 2019
将生成的模型里的内容打印出来

@author: zhuyangze
"""

# import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

savedir = "log/"
print_tensors_in_checkpoint_file(savedir + "linermodel.cpkt", None, True)