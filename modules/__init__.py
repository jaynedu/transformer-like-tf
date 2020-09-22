# -*- coding: utf-8 -*-
# @Date    : 2020/7/17 11:26 下午
# @Author  : Du Jing
# @FileName: __init__.py
# ---- Description ----
#

from .attention import multi_head_attention
from .attention import scaled_dot_product_attention
from .attention import linear_attention_tayler
from .base import gelu
from .base import layer_normalization
from .layers import feed_forward
from .layers import position_encoding
from .layers import residual_connection
from .utils import label_smoothing
from .utils import noam_scheme
