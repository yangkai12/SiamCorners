from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.corners.cornerNet import corners_head

corners_head_ = {
         'corners_head': corners_head
        }

def corners_mods_head(name, **kwargs):
    return corners_head_[name](**kwargs)

