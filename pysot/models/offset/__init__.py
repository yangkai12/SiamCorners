from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.offset.offsets import offs_head

offs_head_ = {
         'offs_head': offs_head
        }

def off_mods_head(name, **kwargs):
    return offs_head_[name](**kwargs)

