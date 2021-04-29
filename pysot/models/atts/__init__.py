from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.atts.attention import atts_head

atts_head_ = {
         'atts_head': atts_head
        }

def att_mods_head(name, **kwargs):
    return atts_head_[name](**kwargs)

