# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python [conda env:ldm2]
#     language: python
#     name: conda-env-ldm2-py
# ---

# +
from streaming.base import MDSWriter
from tqdm import tqdm

from ldm.data.hpa23 import HPA

# + jupyter={"outputs_hidden": true}
group = "validation"
dataset = HPA(include_densenet_embedding="all", include_seq_embedding="prott5-swissprot", group=group)
dataset[0]
# -

{k:type(v) for k,v in dataset[0].items()}

# +
# Local or remote directory path to store the output compressed files.
# For remote directory, the output files are automatically upload to a remote cloud storage
# location.
out_root = 's3://ai-residency-stanford-subcellgenai/super_multiplex_cell/data/hpa23_rescaled_mds'

# A dictionary of input fields to an Encoder/Decoder type
columns = {'hpa_index': "int64",
 'bbox_label': "int64",
 'condition_caption': "str",
 'location_caption': "str",
 'matched_location_classes': "ndarray",
 'image': "ndarray",
 'ref-image': "ndarray",
 'mask': "ndarray",
 'bbox_coords': "ndarray",
 'densent_avg': "ndarray",
 'seq_embed': "ndarray"}

# Compression algorithm name
compression = 'zstd'

# Hash algorithm name
hashes = 'sha1', 'xxh64'

# Call `MDSWriter` to iterate through the input data and write into a shard `mds` file
with MDSWriter(out=f"{out_root}/{group}", columns=columns, compression=compression, hashes=hashes) as out:
    for sample in tqdm(dataset):
        out.write(sample)
        # break
# -


