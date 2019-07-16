"""This script evaluates a given matchnet model (including feature net and metric
   net) on a given ubc test set.
"""

import sys

sys.path.insert(0, 'your-caffe-python-path')
import caffe

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import leveldb, numpy as np, skimage
from caffe.proto import caffe_pb2
from caffe.io import *


def ParseArgs():
    """Parse input arguments.
    """
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset_dir', help='Patch to the directory of .bmp files.')
    parser.add_argument('patch_pairs_info',
                        help=('patch pairs in text format. Patches should be in test_db. ' +
                              'Following the original UBC dataset format, each line has ' +
                              '6 integers separated by space, 3 for each patch. The three ' +
                              'numbers for each point are: patch_id, 3D point_id, 0. ' +
                              'Two patches match if their point_id match.'))
    parser.add_argument('output_pairs_db', help='Path to output patch pairs leveldb database.')
    args = parser.parse_args()
    return args


def GetPatchImage(patch_id, container_dir):
    """Returns a 64 x 64 patch with the given patch_id. Catch container images to
       reduce loading from disk.
    """
    # Define constants. Each container image is of size 1024x1024. It packs at
    # most 16 rows and 16 columns of 64x64 patches, arranged from left to right,
    # top to bottom.
    PATCHES_PER_IMAGE = 16 * 16
    PATCHES_PER_ROW = 16
    PATCH_SIZE = 64

    # Calculate the container index, the row and column index for the given
    # patch.
    container_idx, container_offset = divmod(patch_id, PATCHES_PER_IMAGE)
    row_idx, col_idx = divmod(container_offset, PATCHES_PER_ROW)

    # Read the container image if it is not cached.
    if GetPatchImage.cached_container_idx != container_idx:
        GetPatchImage.cached_container_idx = container_idx
        GetPatchImage.cached_container_img = \
            skimage.img_as_ubyte(skimage.io.imread('%s/patch%07d.bmp' % \
                                                   (container_dir, container_idx), as_grey=True))

    # Extract the patch from the image and return.
    patch_image = GetPatchImage.cached_container_img[ \
                  PATCH_SIZE * row_idx:PATCH_SIZE * (row_idx + 1), \
                  PATCH_SIZE * col_idx:PATCH_SIZE * (col_idx + 1)]
    return patch_image


# Static variables initialization for GetPatchImage.
GetPatchImage.cached_container_idx = None
GetPatchImage.cached_container_img = None


def main():
    args = ParseArgs()

    total = sum(1 for line in open(args.patch_pairs_info))

    pair_info = np.loadtxt(args.patch_pairs_info)

    # Create the output patch pairs leveldb database, fail if exists.
    pairs_db = leveldb.LevelDB(args.output_pairs_db,
                               create_if_missing=True,
                               error_if_exists=True)

    batch = leveldb.WriteBatch()
    patch_pair = np.zeros((2, 64, 64), dtype=np.uint8)  # data
    processed = 0

    for i in range(pair_info.shape[0]):

        datum = caffe_pb2.Datum()
        datum.channels, datum.height, datum.width = (2, 64, 64)

        patch_pair[0] = GetPatchImage(int(pair_info[i][0]), args.dataset_dir)
        patch_pair[1] = GetPatchImage(int(pair_info[i][1]), args.dataset_dir)

        datum.data = patch_pair.tostring()  # data

        datum.label = int(pair_info[i][2])  # label

        batch.Put(str(processed), datum.SerializeToString())
        processed += 1
        if processed % 500 == 0:

            # Write the current batch.
            pairs_db.Write(batch, sync=True)

            # Verify the last written record.
            d = caffe_pb2.Datum()
            d.ParseFromString(pairs_db.Get(str(processed - 1)))
            assert (d.data == datum.data)

            # Start a new batch
            batch = leveldb.WriteBatch()
    pairs_db.Write(batch, sync=True)


if __name__ == '__main__':
    main()