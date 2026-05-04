from chromnitron.training.infra.origami.base.track import Track, TrackSum

class SequenceEmbeddingTrack(Track):

    def visualize(self, track_data):
        track = track_data.mean(axis = 1)

    def save(self, track_data, save_path):
        ''' Save track data '''
        raise NotImplementedError

def main():
    from base.storages import ZarrStorage, HiCNpzStorage

    enformer_storage = ZarrStorage('/gpfs/data/tsirigoslab/home/jt3545/data/datalake/sequence_embedding/human/enformer/s2_zarr/chrs.zarr', 'hg38', check_length = False)
    enformer_track = SequenceEmbeddingTrack(enformer_storage, resolution = 128)

    atac_storage_1 = ZarrStorage('/gpfs/data/tsirigoslab/home/jt3545/data/datalake/atac_seq/imr90/encode_control/s7_zarr/rep1/chrs.zarr', 'hg38')
    atac_storage_2 = ZarrStorage('/gpfs/data/tsirigoslab/home/jt3545/data/datalake/atac_seq/imr90/encode_control/s7_zarr/rep2/chrs.zarr', 'hg38')
    atac1_track = Track(atac_storage_1, resolution = 1)
    atac2_track = Track(atac_storage_2, resolution = 1)
    atac_track = TrackSum([atac_storage_1, atac_storage_2], resolution = 1)

    print(atac1_track.get('chr2', 2000000, 2000201))
    print(atac2_track.get('chr2', 2000000, 2000201))
    print(atac_track.get('chr2', 2000000, 2000201))

    microc_storage = ZarrStorage('/gpfs/data/tsirigoslab/home/jt3545/corigami2/data/hic/human/micro-c/hff/numpy/npy_by_chr/4DNFI9FVHJZQ/KR_zarr/hic_zstd_sm.zarr', 'hg38', check_length = False)

    microc_track = Track(microc_storage, resolution = 1000)

    hic_storage = HiCNpzStorage('/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/C.Origami-release/corigami_data/data/hg38/imr90/hic_matrix', 'hg38', check_length = False)


    hic_track = Track(hic_storage, resolution = 10000)

    import matplotlib.pyplot as plt
    import numpy as np
    hic_data = np.log(hic_track.get('chr2', 1000000, 2000000) + 1).copy().astype(np.float32)
    plt.imshow(hic_data)

    plt.savefig('hic.png')

    breakpoint()

if __name__ == '__main__':
    main()
