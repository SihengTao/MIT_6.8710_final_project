import torch
import numpy as np
import einops


def main():
    pass


def read_config(setup_dict_path: str) -> dict:
    with open(setup_dict_path, 'r') as file:
        import yaml
        # TODO include yaml as external resource
        setup_dict = yaml.safe_load(file)
    return setup_dict


def bin_track(input_track, bin_size=500, bin_dim=-1):
    # Cut off the end of the track if not divisible
    input_track = input_track[..., :input_track.shape[bin_dim] // bin_size * bin_size]
    reduced_track = einops.reduce(input_track, '... (i b) -> ... i', 'mean', b=bin_size)
    return reduced_track


def calculate_corr_top(track1, track2, corr_type, top_percentile=0.25):
    track1 = track1.flatten()
    track2 = track2.flatten()
    track1_top_idx = np.argsort(track1)[-int(len(track1) * top_percentile):]
    track2_top_idx = np.argsort(track2)[-int(len(track2) * top_percentile):]
    union_idx = np.union1d(track1_top_idx, track2_top_idx)
    track1_top = track1[union_idx]
    track2_top = track2[union_idx]
    try:
        corr = calculate_corr(track1_top, track2_top, corr_type)
    except:
        print('Error in calculating correlation')
        corr = 0
    return corr


def calculate_corr(track1, track2, corr_type):
    track1 = track1.flatten()
    track2 = track2.flatten()
    import scipy
    # Calculate the correlation between two tracks
    if corr_type == 'pearson':
        corr = scipy.stats.pearsonr(track1, track2)[0]
    elif corr_type == 'spearman':
        corr = scipy.stats.spearmanr(track1, track2)[0]
    return corr


if __name__ == '__main__':
    main()
