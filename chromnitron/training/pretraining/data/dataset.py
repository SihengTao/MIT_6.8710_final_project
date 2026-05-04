import os
import sys
import torch
import zarr
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from chromnitron.training.infra.origami.base.storages import ZarrStorage
from chromnitron.training.infra.origami.base.partitions import GenomeRegion
from chromnitron.training.infra.origami.base.genome import Genome
from chromnitron.training.infra.origami.track.tracks import Track

import chromnitron.training.pretraining.data.transforms as transforms
from chromnitron.training.pretraining.utils import read_config


def get_dataset(args):
    """Get dataset from config file"""
    """use_merged_zarr, use_chunk_optimization, esm_single_task"""
    '''
    cell_type_features, (num_targets, targets) = get_feature_paths_merged_zarr(args)
    dataset = CelltypesESMDataset(args['input_seq'], cell_type_features, 
                        mode, args['excluded_region_file'],
                        window_size = args['window_size'],
                        chunk_size = args['chunk_size'],
                        sample_per_chunk = args['sample_per_chunk'],
                        ESM_npz_path = args['esm_npz_path'],
                        num_target_per_sample = args['num_target_per_sample'],
                        args = args)
    '''
    dataset = CelltypesESMDataset(**args)
    return dataset

def get_feature_paths_merged_zarr(sample_sheet, merged_zarr_path, cell_types, write_targets_to_file = False, target_csv_save_output_path = None):
    #sample_sheet = args['sample_sheet']
    #merged_zarr_path = args['merged_zarr_path']
    #cell_types = args['cell_types']

    sample_df = pd.read_csv(sample_sheet)
    target_df_dict = {}
    target_zarr_dict = {}
    for file in os.listdir(merged_zarr_path):
        if file.endswith(".csv"):
            target_df_dict[file.split('.csv')[0]] = pd.read_csv(os.path.join(merged_zarr_path, file))
            zarr_path = os.path.join(merged_zarr_path, file.split('.csv')[0] + '.zarr')
            target_zarr_dict[file.split('.csv')[0]] = zarr_path
            if not os.path.exists(zarr_path):
                raise ValueError('Zarr file does not exist: {}'.format(zarr_path))
    target_df = pd.concat(target_df_dict.values(), ignore_index=True)


    all_targets = sorted(target_df[target_df['Target'] != 'ATAC_seq']['Target'].unique().tolist())
    filtered_targets = all_targets

    # Check if cell types are in the keys
    for cell_type in cell_types:
        if cell_type not in target_df_dict:
            raise ValueError(f'{cell_type} not in {args["merged_zarr_path"]}')

    num_targets = len(filtered_targets)
    targets = filtered_targets

    cell_type_feature_dict = {}
    for cell_type in cell_types:
        atac_path = get_atac_path(sample_df, cell_type) # This is from original sample sheet
        target_dict = {}
        for target in filtered_targets:
            target_dict[target] = get_chip_index(target_df_dict[cell_type], target)
        cell_type_feature_dict[cell_type] = [atac_path, target_dict, target_df_dict[cell_type], target_zarr_dict[cell_type]]

    # Export targets to csv
    if write_targets_to_file:
        targets_path = target_csv_save_output_path
        os.makedirs(targets_path, exist_ok=True)
        if not os.path.exists(f'{targets_path}/targets.csv'):
            with open(f'{targets_path}/targets.csv', 'w') as f:
                for target in filtered_targets:
                    f.write(target + '\n')
        else:
            #raise ValueError(f'{targets_path}/targets.csv already exists')
            print(f'{targets_path}/targets.csv already exists')

    return cell_type_feature_dict, (num_targets, targets)


def get_chip_index(sample_df, target):
    idx_list = sample_df[sample_df['Target'] == target].index.tolist()
    if len(idx_list) == 0:
        return np.nan
    else:
        return idx_list[0]


def get_atac_path(sample_df, cell_type):
    atac_path = sample_df[np.logical_and(sample_df['CellType'] == cell_type, sample_df['Target'] == 'ATAC_seq')]['Path'].values[0]
    return atac_path

# Cell types dataset

class CelltypesDataset(Dataset):
    def __init__(self, input_seq_path,
                 mode, excluded_region_file, 
                 cell_type_dict = None,
                 val_chrs = ['chr10'],
                 test_chrs = ['chr20'],
                 assembly = 'hg38',
                 window_size = 4000, 
                 sample_size = 8000, step_size = 4000, chr_margin = 500000,
                 verbose = False,
                 *args, **kwargs):
        # Initialize the genome dataset for each cell type
        genomes = []
        metadata = {}
        for i, cell_dict in enumerate(cell_type_dict.items()):
            cell_type, feature_paths = cell_dict
            input_features_path, target_features_dict = feature_paths
            targets = list(target_features_dict.keys())
            target_features_paths = list(target_features_dict.values())
            metadata[i] = {'celltype' : cell_type,
                            'target' : targets}
            genome = GenomeDataset(input_seq_path, 
                            input_features_path, target_features_paths, 
                            mode, excluded_region_file,
                            val_chrs, test_chrs,
                            assembly, window_size, sample_size, step_size, chr_margin, verbose, i)
            genomes.append(genome)

        self.genomes = genomes
        self.metadata = metadata

    def subsample_region(self, all_counts, seed = 0):
        ''' Subsample regions '''
        count = all_counts // len(self.genomes)
        for genome in self.genomes:
            genome.subsample_region(count, seed)

    def __getitem__(self, idx):
        # Get the genome and the index of the sample within the genome
        genome_idx = 0
        while idx >= len(self.genomes[genome_idx]):
            idx -= len(self.genomes[genome_idx])
            genome_idx += 1
        genome = self.genomes[genome_idx]
        sample_idx = idx
        return genome[sample_idx]
    
    def __len__(self):
        return sum([len(genome) for genome in self.genomes])


class CelltypesMergedZarrDataset(CelltypesDataset):
    def __init__(self, input_seq_path,
                 mode, excluded_region_file, 
                 cell_type_dict = None,
                 val_chrs = ['chr10'],
                 test_chrs = ['chr20'],
                 assembly = 'hg38',
                 window_size = 4000, 
                 sample_size = 8000, step_size = 4000, chr_margin = 500000,
                 verbose = False,
                 *args, **kwargs):
        # Initialize the genome dataset for each cell type
        genomes = []
        metadata = {}
        for i, cell_dict in enumerate(cell_type_dict.items()):
            cell_type, feature_paths = cell_dict
            input_features_path, target_features_dict, meta_df, zarr_path = feature_paths
            targets = list(target_features_dict.keys())
            target_features_indices = list(target_features_dict.values())
            metadata[i] = {'celltype' : cell_type,
                            'target' : targets}
            genome = GenomeMergedZarrDataset(input_seq_path, 
                            input_features_path, target_features_indices, 
                            meta_df, zarr_path,
                            mode, excluded_region_file,
                            val_chrs, test_chrs,
                            assembly, window_size, sample_size, step_size, chr_margin, verbose, i)
            genomes.append(genome)

        self.genomes = genomes
        self.metadata = metadata

class CelltypesESMDataset(CelltypesMergedZarrDataset):
    def __init__(self, input_seq_path,
                 mode, excluded_region_file,
                 val_chrs = ['chr10'],
                 test_chrs = ['chr20'],
                 assembly = 'hg38',
                 chunk_size = 100000,
                 sample_per_chunk = 4,
                 window_size = 8192,
                 sample_sheet = None, merged_zarr_path = None, cell_types = None, write_targets_to_file = False, target_csv_save_output_path = None,
                 verbose = False,
                 esm_zarr_path = None,
                 num_target_per_sample = None,
                 subsample_CAPs = False, 
                 subsample_CAPs_list_path = None,
                 atac_log1p = True,
                 args = None):
        # Initialize the genome dataset for each cell type
        # Check chunk have enough samples (overlapping 2)
        assert window_size * sample_per_chunk <= chunk_size
        cell_type_dict, (num_targets, targets) = get_feature_paths_merged_zarr(sample_sheet, merged_zarr_path, cell_types, write_targets_to_file, target_csv_save_output_path)
        self.num_targets = num_targets
        self.targets = targets
        self.mode = mode
        first_target_features_dict = list(cell_type_dict.values())[0][1]
        targets = list(first_target_features_dict.keys())
        # Check if subsample caps
        if subsample_CAPs:
            self.ESM_mat, self.ESM_mask = self.load_selected_ESM(targets, esm_zarr_path, subsample_CAPs_list_path)
        else:
            self.ESM_mat, self.ESM_mask = self.load_ESM(targets, esm_zarr_path)
        genomes = []
        metadata = {}
        for i, cell_dict in enumerate(cell_type_dict.items()):
            cell_type, feature_paths = cell_dict
            input_features_path, target_features_dict, meta_df, zarr_path = feature_paths
            targets = list(target_features_dict.keys())
            target_features_indices = list(target_features_dict.values())
            metadata[i] = {'celltype' : cell_type,
                            'target' : targets}
            genome = GenomeESMDataset(input_seq_path, 
                            input_features_path, target_features_indices, 
                            meta_df, zarr_path,
                            mode, excluded_region_file,
                            val_chrs, test_chrs,
                            assembly, 
                            chunk_size, sample_per_chunk, window_size, verbose, i,
                            num_target_per_sample = num_target_per_sample,
                            ESM_mask = self.ESM_mask,
                            atac_log1p = atac_log1p)
            genomes.append(genome)

        self.genomes = genomes
        self.metadata = metadata
        self.num_target_per_sample = num_target_per_sample

    def load_ESM(self, target_list, ESM_zarr_path):
        esm_zarr = zarr.open(ESM_zarr_path, mode = 'r')
        ESM_mat = esm_zarr['ESM']
        ESM_mask = esm_zarr['mask'][:]
        target_list_load = esm_zarr['target_list'][:]
        # Compare target_list
        compare = [target == target_list_load[i] for i, target in enumerate(target_list)]
        assert np.all(compare)
        return ESM_mat, ESM_mask

    def load_selected_ESM(self, target_list, ESM_zarr_path, subsample_CAPs_list_path):
        ESM_mat, ESM_mask = self.load_ESM(target_list, ESM_zarr_path)
        selected_target_list = pd.read_csv(subsample_CAPs_list_path, header = None).values.flatten().tolist()
        # Adjust ESM_mask
        ESM_selection_mask = [target in selected_target_list for target in target_list]
        ESM_mask = ESM_mask * ESM_selection_mask
        print(f'CAUTION!: Subsampling {sum(ESM_selection_mask)} CAPs out of {len(ESM_mask)} for mode {self.mode}')
        selected_target = [target_list[i] for i in range(len(ESM_mask)) if ESM_mask[i]]
        print(selected_target)
        return ESM_mat, ESM_mask

    def __getitem__(self, idx):
        # Get the genome and the index of the sample within the genome
        genome_idx = 0
        while idx >= len(self.genomes[genome_idx]):
            idx -= len(self.genomes[genome_idx])
            genome_idx += 1
        genome = self.genomes[genome_idx]
        sample_idx = idx
        sample = genome[sample_idx]
        seq, input_features, target_features, metadata, selected_target_idx = sample
        ESM_array = torch.tensor(self.ESM_mat[selected_target_idx, :, :])
        start, end, chrom, region_id, target_mask, metadata_key = metadata
        pseudo_batch_size = start.shape[0]
        target_idx = np.repeat(selected_target_idx.reshape(1, -1), pseudo_batch_size, axis = 0)

        metadata = start, end, chrom, region_id, target_mask, metadata_key, target_idx
        sample_with_ESM = (seq, input_features, target_features, ESM_array, metadata)
        return sample_with_ESM
    
    def __len__(self):
        return sum([len(genome) for genome in self.genomes])


# Genome dataset collection

class GenomeDataset(Dataset):
    def __init__(self, input_seq_path, input_features_path, target_features_path,  # Features
                 mode, excluded_region_file, 
                 val_chrs = ['chr10'],
                 test_chrs = ['chr20'],
                 assembly = 'hg38',
                 window_size = 4000, 
                 sample_size = 8000, step_size = 4000, chr_margin = 500000,
                 verbose = False, metadata_key = None):
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f'Invalid mode: {mode}')
        # Print target features
        if verbose:
            print(f'Loading {mode} data...')
            print(f'Loading input seq from {input_seq_path}')
            print(f'Loading input features from {input_features_path}')
            print(f'Loading target features from {target_features_path}')
        self.aug_bool = mode == 'train'
        self.metadata_key = metadata_key
        self.input_seq_path = input_seq_path
        self.input_features_path = input_features_path
        self.target_features_path = target_features_path
        self.target_mask = [target != '' for target in target_features_path]
        self.mode = mode
        self.excluded_region_file = excluded_region_file
        self.val_chrs = val_chrs
        self.test_chrs = test_chrs
        self.assembly = assembly
        self.window_size = window_size
        self.sample_size = sample_size
        self.step_size = step_size
        self.chr_margin = chr_margin
        self.verbose = verbose

        # Initalize region
        self.all_region = self.get_region(sample_size, step_size, chr_margin, excluded_region_file, assembly, val_chrs, test_chrs, mode, verbose)
        self.region = self.all_region # For subsampling
        # Initialize data
        self.data = self.load_data(input_seq_path, input_features_path, target_features_path, assembly, verbose)

    def subsample_region(self, count, seed = 0):
        ''' Subsample regions '''
        sorted_sample_idx = sorted(np.random.RandomState(seed).choice(len(self.all_region), count, replace = False))
        import copy
        self.region = copy.deepcopy(self.all_region)
        self.region.loci = self.region.loci[sorted_sample_idx]

    def get_region(self, sample_size, step_size, chr_margin, excluded_region_file, assembly, val_chrs, test_chrs,  mode, verbose):
        ''' Get region chrs based on train/val/test '''
        chr_list = list(Genome(assembly).chr_lengths.keys())
        train_chrs = [x for x in chr_list if x not in val_chrs + test_chrs]
        if mode == 'train':
            excluded_chrs = val_chrs + test_chrs
        elif mode == 'val':
            excluded_chrs = train_chrs + test_chrs
        elif mode == 'test':
            excluded_chrs = train_chrs + val_chrs
        excluded_chrs += ['chrX', 'chrY', 'chrM']
        return GenomeRegion(sample_size, step_size, chr_margin, excluded_region_file, assembly, excluded_chrs, verbose)

    def load_data(self, input_seq_path, input_features_path, target_features_path, assembly, verbose):
        ''' Load data from input files '''
        data_dict = {'seq' : None,
                     'input_features' : None,
                     'target_features' : None}
        data_dict['seq'] = Track(ZarrStorage(input_seq_path, assembly))
        data_dict['input_features'] = self.load_storage_with_paths(assembly, 'input_features', input_features_path)
        data_dict['target_features'] = self.load_storage_with_paths(assembly, 'target_features', target_features_path)
        return data_dict

    def load_storage_with_paths(self, assembly, feature_name, paths):
        if isinstance(paths, str):
            if paths == '':
                return None
            return Track(ZarrStorage(paths, assembly))
        elif isinstance(paths, list):
            return [self.load_storage_with_paths(assembly, feature_name, path) for path in paths]
        else:
            raise ValueError(f'Invalid paths: {paths}')

    def get_features(self, features, chrom, start, end):
        if features is None:
            return np.nan * np.ones((end - start))
        if isinstance(features, Track):
            return np.nan_to_num(features.get(chrom, start, end).astype(np.float32))
        elif isinstance(features, list):
            return [self.get_features(feature, chrom, start, end) for feature in features]
        else:
            raise ValueError(f'Invalid features: {features}')

    def get_features_target(self, features, chrom, start, end):
        return self.get_features(features, chrom, start, end)

    def __len__(self):
        return len(self.region)

    def __getitem__(self, idx):
        # Get sampled region
        chrom, start_str, end_str, region_id = self.region[idx]
        start, end = int(start_str), int(end_str)
        # Get region subset and locus augmentation
        start, end = transforms.subsample_locus(self.window_size, start, end, self.aug_bool)
        # Get features
        seq = transforms.to_onehot(self.data['seq'].get(chrom, start, end))
        input_features = self.get_features(self.data['input_features'], chrom, start, end)
        target_features = self.get_features_target(self.data['target_features'], chrom, start, end)
        # Value augmentations
        if self.aug_bool:
            seq, input_features, target_features = transforms.reverse_features(seq, input_features, target_features)
            seq, input_features, target_features = transforms.add_gaussian_noise(seq, input_features, target_features)
        # log(1+x) transform features
        input_features, target_features = transforms.log1p_features(input_features, target_features)
        return seq, input_features, target_features, (start, end, chrom, region_id, self.target_mask, self.metadata_key)



class GenomeMergedZarrDataset(GenomeDataset):
    def __init__(self, input_seq_path, input_features_path, target_features_path, target_meta_data, target_zarr_path, # Features
                 mode, excluded_region_file, 
                 val_chrs = ['chr10'],
                 test_chrs = ['chr20'],
                 assembly = 'hg38',
                 window_size = 4000, 
                 sample_size = 8000, step_size = 4000, chr_margin = 500000,
                 verbose = False, metadata_key = None):
        self.target_meta_data = target_meta_data
        self.target_zarr_path = target_zarr_path
        super().__init__(input_seq_path, input_features_path, target_features_path, mode, excluded_region_file, val_chrs, test_chrs, assembly, window_size, sample_size, step_size, chr_margin, verbose, metadata_key)
        self.target_mask = [not np.isnan(target) for target in target_features_path]

    def load_data(self, input_seq_path, input_features_path, target_features_path, assembly, verbose):
        ''' Load data from input files '''
        data_dict = {'seq' : None,
                     'input_features' : None,
                     'target_features' : None}
        data_dict['seq'] = Track(ZarrStorage(input_seq_path, assembly))
        data_dict['input_features'] = self.load_storage_with_paths(assembly, 'input_features', input_features_path)
        data_dict['target_features'] = Track(ZarrStorage(self.target_zarr_path, assembly))
        return data_dict

    def get_features_target(self, _, chrom, start, end):
        stored_features = self.data['target_features'].get(chrom, start, end)
        # Rearrange features according to target_meta_data
        features = []
        for feature_idx in self.target_features_path:
            if np.isnan(feature_idx):
                features.append(np.nan * np.ones((end - start)))
            else:
                features.append(stored_features[:, int(feature_idx)])
        return features

class GenomeMergedZarrChunkOptimizedDataset(GenomeMergedZarrDataset):
    def __init__(self, input_seq_path, input_features_path, target_features_indices, target_meta_data, target_zarr_path, # Features
                 mode, excluded_region_file, 
                 val_chrs = ['chr10'],
                 test_chrs = ['chr20'],
                 assembly = 'hg38',
                 chunk_size = 100000,
                 sample_per_chunk = 4,
                 window_size = 8192, 
                 verbose = False, metadata_key = None,
                 atac_log1p = True,
                 *args, **kwargs):
        self.target_meta_data = target_meta_data
        self.target_zarr_path = target_zarr_path
        # For sampling real samples
        self.chunk_size = chunk_size
        self.sample_per_chunk = sample_per_chunk
        self.real_window_size = window_size
        # For chunk sampling
        window_size = chunk_size
        sample_size = chunk_size
        step_size = chunk_size
        chr_margin = 0
        super().__init__(input_seq_path, 
                         input_features_path, target_features_indices, 
                         target_meta_data, target_zarr_path,
                         mode, excluded_region_file,
                         val_chrs, test_chrs,
                         assembly, window_size, sample_size, step_size, chr_margin, verbose, metadata_key)
        self.target_mask = np.array([not np.isnan(target) for target in target_features_indices])
        self.atac_log1p = atac_log1p

    #@profile
    def get_data(self, chrom, start, end):
        # Get features
        seq = transforms.to_onehot(self.data['seq'].get(chrom, start, end))
        input_features = self.get_features(self.data['input_features'], chrom, start, end)
        target_features = self.get_features_target(self.data['target_features'], chrom, start, end)
        additional_features = None
        return seq, input_features, target_features, additional_features

    def __getitem__(self, idx):
        # Get sampled region
        chrom, start_str, end_str, region_id = self.region[idx]
        start, end = int(start_str), int(end_str)
        seq, input_features, target_features, other_features= self.get_data(chrom, start, end)
        # Value augmentations
        if self.aug_bool:
            seq, input_features, target_features = transforms.reverse_features(seq, input_features, target_features)
            seq, input_features, target_features = transforms.add_gaussian_noise(seq, input_features, target_features)
        # log(1+x) transform features
        if self.atac_log1p:
            input_features, target_features = transforms.log1p_features(input_features, target_features)
        else:
            target_features = transforms.log1p_clip_negative(target_features)
        # Split chunk into samples
        region_starts, region_ends = self.get_half_overlapping_regions(0, seq.shape[0], self.sample_per_chunk)
        seq_batch = []
        input_features_batch = []
        target_features_batch = []
        for region_start, region_end in zip(region_starts, region_ends):
            # Get region subset and locus augmentation
            start_aug, end_aug = transforms.subsample_locus(self.real_window_size, region_start, region_end, self.aug_bool)
            seq_batch.append(seq[start_aug:end_aug])
            input_features_batch.append(input_features[start_aug:end_aug])
            target_features_batch.append([feature[start_aug:end_aug] for feature in target_features])
        seq_batch = np.stack(seq_batch)
        input_features_batch = np.stack(input_features_batch)
        target_features_batch = np.stack(target_features_batch)
        target_mask_batch = np.stack([self.target_mask for _ in range(self.sample_per_chunk)])
        # Other loci information
        start_batch = np.array([start] * self.sample_per_chunk)
        end_batch = np.array([end] * self.sample_per_chunk)
        chrom_batch = np.array([int(chrom.split('chr')[1])] * self.sample_per_chunk)
        region_id_batch = np.array([int(region_id.split('region_')[1])] * self.sample_per_chunk)
        metadata_key_batch = np.array([self.metadata_key] * self.sample_per_chunk)

        return_batch = self.organize_return_features(seq_batch, input_features_batch, target_features_batch, start_batch, end_batch, chrom_batch, region_id_batch, target_mask_batch, metadata_key_batch, other_features)
        return return_batch

    def organize_return_features(self, seq_batch, input_features_batch, target_features_batch, start_batch, end_batch, chrom_batch, region_id_batch, target_mask_batch, metadata_key_batch, other_features):
        info_batch = (start_batch, end_batch, chrom_batch, region_id_batch, target_mask_batch, metadata_key_batch)
        return seq_batch, input_features_batch, target_features_batch, info_batch

    def get_half_overlapping_regions(self, start, end, sample_per_chunk):
        step_size = (end - start) // (sample_per_chunk + 1)
        region_starts = np.arange(start, end - 1, step_size)[:-1]
        region_ends = region_starts + step_size * 2
        return region_starts, region_ends

class GenomeESMDataset(GenomeMergedZarrChunkOptimizedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get num_target_per_sample parameter
        self.num_target_per_sample = kwargs['num_target_per_sample']
        self.ESM_mask = kwargs['ESM_mask']

    def get_features_target(self, _, chrom, start, end):
        genome_zarr = self.data['target_features'].storage.chrs
        #stored_features = self.data['target_features'].get(chrom, start, end)
        # Rearrange features according to target_meta_data
        features = []

        # Sample from non-nan target and ESM
        target_mask = torch.tensor(~np.isnan(self.target_features_path))
        mask = target_mask & self.ESM_mask

        # Sample from non-nan target and ESM during training
        if self.mode == 'train':
            target_samples = min(self.num_target_per_sample, mask.sum())
            selected_target_idx = np.random.choice(np.where(mask)[0], target_samples, replace = False)
        else:
            selected_target_idx = np.where(mask)[0]
        zarr_idx = np.array(self.target_features_path)[selected_target_idx].astype(int)
        features_mat = self.data['target_features'].get(chrom, start, end)
        features_mat = features_mat[:, zarr_idx].copy()

        features = [features_mat[:, i] for i in range(features_mat.shape[1])]
        return features, selected_target_idx

    def get_data(self, chrom, start, end):
        # Get features
        seq = transforms.to_onehot(self.data['seq'].get(chrom, start, end))
        input_features = self.get_features(self.data['input_features'], chrom, start, end)
        target_features, selected_target_idx = self.get_features_target(self.data['target_features'], chrom, start, end)
        return seq, input_features, target_features, selected_target_idx

    def organize_return_features(self, seq_batch, input_features_batch, target_features_batch, start_batch, end_batch, chrom_batch, region_id_batch, target_mask_batch, metadata_key_batch, selected_target_idx):
        info_batch = (start_batch, end_batch, chrom_batch, region_id_batch, target_mask_batch, metadata_key_batch)
        return seq_batch, input_features_batch, target_features_batch, info_batch, selected_target_idx


def test_esm():
    from tqdm import tqdm
    from chromnitron.training.pretraining.config.default import get_feature_paths, load_config

    args = read_config('data_config.yaml')

    args = args['dataset']['train']['args']
    
    dataset = get_dataset(args)

    args = load_config()
    rank = 0
    from model.chromnitron_models import get_model
    model = get_model(args).to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    dataset[0]
    '''
    for i in tqdm(range(len(dataset))):
        dataset[i]
        if i == 10:
            break
    exit()
    '''
    from torch.utils.data import DataLoader
    #data_loader = DataLoader(dataset, batch_size = 2, shuffle = True, num_workers = 40, pin_memory = False)
    data_loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 4, pin_memory = False)
    for seq, input_features, target_features, esm_embeddings, loc_info in tqdm(data_loader):
        seq = seq.to(rank, non_blocking=True)
        input_features = input_features.to(rank, non_blocking=True)
        target_features = target_features.to(rank, non_blocking=True)
        esm_embeddings = esm_embeddings.to(rank, non_blocking=True)
        esm_embeddings = esm_embeddings.float().transpose(-1, -2)

        batch_size, mini_bs, seq_len, seq_dim = seq.shape
        seq = seq.view(batch_size * mini_bs, seq_len, seq_dim)
        input_features = input_features.view(batch_size * mini_bs, -1)
        batch_size, mini_bs, num_target, seq_len = target_features.shape
        target_features = target_features.view(batch_size * mini_bs, num_target, seq_len)
        seq = seq.transpose(1, 2).float()
        input_features = input_features.unsqueeze(2).transpose(1, 2).float()

        inputs = (seq, input_features)
        labels = target_features

        # Check number of esm_embeddings
        num_targets = esm_embeddings.shape[1]
        if num_targets > 4:
            print(f'Chunking esm_embeddings with {num_targets} targets')
            # Chunk the esm_embeddings
            esm_chunks = esm_embeddings.chunk(num_targets // 4 + 1, dim = 1)
            preds_list, confidence_list = [], []
            for esm_chunk in esm_chunks:
                preds, confidence = model(inputs, esm_chunk)
                preds_list.append(preds.detach().cpu())
                confidence_list.append(confidence.detach().cpu())
            preds = torch.cat(preds_list, dim = 1)
            confidence = torch.cat(confidence_list, dim = 1)
        else:
            preds, confidence = model(inputs, esm_embeddings)

        breakpoint()
        # Only select the target locations to compute loss

        criterion = torch.nn.MSELoss()
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        start, end, chr_name, region_id, target_mask, metadata_key, target_idx = loc_info

        start = start.view(batch_size * mini_bs, 1)
        end = end.view(batch_size * mini_bs, 1)
        chr_name = chr_name.view(batch_size * mini_bs, 1)
        region_id = region_id.view(batch_size * mini_bs, 1)
        metadata_key = metadata_key.view(batch_size * mini_bs, 1)
        pass

if __name__ == '__main__':
    test_esm()