from chromnitron.training.infra.origami.base.partition import Partition
import numpy as np

class CustomRegion(Partition):
    ''' Partition with custom input '''

    def load(self, path):
        ''' Load loci data 
        return: loci, an iterable object
        '''
        import pandas as pd
        return pd.read_csv(path, sep='\t', header=None).to_numpy()

    def load_excluded(self, path):
        ''' Load excluded loci data 
        return: excluded_loci, an iterable object
        '''
        import pandas as pd
        return pd.read_csv(path, sep='\t', header=None).to_numpy()

    def get_loci_chr_location(self, loci_entry):
        ''' Get chromosome name and location from a loci entry '''
        return loci_entry[0], int(loci_entry[1]), int(loci_entry[2])

    def exclude_chrs(self, loci):
        ''' Exclude loci on chromosomes '''
        loci_on_chrs = []
        loci_off_chrs = []
        all_chrs = set(self.chr_lengths.keys())
        for loci_entry in loci:
            chr_name, start, end = self.get_loci_chr_location(loci_entry)
            if chr_name in all_chrs:
                loci_on_chrs.append(loci_entry)
            else:
                loci_off_chrs.append(loci_entry)
                if self.verbose: print(f'Off chr loci: {loci_entry}')
        loci_on_chrs = np.array(loci_on_chrs)
        loci_off_chrs = np.array(loci_off_chrs)
        return loci_on_chrs, loci_off_chrs

    def exclude_loci(self, loci, excluded_loci):
        ''' Exclude loci from the partition '''
        new_loci = []
        filtered_loci = []
        for loci_entry in loci:
            chr_name, start, end = self.get_loci_chr_location(loci_entry)
            ex_chr = excluded_loci[:, 0]
            ex_start = excluded_loci[:, 1].astype(int)
            ex_end = excluded_loci[:, 2].astype(int)
            overlap = np.logical_and(np.logical_and(ex_end > start, ex_start < end), ex_chr == chr_name)
            if overlap.sum() == 0:
                new_loci.append(loci_entry)
            else:
                filtered_loci.append(loci_entry)
                if self.verbose: print('Excluded loci:', loci_entry)
        new_loci = np.array(new_loci)
        return new_loci, filtered_loci

    def export(self, path):
        ''' Export loci to a bed file '''
        import pandas as pd
        df = pd.DataFrame(self.loci)
        df.to_csv(path, sep='\t', header=None, index=None)

class SlidingWindowRegion(CustomRegion):

    def __init__(self, window_size, step_size,
                 loci_info, excluded_loci, assembly, 
                 excluded_chrs=['chrX', 'chrY', 'chrM'],
                 check_length=True, 
                 verbose=False):
        ''' Initialize the partition '''
        super().__init__(loci_info, excluded_loci, assembly,
                         excluded_chrs, check_length, verbose)
        self.window_size = window_size
        self.step_size = step_size
        self.loci = self.split_pad_loci(self.loci)
        self.check_loci_within_chr()

    def split_pad_loci(self, loci):
        ''' Split loci that are too long and pad loci that are too short '''
        new_loci = []
        for region_id, loci_entry in enumerate(loci):
            chr_name, start, end = self.get_loci_chr_location(loci_entry)
            chr_length = self.chr_lengths[chr_name]
            if end - start > self.window_size:
                n_windows = (end - start) // self.step_size
                for region_sub_id, i in enumerate(range(n_windows)):
                    new_loci.append([chr_name, start + i * self.step_size, start + i * self.step_size + self.window_size, f'region_{region_id}_{region_sub_id}'])
                #if (end - start) % self.step_size != 0:
                #    new_loci.append([chr_name, end - self.window_size, end, f'region_{region_id}'])
            else:
                new_loci.append([chr_name, start, start + self.window_size, f'region_{region_id}_0'])
        return np.array(new_loci)

class CustomRangeRegion(SlidingWindowRegion):
    ''' Partition with custom input '''

    def load(self, loci_tuple):
        ''' Load loci data 
        return: loci, an iterable object
        '''
        # Check if loci tuple is the format chr, start, end
        assert len(loci_tuple[0]) == 3
        return np.array(loci_tuple)

class GenomeRegion(CustomRegion):

    def __init__(self, window_size, step_size, chr_margin,
                 excluded_loci, assembly, 
                 excluded_chrs=['chrX', 'chrY', 'chrM'],
                 check_length=True, 
                 verbose=False):
        self.window_size = window_size
        self.step_size = step_size
        self.chr_margin = chr_margin
        super().__init__(None, excluded_loci, assembly, excluded_chrs, check_length, verbose)

    def load(self, path):
        ''' Generate new loci data
        return: loci, an iterable object
        '''
        window_size = self.window_size
        step_size = self.step_size
        chr_margin = self.chr_margin
        loci = []
        region_id = 0
        if self.verbose: print('Generating loci')
        for chr_name, chr_length in self.chr_lengths.items():
            for start in range(chr_margin, chr_length - window_size - chr_margin, step_size):
                end = start + window_size
                loci.append([chr_name, start, end, f'region_{region_id}'])
                region_id += 1
        loci = np.array(loci)
        return loci

class GeneRegion(CustomRegion):

    def get_loci_chr_location(self, gff_entry):
        ''' Get chromosome name and location from a loci entry '''
        chr_name = gff_entry[0]
        anno_type = gff_entry[2]
        start = int(gff_entry[3])
        end = int(gff_entry[4])
        strand = gff_entry[6]
        return chr_name, start, end

class GeneGenomeRegion(GeneRegion):

    def __init__(self, loci_path, excluded_loci, assembly, chr_margin,
                 excluded_chrs=['chrX', 'chrY', 'chrM'],
                 check_length=True, 
                 verbose=False):
        self.chr_margin = chr_margin
        super().__init__(loci_path, excluded_loci, assembly, excluded_chrs, check_length, verbose)

    def load(self, path):
        ''' Load loci data 
        return: loci, an iterable object
        '''
        import pandas as pd
        df = pd.read_csv(path, sep='\t', header=None)
        df['gff_idx'] = df.index
        return df.to_numpy()

    def exclude_loci(self, loci, excluded_loci):
        margin_size = self.chr_margin
        chr_dict = self.chr_lengths
        chr_start_margins = [[chr_n, 0, margin_size, 'Chr Start Region'] for chr_n in chr_dict.keys()]
        chr_end_margins = [[chr_n, length - margin_size, length, 'Chr End Region'] for chr_n, length in chr_dict.items()]
        
        new_excluded_loci = np.concatenate([excluded_loci, np.array(chr_start_margins), np.array(chr_end_margins)], axis = 0)
        return super().exclude_loci(loci, new_excluded_loci)

def main():
    pass
    '''
    gene_gff = '/gpfs/data/tsirigoslab/home/jt3545/generon/data/genomes/gencode.v24.annotation.genes.unique.protein_coding_lincRNA.gff3'
    excluded = '/gpfs/data/tsirigoslab/home/jt3545/resources/blacklist/hg38-blacklist.v2.bed'
    genes = GeneRegion(gene_gff, excluded, 'hg38', 
    genes.export('gene_regions.gff3')

    # Load storage
    excluded = '/gpfs/data/tsirigoslab/home/jt3545/resources/blacklist/hg38-blacklist.v2.bed'
    regions = GenomeRegion(2000000, 500000, 1000000, excluded, 'hg38', verbose=False, excluded_chrs = ['chrX', 'chrY'])
    print(regions.loci.shape)
    regions.export('test.bed')

    atac_bed = '/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/ATAC-seq/imr90/macs2/MACS2_peaks.narrowPeak'
    excluded = '/gpfs/data/tsirigoslab/home/jt3545/resources/blacklist/hg38-blacklist.v2.bed'
    regions = CustomRegion(atac_bed, excluded, 'hg38', verbose=True)
    print(regions.loci.shape)
    regions.export('test.bed')
    '''

if __name__ == '__main__':
    main()
