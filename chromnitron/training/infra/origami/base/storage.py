from chromnitron.training.infra.origami.base.genome import Genome

class Storage(Genome):
    ''' Storage class for storing and retrieving data from a genome. '''

    def __init__(self, path, assembly, 
                 excluded_chrs=['chrX', 'chrY', 'chrM'],
                 check_length=True, 
                 verbose=False):
        ''' Initialize storage
        path: path to the storage
        assembly: assembly name
        excluded_chrs: excluded chromosomes
        '''
        super().__init__(assembly, excluded_chrs)
        self.path = path
        self.verbose = verbose
        self.chrs = self.load(path) # load storage
        if check_length:
            self.check_data_chr_length()
        else:
            if self.verbose: print('Skip checking chromosome length')

    def load(self, path):
        ''' Load data from storage path
        return: chrs, a dictionary
        '''
        raise NotImplementedError

    def get(self, chr_name, start, end):
        ''' Get feature from storage
        chr_name: chromosome name
        start: start position
        end: end position
        '''
        raise NotImplementedError

    def get_data_chr_length(self, chr_data):
        ''' Get chromosome length from data
        chr_data: chromosome data loaded from storage
        '''
        raise NotImplementedError

    def get_length_dict(self):
        ''' Get a dictionary of length for all chrs '''
        data_chr_length_dict = {}
        for chr_name, chr_length in self.chr_lengths.items():
            if chr_name not in self.chrs:
                if self.verbose: print(f'Skipping {chr_name}: chromosome not found in storage')
                continue
            data_chr_length_dict[chr_name] = self.get_data_chr_length(self.chrs[chr_name])
        return data_chr_length_dict

    def check_data_chr_length(self):
        ''' Check if the chromosome length in data is consistent with the chromosome length in the dictionary '''
        for chr_name, chr_length in self.chr_lengths.items():
            if chr_name not in self.chrs:
                if self.verbose: print(f'Skipping {chr_name}: chromosome not found in storage')
                continue
            chr_data = self.chrs[chr_name]
            data_chr_length = self.get_data_chr_length(chr_data)
            if chr_length != data_chr_length:
                raise Exception(f'Chromosome {chr_name} length in data {data_chr_length} is not consistent with the chromosome length in the dictionary {chr_length}')
        if self.verbose: print('Chromosome length in data is consistent.')
