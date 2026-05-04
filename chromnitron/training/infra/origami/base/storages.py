from chromnitron.training.infra.origami.base.storage import Storage

class NpyStorage(Storage):
    ''' Npy storage assume npy files are stored by chromosomes (chrX.npy, etc. '''
    def load(self, path):
        import numpy as np
        if self.verbose: print(f'Loading npy files from {path}...')
        chrs = {}
        for chr_name, chr_length in self.chr_lengths.items():
            if self.verbose: print(f'Loading {chr_name}')
            chr_data = np.load(f'{path}/{chr_name}.npy')
            chrs[chr_name] = chr_data
        return chrs

    def get(self, chr_name, start, end):
        return self.chrs[chr_name][start:end].copy()

    def get_data_chr_length(self, chr_data):
        return len(chr_data)

class ZarrStorage(NpyStorage):
    ''' Zarr storage assume zarr files are stored with chrs as groups '''
    def load(self, path):
        import zarr
        if self.verbose: print(f'Loading zarr files from {path}...')
        chrs = zarr.open(path, mode='r')['chrs']
        return chrs

class HiCNpzStorage(NpyStorage):
    ''' Npz storage assume npy files are stored by chromosomes (chrX.npy, etc. '''
    def load(self, path):
        import numpy as np
        if self.verbose: print(f'Loading npy files from {path}...')
        chrs = {}
        for chr_name, chr_length in self.chr_lengths.items():
            if self.verbose: print(f'Loading {chr_name}')
            with np.load(f'{path}/{chr_name}.npz') as chr_data:
                chrs[chr_name] = dict(chr_data)
        return chrs

    def get(self, chr_name, start, end):
        import numpy as np
        ori_load = self.chrs[chr_name]
        square_len = end - start
        diag_load = {}
        for diag_i in range(square_len):
            diag_load[str(diag_i)] = ori_load[str(diag_i)][start : start + square_len - diag_i]
            diag_load[str(-diag_i)] = ori_load[str(-diag_i)][start : start + square_len - diag_i]
        start -= start
        end -= start

        diag_region = []
        for diag_i in range(square_len):
            diag_line = []
            for line_i in range(-1 * diag_i, -1 * diag_i + square_len):
                if line_i < 0:
                    diag_line.append(diag_load[str(line_i)][start + line_i + diag_i])
                else:
                    diag_line.append(diag_load[str(line_i)][start + diag_i])
            diag_region.append(diag_line)
        diag_region = np.array(diag_region).reshape(square_len, square_len)
        return diag_region

    def get_data_chr_length(self, chr_data):
        return len(chr_data['0'])

def main():
    # Load storage
    sto_npy = NpyStorage('/gpfs/data/tsirigoslab/home/jt3545/data/datalake/pro_seq/imr90/jin_nature_2013/s6_numpy/rep1/forward', 'hg38', verbose = True)
    sto_zarr = ZarrStorage('/gpfs/data/tsirigoslab/home/jt3545/data/datalake/pro_seq/imr90/jin_nature_2013/s7_zarr/rep1/forward/chrs.zarr', 'hg38', verbose = True)
    # Get feature
    chr_name = 'chr1'
    start = 1000000
    end = 1001000
    print(sto_npy.get(chr_name, start, end))
    print(sto_zarr.get(chr_name, start, end))

if __name__ == '__main__':
    main()
