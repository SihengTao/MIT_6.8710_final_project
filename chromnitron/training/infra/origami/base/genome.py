
from __future__ import annotations

import json
import os
from pathlib import Path


class Genome:
    ''' Abstract geome class for inheritance purpose. '''

    def __init__(self, assembly, excluded_chrs=['chrX', 'chrY', 'chrM']):
        self.assembly = assembly
        self.chr_lengths = self.load_chr_dict(assembly, excluded_chrs) # load chromosome lengths

    def load_chr_dict(self, assembly, excluded_chrs=[]):
        ''' Load chromosome dictionary and remove excluded chromosomes '''
        explicit_path = os.environ.get('CHROMNITRON_CHR_SIZES_PATH')
        if explicit_path:
            assembly_path = Path(explicit_path)
            if assembly_path.exists():
                with open(assembly_path, 'r') as f:
                    chr_lengths = {}
                    for line in f:
                        parts = line.rstrip('\n').split('\t')
                        if len(parts) < 2:
                            continue
                        chr_name = parts[0]
                        if not self.is_primary_chr(chr_name):
                            continue
                        chr_lengths[chr_name] = int(parts[1])
                for chr_name in excluded_chrs:
                    if chr_name in chr_lengths:
                        del chr_lengths[chr_name]
                return chr_lengths

        candidate_dirs = [
            os.environ.get('CHROMNITRON_CHR_SIZES_DIR'),
            '/gpfs/data/tsirigoslab/home/jt3545/resources/chr_sizes',
        ]
        for path in candidate_dirs:
            if not path:
                continue
            assembly_path = Path(path) / f'{assembly}.autosomes.chrX.chrom.sizes'
            if assembly_path.exists():
                with open(assembly_path, 'r') as f:
                    chr_lengths = {}
                    for line in f:
                        parts = line.rstrip('\n').split('\t')
                        if len(parts) < 2:
                            continue
                        chr_name = parts[0]
                        if not self.is_primary_chr(chr_name):
                            continue
                        chr_lengths[chr_name] = int(parts[1])
                for chr_name in excluded_chrs:
                    if chr_name in chr_lengths:
                        del chr_lengths[chr_name]
                return chr_lengths
        return self.load_chr_dict_deprecated(assembly, excluded_chrs)

    def load_chr_dict_deprecated(self, assembly, excluded_chrs=[]):
        ''' Load chromosome dictionary and remove excluded chromosomes '''
        assembly_path = (
            Path(__file__).resolve().parent.parent
            / 'info'
            / 'assembly_lengths'
            / f'{assembly}.json'
        )
        with open(assembly_path, 'r') as f:
            chr_lengths = json.load(f)
        chr_lengths = { 'chr' + k: v for k, v in chr_lengths.items() }
        for chr_name in excluded_chrs:
            if chr_name in chr_lengths:
                del chr_lengths[chr_name]
        return chr_lengths

    def get_chr_length(self, chr_name):
        ''' Get chromosome length '''
        return self.chr_lengths[chr_name]

    @staticmethod
    def is_primary_chr(chr_name):
        '''Keep only primary assembly chromosomes and drop alt/random contigs.'''
        if not chr_name.startswith('chr'):
            return False
        suffix = chr_name[3:]
        if suffix in {'X', 'Y', 'M'}:
            return True
        return suffix.isdigit() and 1 <= int(suffix) <= 22
