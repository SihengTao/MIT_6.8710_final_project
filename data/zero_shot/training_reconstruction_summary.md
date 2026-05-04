# Chromnitron ENCODE reconstruction

## Summary

- Pair-level supplement target reconstructed to 1081 selected cell line/CAP pairs.
- Accession-level reconstruction expanded to 1105 ChIP-seq experiments.
- Unique CAPs retained after pair selection: 767.
- Best-fit pair reconstruction matches all Supplementary Fig. 2 overlaps except HepG2-IMR-90, which remains one pair lower than the panel total.

## Raw ENCODE candidate counts

| Cell line | Matched experiments | Matched cell line/CAP pairs | Explicit >=40M experiment candidates | Explicit >=40M pair candidates | Selected 1081 pairs | Selected 1105 accessions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GM12878 | 157 | 126 | 147 | 119 | 106 | 109 |
| K562 | 556 | 342 | 488 | 318 | 263 | 282 |
| HepG2 | 787 | 684 | 767 | 677 | 674 | 676 |
| HCT116 | 41 | 23 | 36 | 19 | 22 | 22 |
| IMR-90 | 16 | 16 | 16 | 16 | 16 | 16 |

## Supplement fit

| Pair of cell lines | Reconstructed overlap | Supplement target | Delta |
| --- | ---: | ---: | ---: |
| GM12878 / K562 | 78 | 78 | 0 |
| GM12878 / HepG2 | 73 | 73 | 0 |
| GM12878 / HCT116 | 19 | 19 | 0 |
| GM12878 / IMR-90 | 15 | 15 | 0 |
| K562 / HepG2 | 187 | 187 | 0 |
| K562 / HCT116 | 22 | 22 | 0 |
| K562 / IMR-90 | 15 | 15 | 0 |
| HepG2 / HCT116 | 21 | 21 | 0 |
| HepG2 / IMR-90 | 13 | 14 | -1 |
| HCT116 / IMR-90 | 4 | 4 | 0 |

## Selected ATAC per cell line

| Cell line | Selected ATAC accession | Reads | Score |
| --- | --- | ---: | ---: |
| GM12878 | ENCSR637XSC | 808164068 | 6.000 |
| K562 | ENCSR868FGK | 820247598 | 6.000 |
| HepG2 | ENCSR291GJU | 847689118 | 6.000 |
| HCT116 | ENCSR872WGW | 167214500 | 4.230 |
| IMR-90 | ENCSR200OML | 179152712 | 3.389 |

## Interpretation

- The supplement behaves like a deduplicated cell line/CAP table rather than a raw experiment count.
- The 1105/1081 gap is modeled as 24 additional accessions on already selected pairs.
- Control read depth was not directly audited here because the ENCODE embedded experiment payload exposes control accessions but not their per-control read counts.

## ENCODE sources

- TF ChIP-seq GM12878: https://www.encodeproject.org/search/?type=Experiment&assay_title=TF+ChIP-seq&status=released&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_ontology.term_name=GM12878&limit=all&format=json&frame=embedded
- TF ChIP-seq K562: https://www.encodeproject.org/search/?type=Experiment&assay_title=TF+ChIP-seq&status=released&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_ontology.term_name=K562&limit=all&format=json&frame=embedded
- TF ChIP-seq HepG2: https://www.encodeproject.org/search/?type=Experiment&assay_title=TF+ChIP-seq&status=released&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_ontology.term_name=HepG2&limit=all&format=json&frame=embedded
- TF ChIP-seq HCT116: https://www.encodeproject.org/search/?type=Experiment&assay_title=TF+ChIP-seq&status=released&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_ontology.term_name=HCT116&limit=all&format=json&frame=embedded
- TF ChIP-seq IMR-90: https://www.encodeproject.org/search/?type=Experiment&assay_title=TF+ChIP-seq&status=released&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_ontology.term_name=IMR-90&limit=all&format=json&frame=embedded
- ATAC-seq GM12878: https://www.encodeproject.org/search/?type=Experiment&assay_title=ATAC-seq&status=released&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_ontology.term_name=GM12878&limit=all&format=json&frame=embedded
- ATAC-seq K562: https://www.encodeproject.org/search/?type=Experiment&assay_title=ATAC-seq&status=released&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_ontology.term_name=K562&limit=all&format=json&frame=embedded
- ATAC-seq HepG2: https://www.encodeproject.org/search/?type=Experiment&assay_title=ATAC-seq&status=released&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_ontology.term_name=HepG2&limit=all&format=json&frame=embedded
- ATAC-seq HCT116: https://www.encodeproject.org/search/?type=Experiment&assay_title=ATAC-seq&status=released&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_ontology.term_name=HCT116&limit=all&format=json&frame=embedded
- ATAC-seq IMR-90: https://www.encodeproject.org/search/?type=Experiment&assay_title=ATAC-seq&status=released&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_ontology.term_name=IMR-90&limit=all&format=json&frame=embedded

## Notes on duplicates added to reach 1105

| Cell line | CAP | Primary accession | Duplicate accession | Duplicate score |
| --- | --- | --- | --- | ---: |
| K562 | KDM1A | ENCSR908CMW | ENCSR360HRA | 6.400 |
| K562 | NCOA2 | ENCSR349TZO | ENCSR803EKW | 6.400 |
| K562 | CEBPB | ENCSR269ZGG | ENCSR416QLJ | 5.487 |
| K562 | GMEB1 | ENCSR928KOR | ENCSR376RCX | 5.441 |
| K562 | HDAC1 | ENCSR568PGX | ENCSR711VWL | 5.394 |
| K562 | NRF1 | ENCSR998AJK | ENCSR494TDU | 5.195 |
| K562 | RAD21 | ENCSR942XQI | ENCSR879KXD | 5.144 |
| GM12878 | IKZF1 | ENCSR441VHN | ENCSR874AFU | 4.900 |
| K562 | NR4A1 | ENCSR130PDE | ENCSR692RET | 4.866 |
| K562 | ZNF354B | ENCSR674SCQ | ENCSR044IXA | 4.748 |
| K562 | ATF4 | ENCSR044UJJ | ENCSR145TSJ | 4.733 |
| K562 | TCF12 | ENCSR189TRZ | ENCSR744WOO | 4.699 |
| K562 | TRIM24 | ENCSR907MZR | ENCSR957LDM | 4.607 |
| K562 | NFATC3 | ENCSR670FDA | ENCSR051OUX | 4.584 |
| K562 | ZBTB11 | ENCSR468DVP | ENCSR985OYK | 4.457 |
| K562 | ATF3 | ENCSR028UIU | ENCSR632DCH | 4.379 |
| K562 | TEAD4 | ENCSR985RPY | ENCSR000BRK | 4.227 |
| K562 | MAZ | ENCSR643JRH | ENCSR163IUV | 3.939 |
| GM12878 | SRF | ENCSR041XML | ENCSR000BGE | 3.923 |
| K562 | POLR2A | ENCSR388QZF | ENCSR000BMR | 3.888 |
| HepG2 | SOX6 | ENCSR543BVU | ENCSR766TSU | 3.777 |
| GM12878 | BCLAF1 | ENCSR342THD | ENCSR000BJZ | 3.775 |
| HepG2 | MAFK | ENCSR000EEB | ENCSR000EDZ | 3.715 |
| K562 | HDAC2 | ENCSR075HTM | ENCSR893WSB | 3.706 |
