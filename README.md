# Improving model fairness in image-based computer-aided diagnosis

[![DOI](https://zenodo.org/badge/609991694.svg)](https://zenodo.org/badge/latestdoi/609991694)

## Datasets

The first dataset is provided by Medical Imaging and Data Resource Center (MIDRC) and is available through this website (https://data.midrc.org/). The AREDS dataset is publicly available on NCBI dbGAP (https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000001.v3.p1). The OHTS dataset is available upon request due to patient protection (https://ohts.wustl.edu/). The MIMIC-CXR dataset is publicly available on PhysioNet (https://www.physionet.org/content/mimic-cxr-jpg/).

## Getting started

### Prerequisites

* python >=3.6
* pytorch = 1.11.0
* torchvision = 0.12.0
* sklearn = 0.23.2
* pandas = 1.4.1
* opencv = 4.5.0
* skimage = 0.17.2
* tqdm = 4.48.2
* json = 0.9.6
* pickle = 2.2.1

### Quickstart

I used the experiment on the MIMIC-CXR dataset on the intersectional groups as an example.

```sh
python train_mimic_intersection.py
```

### Reference



### Acknowledgment

This work was supported by the National Library of Medicine under Award No. 4R00LM013001, NSF CAREER Award No. 2145640, and Amazon Research Award.
