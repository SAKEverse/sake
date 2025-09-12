# SAKE (Spectral Analysis Kit for EEG processing)


<p align="left">
<img src="/sakeplan/assets/sakeico.png" alt="Banner" width="100" height="100">
</p>

SAKE is a python toolbox for analysis EEG/LFP brain signals from [labchart](https://www.adinstruments.com/products/labchart) files.
 
---
 
## Overview
SAKE currently consists of 3 different toolboxes for data organization, spectral, and connectivity analysis of LFP/EEG signals
 
- **[SAKE-plan](/docs/sakeplan.md)**
- **[SAKE-plot](/docs/sakeplot.md)**
- **[SAKE-connectivity](/docs/sakeconnectivity.md)**

 ---
 
 ## How to install and run
1) Download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html) Python distribution, if you don't already have conda installed on your computer.
2) Clone or download [sake](https://github.com/SAKEverse/sake) repository.
3) Run *SAKE Plan.lnk* file in the sake-plan repository. First run will create the sake environemnt which can take several minutes.


## Considerations
* Only supports Windows OS due to labchart API support.
* Labchart files should contain only one block, otherwise the longest block will be selected.
