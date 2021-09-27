This repository contains all the code to reproduce the analysis and figures
of the following paper, which analyses and models the ORN-LN circuit in the 
antennal lobe of the Drosophila larva:

## Normative and mechanistic model of an adaptive circuit for efficient encoding and feature extraction
### Nikolai M Chapochnikov, Cengiz Pehlevan, Dmitri B Chklovskii


https://www.biorxiv.org/content/10.1101/2021.09.24.461723v1

Please cite this paper if you use the code.

Code author: Nikolai M. Chapochnikov


##  Steps to reproduce the analysis and figures of the paper

Install Python 3 and the required packages (e.g., via Anaconda) in the file
"required_packages.txt"



Put the folder root folder "ORN-LN_circuit" in your home directory.


Run the following files in the "ORN-LN_circuit/bin" folder (this step may be skipped):
- act_preprocess.py
- con_preprocess.py
- act_odors_ORN_vs_con_ORN-LN.py
- act_ORN_vs_con_ORN-LN.py
- act_ORN_whitening.py
- olf_circ_offline_sims.py

This will regenerate files that are already present in the "results" folder.

Then run the file "ORN-LN_circuit/bin/plots_paper.py", which will generate the plots in the paper in the folder "plots/plots_paper".

To obtain additional supplementary figures, rerun the file "ORN-LN_circuit/bin/plots_paper.py" with SCAL_W = 10 instead of SCAL_W = 2 (on line 3032 of "plots_paper.py"), which will in addition generate the folder whitening10 inside plots/plots_paper/[date_time].


NOTE: because some simulations are non deterministic, the generated plots might slightly differ from the plots in the paper.