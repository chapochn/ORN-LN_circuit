This repository contains all the code to reproduce the analysis and figures
of the following paper, which analyses and models the ORN-LN circuit in the 
antennal lobe of the Drosophila larva:

## Normative and mechanistic model of an adaptive circuit for efficient encoding and feature extraction
### Nikolai M. Chapochnikov, Cengiz Pehlevan, Dmitri B. Chklovskii


https://doi.org/10.1073/pnas.2117484120

Please cite this paper if you use the code.

Code author: Nikolai M. Chapochnikov


##  Steps to reproduce the analysis and figures of the paper

Install Python 3 and the required packages (e.g., via Anaconda) in the file
"required_packages.txt"



Put the folder root folder "ORN-LN_circuit" (or a link to it) in your home directory.


Running the following files "ORN-LN_circuit/bin" folder will compute data
in the 'results' folder,
necessary to generate the plots (below). These simulations have been prerun, so
this step can be skipped.
- act_preprocess.py
- con_preprocess.py
- con_analysis_MvsW.py (only prints a significance used for plotting)
- olf_circ_offline_sims_ORN-data.py
- olf_circ_offline_sims.py
- olf_circ_online_sims_real-syn-counts.py
- olf_circ_noM_offline_sims.py
- act_odors_ORN_vs_con_ORN-LN.py
- act_odors_ORN_vs_con_ORN-LN_recon.py
- act_odors_ORN_vs_con_ORN-LN_cdf-simulations.py (only gives plots)
- act_odors_ORN_vs_con_NNC.py
- act_ORN_vs_con_ORN-LN.py

The following files with generate all the plots present in the paper, they are
saved in plots/plots/folder_name/ (parameter in plots_import.py):
- plots_activity.py
- plots_activity-conn-cdf.py
- plots_connectivity.py
- plots_models-theory.py
- plots_models-theory-noM.py
- plots_models-ORN-act.py (needs to be run twice, with SCAL_W=2 and with SCAL_W=10 inside the file)
- plots_model-real-syn-counts.py
plots in the paper in the folder "plots/plots_paper".


NOTE: because some simulations are non-deterministic, the generated plots might
slightly differ from the plots in the paper.
