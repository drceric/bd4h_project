# bd4h_project
The repo is for OMSCS Big data for health care final project.

The structure of the project are given below.

- data
  1. data_synthetic - all the processed data that fed into the model
  2. mimic-code
    - this is the sql files that build all the data we need. Inthis folder, all the sql files under concepts folder are developed by our own (not the folder under concepts) 
  4. query 
  5. mimic-iii-clinical-database-1.4 - the source data, we do not include this here
  6. sepsis_utils
  7. static - this folder stored all the static data after processing.
  8. mimic_simulation.ipynb - this is the notebook that we run the mimic based stnthetic simulation, based on the original and edit accordingly.
  9. sepsis-3-get-data.ipynb - this is the notebook we extract our target data and processed the data.

- dsw
This is the package based on the original paper and code.
1. mimic_dsw.ipynb is the notebook that train the mmodel and generate the Figure 1 in the paper.

- simulation
  1. full_synthetic_simulation.py - this it the pyhon file that run the original source simulation
  2.  Please notice that the MIMIC-based simulation in under data folder.

- model_eval.ipynb
This is the notebook that run all the evaluation in our tables. Developed by our own.


**How to run the code?**
1. We assume the MIMIC dataset is available to the reader. If not, follow this https://doi.org/10.5281/zenodo.1256723 to build the databse. 
2. If not building the database yourself, the data are available in the data_synthetic folder.
3. model_eval.ipynb is the main code source we use for model. This generate the target Table 1 and Table 2.
4. Figure 1 code can be found under dsw folder, the notebook mimic_dsw.ipynb run the dsw model and generate the related distribution plots.

**Dependency**
The original repo is based on python 3.6, pytorch 1.4.
But using the most recent Python 3.8 and the most recent Pytorch works.
