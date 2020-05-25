# iib-neuro-proj
Library containing files for IIB project investigating dendritic computation.

functs_inputs: contains low-level functions used for input generation

gen_inputs: contains high level function to produced realistic inputs

sim_hLN: contains function to simulate sub-threshold membrane potential of an hLN model

train_hLN: contains functions used to train the parameters of hLN models, including basic hLN class definition and a cluster
           of training routines.
           
init_hLN: contains initialization routines required to train complex architectures.
           
train_run: file used to run large training or experimental routines, usually on `fields' in department. Contains high level
           functions used to train complex hLN architectures
           
utils: file containing basic, low-level functions used by various high level functions across the project

plot: file containing plotting functions to visualize results from some frequently-run experiments. Some of these migrated
      in edited form to the Jupyter notebooks


Notebooks - folder containing Jupyter notebooks used to run investigative experiments throughout the course of the project.
            Not many key results contained here, each notebook explains at the top the theme of experiments within it
