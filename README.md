# README #

This README provides steps that are necessary to train and test the Deep Learning (DL) methods of PIPENN described in the paper. You can also use these steps to run PIPENN for the interface prediction of your own protein sequences.

### What is PIPENN? ###

PIPENN stands for 'Protein Interface Prediction with an Ensemble of Neural Nets'. PIPENN is a set of various DL methods for predicting protein bindings of different interaction types (protein-protein, protein-small-molecule, protein-nucleotide(DNA/RNA)) at residue-level, using only information from a protein sequence.

### How can I set up the required environment? ###

1. Be sure that you have already installed *Python 3.7* (or higher version). 
1. Get [mini-conda](https://docs.conda.io/en/latest/miniconda.html), install it, create a virtual environment (*my-env*), and activate *my-env*.
1. Install all packages mentioned in *conda-packages.yaml* in your virtual environment. Note that if you want to run the PIPENN models on GPU's, you have to install *tensorflow-gpu* instead of *tensorflow*.
1. Clone this repository somewhere in your path (*my-path*).
1. Go to *my-path/pipenn* and you will see the same folder structure as you can see in this repository.

### What do these folders contain? ###

There are seven folders, each containing the source code of a specific DL method:

1. *ann-ppi*: the source code for the PIPENN's *ann-predictors* based on the fully connected Artificial Neural Network (ANN).
1. *dnet-ppi*: the source code for the PIPENN's *dnet-predictors* based on the Dilated Convolutional Networks (DCN).
1. *unet-ppi*: the source code for the PIPENN's *unet-predictors* based on a special variant of the Convolutional Neural Networks (CNN) tuned for object detection (U-Net).
1. *rnet-ppi*: the source code for the PIPENN's *rnet-predictors* based on the Residual Convolutional Networks (ResNet).
1. *rnn-ppi*: the source code for the PIPENN's *rnn-predictors* based on the Recurrent Neural Networks (RNN).
1. *cnn-rnn-ppi*: the source code for the PIPENN's *cnet-predictors* based on the hybrid architectures of CNNs and RNNs.
1. *ensnet-rnn-ppi*: the source code for the PIPENN's *ensnet-predictors* based on an additional neural net architecture, which ensembles the outputs of the six neural nets and selects the best predictions.

The following Python modules are used by all the different DL methods:

1. *utils/PPIDataset*: contains a data set related configuration items, takes care for making Pandas data frames, converts data sets, generates and plots statistics about a data set, etc.
1. *utils/PPItrainTest*: contains a train/test related configuration items, takes care for training and testing models using Keras API's, prints models, generates trained models, loads trained models, logs train/test experiments, etc.
1. *utils/PPILoss*: contains a loss function related configuration items, implements different loss functions, calculates threshold, logs validation/test metrics, etc.
1. *utils/PPIPredPlot*: generates performance figures and embed them in a standalone HTML page containing ROC-AUC and Precision-Recall curves, and the prediction probabilities.
1. *utils/PPIExplanation*: uses SHAP library to explain the result of a model (feature importance).
1. *utils/PPIParams*: contains all default parameters. These parameters can be overwritten in the Python module for a specific DL method.
1. *utils/PPIlogger*: defines PIPENN logger that is used by all Python modules.

The following folders are used by all the different DL methods:

1. *config*: contains configuration for the different loggers of PIPENN. Each DL method has its own logger.
1. *logs*: will contain the log of each specific DL method (e.g., *dnet-ppi.log*) after running (training or testing) the method. At this moment, it's empty.
1. *jobs*: contains SLURM job definitions and shell scripts for running the Python modules. You can use them if you want to run (specially to train) the Python modules on GPU's or CPU's.

### How can I train a DL method with an existing training data set? ###

1. Go to https://www.ibi.vu.nl/downloads/PIPENN/BioDL-Datasets.
1. There are four training data sets: 
	* *prepared_biolip_win_a_training.csv*: contains all types of interaction data (protein-protein, protein-small-molecule, and protein-DNA/RNA).
	* *prepared_biolip_win_p_training.csv*: contains only protein-protein interaction data.
	* *prepared_biolip_win_s_training.csv*: contains only protein-small-molecule interaction data.
	* *prepared_biolip_win_n_training.csv*: contains only protein-DNA/RNA interaction data.	
1. Open one of the DL methods (e.g., *my-path/pipenn/dnet-ppi/dnet-XD-ppi-keras.py*) and be sure that the following parameters have been set properly:
	* *datasetLabel = 'Biolip_N'* #if the training data set is *prepared_biolip_win_n_training.csv* (the other options are 'Biolip_A', 'Biolip_P', and 'Biolip_S' for other training data sets)
	* *ONLY_TEST = False*
1. Be sure that you have activated your mini-conda environment (*my-env*) by using: *conda activate my-env*.
1. As we immediately apply the trained model on the testing data sets, you have to also download the corresponding testing data sets (e.g., *prepared_biolip_win_n_testing.csv* and *prepared_ZK448_win_n_benchmark.csv*) and copy them in *my-path/pipenn/data*.
1. Go to the *jobs* folder of the DL method you want to train (e.g., *my-path/pipenn/jobs/dnet-ppi*) and run one of the shell scripts. There are two shell scripts: *dnet-lisa-job.sh* (for running on HPC using SLURM and GPU's) and *dnet-pcoms-job.sh* (for running on a usual linux using CPU's). If you prefer to run the methods on a Windows machine you can easily change these scripts to a *.bat* scripts.
1. After running your specific DL method (e.g., *my-path/pipenn/dnet-ppi/dnet-XD-ppi-keras.py*), you can see the outputs in *my-path/pipenn/models* (e.g., *my-path/pipenn/models/dnet-ppi*) and you can see the logs in *my-path/pipenn/logs* (e.g., *my-path/pipenn/logs/dnet-ppi.log*). The model-file for the example trained model will be *my-path/pipenn/models/dnet-ppi/dnet-ppi-model.hdf5*; besides you will see the following HTML files for the testing data sets: *my-path/pipenn/models/dnet-ppi/PPLOTS_preds-BLTEST_N.html* and *my-path/pipenn/models/dnet-ppi/PPLOTS_preds-ZK448_N.html*.
1. Just look at the generated HTML files if you want to see the performance metrics for each testing data set.   

### How can I apply a pre-trained model on an existing testing data set? ###

1. Go to https://www.ibi.vu.nl/downloads/PIPENN/PretrainedModels.
1. There are five folders that have been named based on the training data sets:
	* *BioDL-A-Models*: contains all DL models that have been trained on BioDL-A (all interaction types) training data set.
	* *BioDL-P-Models*: contains all DL models that have been trained on BioDL-P (protein-protein) training data set.
	* *BioDL-S-Models*: contains all DL models that have been trained on BioDL-S (protein-small-molecule) training data set.
	* *BioDL-N-Models*: contains all DL models that have been trained on BioDL-N (protein-DNA/RNA) training data set.	
1. Go to one of these folders and download one of the pre-trained DL models (e.g., *BioDL-N-Models/dnet-ppi-model.hdf5*) and copy it in the proper sub-folder in the *models* folder (e.g., *my-path/pipenn/models/dnet-ppi*).
1. Open one of the DL methods (e.g., *my-path/pipenn/dnet-ppi/dnet-XD-ppi-keras.py*) and be sure that the following parameters have been set properly:
	* *datasetLabel = 'Biolip_N'* #if the training data set of the pre-trained model is *prepared_biolip_win_n_training.csv* (the other options are 'Biolip_A', 'Biolip_P', and 'Biolip_S' for other training data sets)
	* *ONLY_TEST = True*
1. Download also the corresponding testing data sets (e.g., *prepared_biolip_win_n_testing.csv* and *prepared_ZK448_win_n_benchmark.csv*) and copy them in *my-path/pipenn/data*.
1. Repeat the steps 6 to 8, from the previous section.

### How can I train/test the ensemble method with an existing training/testing data set? ###

1. Be sure that you have already copied your preferred training and testing data sets to *my-path/pipenn/data* (e.g., *prepared_biolip_win_n_training.csv*, *prepared_biolip_win_n_testing.csv*, *prepared_ZK448_win_n_benchmark.csv*). 
1. Go to https://www.ibi.vu.nl/downloads/PIPENN/PretrainedModels.
1. Go to one of the sub-folders (e.g., *BioDL-N-Models*).
1. Download all pre-trained DL models, except for *ensnet-ppi-model.hdf5*.
1. Copy them in the proper sub-folders of the *models* folder (copy *ann-ppi-model.hdf5* to *my-path/pipenn/models/ann-ppi*, copy  *dnet-ppi-model.hdf5* to *my-path/pipenn/models/dnet-ppi*, etc.).
1. Open *my-path/pipenn/dnet-ppi/ensnet-ppi-keras.py* and set properly the two parameters: *datasetLabel* and *ONLY_TEST*.
1. Go to *my-path/pipenn/jobs/ensnet-ppi* and run one of the shell scripts.
1. You will see the outputs in *my-path/pipenn/models/ensnet-ppi* and the logs in *my-path/pipenn/logs/ensnet.log*.

### How can I use my own training and/or testing data set with the DL methods? ###

1. Your training and testing data sets must have the same layout (features and formats) as the existing data sets.
1. Copy your own training and testing data sets to *my-path/pipenn/data*.
1. Rename your training and testing files to one of the existing data files.
1. Set properly the two parameters: *datasetLabel* and *ONLY_TEST*.
1. Follow the steps explained before.    

### Do you also provide a webserver? ###

Not yet. We are working on it.

### Who can I talk to? ###

* Reza Haydarlou (e-mail in the paper)