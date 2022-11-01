# README #

This README provides steps that are necessary to train and test the Deep Learning (DL) methods of PIPENN described in our paper: 

Bas Stringer*, Hans de Ferrante, Sanne Abeln, Jaap Heringa, K. Anton Feenstra and Reza Haydarlou* (2022).
PIPENN: Protein Interface Prediction from sequence with an Ensemble of Neural Nets. [Bioinformatics](https://doi.org/10.1093/bioinformatics/btac071)

You can also use these steps to run PIPENN for the interface prediction of your own protein sequences.

You may also be interested to use the [PIPENN webserver at www.ibi.vu.nl/programs/pipennwww/](https://www.ibi.vu.nl/programs/pipennwww/) which allows you to try out your queries of interest. Typical runtimes per protein are on the order of 10 minutes.

### What is PIPENN? ###

PIPENN stands for 'Protein Interface Prediction from sequence with an Ensemble of Neural Nets'. PIPENN is a suite of DL methods for predicting protein bindings of different interaction types (protein-protein, protein-small-molecule, protein-nucleotide(DNA/RNA)) at residue-level, using only information from a protein sequence as input.

In the following, we explain how the computing environment can be set up in order to re-train and test the different DL methods provided by the PIPENN suite. As PIPENN is written in *Python*, we first explain how to set up *Python* and the required libraries, after downloading this github repository. Moreover, we explain how to download the PIPENN data sets, copy them in proper folders in your own environment, update the value of some important parameters, and train/test the DL methods. A quick start to get familiar with PIPENN is to apply the provided pre-trained models on the provided test and benchmark data sets.

### How can I set up the required environment? ###

PIPENN suite has been developed in *Python*, using a number of machine learning related *Python* libraries (packages). Please follow the following steps to set up your own environment:
1. Be sure that you have already installed *Python 3.7* (or higher version). 
1. Get [mini-conda](https://docs.conda.io/en/latest/miniconda.html), install it, create a virtual environment (e.g., *my-env*), and activate *my-env*.
1. Install all packages mentioned in *conda-packages.yaml* in your virtual environment. Note that if you want to run the PIPENN models on GPU's, you have to install *tensorflow-gpu* instead of *tensorflow*.
1. Clone this repository somewhere in your path (e.g., *my-path*).
1. Go to *my-path/pipenn* and you will see the same folder structure as you can see in this repository.

### What do source code folders contain? ###

There are seven folders, each containing the source code of a specific DL method:

1. *ann-ppi*: the source code for the PIPENN's *ann-predictors* based on the fully connected Artificial Neural Network (ANN).
1. *dnet-ppi*: the source code for the PIPENN's *dnet-predictors* based on the Dilated Convolutional Networks (DCN).
1. *unet-ppi*: the source code for the PIPENN's *unet-predictors* based on a special variant of the Convolutional Neural Networks (CNN) tuned for object detection (U-Net).
1. *rnet-ppi*: the source code for the PIPENN's *rnet-predictors* based on the Residual Convolutional Networks (ResNet).
1. *rnn-ppi*: the source code for the PIPENN's *rnn-predictors* based on the Recurrent Neural Networks (RNN).
1. *cnn-rnn-ppi*: the source code for the PIPENN's *cnet-predictors* based on the hybrid architectures of CNNs and RNNs.
1. *ensnet-rnn-ppi*: the source code for the PIPENN's *ensnet-predictors* based on an additional neural net architecture, which ensembles the outputs of the six neural nets and selects the best predictions.

The following Python modules are used by all the different DL methods:

1. *utils/PPIDataset*: contains a data set related configuration items, takes care of making Pandas data frames, converts data sets, generates and plots statistics about a data set, etc.
1. *utils/PPItrainTest*: contains train/test related configuration items, takes care of training and testing models using Keras API's, prints models, generates trained models, loads trained models, logs train/test experiments, etc.
1. *utils/PPILoss*: contains a loss function related configuration items, implements different loss functions, calculates threshold, logs validation/test metrics, etc.
1. *utils/PPIPredPlot*: generates performance figures and embed them in a standalone HTML page containing ROC-AUC and Precision-Recall curves, and the prediction probabilities.
1. *utils/PPIExplanation*: uses SHAP library to explain the result of a model (feature importance).
1. *utils/PPIParams*: contains all default parameters. These parameters can be overwritten in the Python module for a specific DL method.
1. *utils/PPIlogger*: defines PIPENN logger that is used by all Python modules.

The following folders are used by all the different DL methods:

1. *config*: contains configuration for the different loggers of PIPENN. Each DL method has its own logger.
1. *logs*: will contain the log of each specific DL method (e.g., *dnet-ppi.log*) after running (training or testing) the method. At this moment, it's empty.
1. *jobs*: contains SLURM job definitions and shell scripts for running the Python modules. You can use them if you want to run (specially to train) the Python modules on GPU's or CPU's.

### What do data set folders contain? ###

Because the data set files are large, we have not included them in this github repository. They are available at the site of our group. Follow the following steps to download them and place in the appropriate folders:
1. Please go to https://www.ibi.vu.nl/downloads/PIPENN/PIPENN/BioDL-Datasets.
1. There are four training data sets: 
	* *prepared_biolip_win_a_training.csv*: contains all types of interaction data (protein-protein, protein-small-molecule, and protein-DNA/RNA).
	* *prepared_biolip_win_p_training.csv*: contains only protein-protein interaction data.
	* *prepared_biolip_win_s_training.csv*: contains only protein-small-molecule interaction data.
	* *prepared_biolip_win_n_training.csv*: contains only protein-DNA/RNA interaction data.	
1. There are four independent testing data sets constructed by our group and are mutual exclusief with the training data sets: 
	* *prepared_biolip_win_a_testing.csv*: contains all types of interaction data (protein-protein, protein-small-molecule, and protein-DNA/RNA).
	* *prepared_biolip_win_p_testing.csv*: contains only protein-protein interaction data.
	* *prepared_biolip_win_s_testing.csv*: contains only protein-small-molecule interaction data.
	* *prepared_biolip_win_n_testing.csv*: contains only protein-DNA/RNA interaction data.	
1. There are four independent benchmark data sets constructed by [Kurgan Lab](http://biomine.cs.vcu.edu/) and are mutual exclusief with the training data sets: 
	* *prepared_ZK448_win_a_benchmark.csv*: contains all types of interaction data (protein-protein, protein-small-molecule, and protein-DNA/RNA).
	* *prepared_ZK448_win_p_benchmark.csv*: contains only protein-protein interaction data.
	* *prepared_ZK448_win_s_benchmark.csv*: contains only protein-small-molecule interaction data.
	* *prepared_ZK448_win_n_benchmark.csv*: contains only protein-DNA/RNA interaction data.	
1. Copy all the downloaded files to *my-path/pipenn/data*.

### How can I train a DL method with an existing training data set? ###

We provide seven DL methods. All DL methods can be trained (re-trained) in a similar way using one of the four training data sets. Here, as an example, we explain steps to be followed for training our *Dilated Convolutional Network (dnet)*, trained on *prepared_biolip_win_n_training.csv*:
1. Be sure that you have activated your mini-conda environment (*my-env*) by using: *conda activate my-env*.
1. Open *my-path/pipenn/dnet-ppi/dnet-XD-ppi-keras.py* and be sure that the following parameters have been set properly:
	* *datasetLabel = 'Biolip_N'* #if the training data set is *prepared_biolip_win_n_training.csv* (the other options are 'Biolip_A', 'Biolip_P', and 'Biolip_S' for other training data sets)
	* *ONLY_TEST = False*
1. Be sure that *prepared_biolip_win_n_training.csv*, *prepared_biolip_win_n_testing.csv*, and *prepared_ZK448_win_n_benchmark.csv* are in *my-path/pipenn/data* (note that we immediately apply the trained model on the testing data sets).
1. Go to *my-path/pipenn/jobs/dnet-ppi* and run one of the shell scripts. There are two shell scripts: *dnet-lisa-job.sh* (for running on HPC using SLURM and GPU's) and *dnet-pcoms-job.sh* (for running on a usual linux using CPU's). If you prefer to run the methods on a Windows machine you can easily change these scripts to a *.bat* scripts.
1. After running the script, you will get the following outputs:
	* *my-path/pipenn/models/dnet-ppi/dnet-ppi-model.hdf5* (model file) 
	* *my-path/pipenn/logs/dnet-ppi.log* (log file)
	* *my-path/pipenn/models/dnet-ppi/PPLOTS_preds-BLTEST_N.html* (metrics for the testing file)
	* *my-path/pipenn/models/dnet-ppi/PPLOTS_preds-ZK448_N.html* (metrics for the benchmark file).
1. Just look at the generated HTML files if you want to see the performance metrics for each testing data set.   

### How can I apply a pre-trained model on an existing testing data set? ###

We already provide our pre-trained DL models for the case if you don't want to train the DL methods but only want to apply them on a testing data set. The testing process is similar for all seven DL models, including our *ensemble* model. Here, as an example, we explain steps to be followed for testing our pre-trained model *Dilated Convolutional Network (dnet)*, trained on *prepared_biolip_win_n_training.csv*:
1. Go to https://www.ibi.vu.nl/downloads/PIPENN/PIPENN/Pretrained-Models/BioDL-N-Models.	
1. Download *dnet-ppi-model.hdf5* and copy it in the proper sub-folder *my-path/pipenn/models/dnet-ppi*.
1. Open *my-path/pipenn/dnet-ppi/dnet-XD-ppi-keras.py* and be sure that the following parameters have been set properly:
	* *datasetLabel = 'Biolip_N'* 
	* *ONLY_TEST = True*
1. Be sure that the corresponding testing data sets *prepared_biolip_win_n_testing.csv* and *prepared_ZK448_win_n_benchmark.csv* are already in *my-path/pipenn/data*.
1. Repeat the steps 4 to 6, from the previous section.

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
