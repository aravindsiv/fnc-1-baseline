# Baseline FNC implementation

Information about the fake news challenge can be found on [FakeChallenge.org](http://fakenewschallenge.org).

This repository contains code that reads the dataset, extracts some simple features, trains a cross-validated model and
performs an evaluation on a hold-out set of data.

Credit:
* Byron Galbraith (Github: @bgalbraith, Slack: @byron)
* Humza Iqbal (GitHub: @humzaiqbal, Slack: @humza)
* HJ van Veen (GitHub/Slack: @mlwave)
* Delip Rao (GitHub: @delip, Slack: @dr)
* James Thorne (GitHub/Slack: @j6mes)
* Yuxi Pan (GitHub: @yuxip, Slack: @yuxipan)

## Questions / Issues
Please raise questions in the slack group [fakenewschallenge.slack.com](https://fakenewschallenge.slack.com)

## Getting Started
The FNC dataset is inlcuded as a submodule and can be FNC Dataset is included as a submodule. You should download the fnc-1 dataset by running the following commands. This places the fnc-1 dataset into the folder fnc-1/

    git submodule init
    git submodule update

## Useful functions
### dataset class
The dataset class reads the FNC-1 dataset and loads the stances and article bodies into two separate containers.

    dataset = DataSet()

You can access these through the ``.stances`` and ``.articles`` variables

    print("Total stances: " + str(len(dataset.stances)))
    print("Total article bodies: " + str(len(dataset.articles)))

* ``.articles`` is a dictionary of articles, indexed by the body id. For example, the text from the 144th article can be printed with the following command:
   ``print(dataset.articles[144])``

### Hold-out set split
Data is split using the ``generate_hold_out_split()`` function. This function ensures that the article bodies between the training set are not present in the hold-out set. This accepts the following arguments. The body IDs are written to disk.

* ``dataset`` - a dataset class that contains the articles and bodies
* ``training=0.8`` - the percentage of data used for the training set (``1-training`` is used for the hold-out set)
* ``base_dir="splits/"``- the directory in which the ids are to be written to disk


### k-fold split
The training set is split into ``k`` folds using the ``kfold_split`` function. This reads the holdout/training split from the disk and generates it if the split is not present.

* ``dataset`` - dataset reader
* ``training = 0.8`` - passed to the hold-out split generation function
* ``n_folds = 10`` - number of folds
* ``base_dir="splits"`` - directory to read dataset splits from or write to

This returns 2 items: a array of arrays that contain the ids for stances for each fold, an array that contains the holdout stance IDs.

### Getting headline/stance from IDs
The ``get_stances_for_folds`` function returns the stances from the original dataset. See ``fnc_kfold.py`` for example usage.



## Scoring Your Classifier

The ``report_score`` function in ``utils/score.py`` is based off the original scorer provided in the FNC-1 dataset repository written by @bgalbraith.

``report_score`` expects 2 parameters. A list of actual stances (i.e. from the dev dataset), and a list of predicted stances (i.e. what you classifier predicts on the dev dataset). In addition to computing the score, it will also print the score as a percentage of the max score given any set of gold-standard data (such as from a  fold or from the hold-out set).

    predicted = ['unrelated','discuss',...]
    actual = [stance['Stance'] for stance in holdout_stances]

    report_score(actual, predicted)

This will print a confusion matrix and a final score your classifier. We provide the scores for a classifier with a simple set of features which you should be able to match and eventually beat!

|               | agree         | disagree      | discuss       | unrelated     |
|-----------    |-------        |----------     |---------      |-----------    |
|   agree       |    118        |     3         |    556        |    85         |
| disagree      |    14         |     3         |    130        |    15         |
|  discuss      |    58         |     5         |   1527        |    210        |
| unrelated     |     5         |     1         |    98         |   6794        |
Score: 3538.0 out of 4448.5	(79.53%)

### Training the first stage model
To train the first stage model run ``stageone_train.sh`` file. The file will train svm, logistic, neural net on the 5 folds of data. The models are saved in ./models. Each model is saved with the filename in the form of 'stage_onefoldmodel' , where ``stage_one`` remains constant and model =[svm, logistic, neural net] for the types of algorithms used to train the model and fold=[0,1,2,3,4] for 5 folds. For example, 
The logistic regression trained on 4th fold will be saved as 'stage_one3logistic' under ./models folder.

### Training the second stage model
To train the second stage model run ``keras_models.py`` file. The file will train one of the following algorithms :(lstm, gru and a neural net which takes the sum of its input word embeddings) on a particular fold of data. Both the type of algorithm and the fold are detemined by the below parameters to the file:
-rnn: The type of neural net being trained on the data. The user can select from the following options(no, lstm, gru) to train a feed forward network based on sum of word embeddings , lstm  or gru respectively.
-fold: The fold of data on which the model is to be trained and tested. The options are on of the following: [0,1,2,3,4] 
The trained models are saved in ./models folder. Each model is saved with the filename in the form of 'fprefixFold.h5' , where  model =[sumRNN, lstm, gru] for the types of neural networks to be trained and Fold=[0,1,2,3,4] for 5 folds. For example, 
The lstm trained on 4th fold will be saved as 'lstm3.h5' under ./models folder.
 
### Running the EDA on training set for stage one
Run ``eda.py`` to generate distribution graphs across the classes. The graphs will be saved in the current directory
Run ``eda2.py`` to generate graphs based on corordinates of features values for stage one. The graphs for all combinations of features and generated and saved in the current directory

### Running both Stage One and Stage Two, end to end
To run both stage one and stage two in conjunction. Run ``final_model.py``. Currently, you can only run the file on certain pre-trained best models. The best model for second stage is already hardcoded in the file. You can run the selected best models of first stage, which have been trained on fold 4, by giving the options, while running this file:
-model : You can either select form the following options(logistic, svm, nn) to run the pretrained logistic regression, svm  or one layer neural network respectively.
-model_path: Mention the path of the model file
The file will print the score of the models, based on the scoring scheme given by the organizers.
