# Better-than-Baseline FNC implementation

An extensive documentation of the Fake News Challenge baseline implementation can be found [here](https://github.com/FakeNewsChallenge/fnc-1-baseline/blob/master/README.md).

All our models ended up beating the baseline implementation, but unfortunately, we ran all the training on Google Cloud, and couldn't retrieve our trained models on time. Maybe one day, we will see how our model could have performed on the actual test data. But until that day comes, we leave our (mostly experimental) code here for reference, as well as a link to the [slides](https://github.com/aravindsiv/fnc-1-baseline/blob/master/StanceDetection.pdf) we made while presenting this project.


### Training the first stage model
To train the first stage models, run ``stageone_train.sh`` file. The file will train SVM, logistic regression, and neural network models on the 5 folds of data. The models are saved in ./models. Each model is saved with the filename in the form of 'stage_onefoldmodel' , where ``stage_one`` remains constant and model =[svm, logistic, neural net] for the types of algorithms used to train the model and fold=[0,1,2,3,4] for 5 folds. For example, the logistic regression trained on 4th fold will be saved as 'stage_one3logistic' under ./models folder.

### Training the second stage model
To train the second stage models, run ``keras_models.py`` file. The file will train one of the following models: (LSTM, GRU and a neural net which takes the sum of its input word embeddings) on a particular fold of data. Both the type of algorithm and the fold are detemined by the below parameters to the file.

-rnn: The type of neural net being trained on the data. The user can select from the following options(no, lstm, gru) to train a feed forward network based on sum of word embeddings , lstm  or gru respectively.

-fold: The fold of data on which the model is to be trained and tested. The options are on of the following: [0,1,2,3,4].
 
The trained models are saved in ./models folder. Each model is saved with the filename in the form of 'fprefixFold.h5' , where  model =[sumRNN, lstm, gru] for the types of neural networks to be trained and Fold=[0,1,2,3,4] for 5 folds. For example, the lstm trained on 4th fold will be saved as 'lstm3.h5' under ./models folder.
 
### Running the EDA on training set for stage one
Run ``eda.py`` to generate distribution graphs across the classes. The graphs will be saved in the current directory.

Run ``eda2.py`` to generate graphs based on corordinates of features values for stage one. The graphs for all combinations of features and generated and saved in the current directory.

### Running both Stage One and Stage Two, end to end
To run both stage one and stage two in conjunction, run ``final_model.py``. Currently, you can only run the file on certain pre-trained best models. The best model for second stage is already hardcoded in the file. You can run the selected best models of first stage, which have been trained on fold 4, by giving the options, while running this file.

-model : You can either select form the following options(logistic, svm, nn) to run the pretrained logistic regression, svm  or one layer neural network respectively.

-model_path: Mention the path of the model file.

The file will print the score of the models, based on the scoring scheme given by the organizers.
