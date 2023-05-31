# Assignment 3 - Language modelling and text generation using RNNs

## 1.	Contributions
While the majority of the work in this assignment is mine, I used Ross’s utility functions in ```req_functions.py``` and in-class notebooks for reference, discussed and compared code with class mates, and used ChatGPT to troubleshoot errors and unwanted outputs.

## 2.	Assignment Description
“Text generation is hot news right now!
For this assignemnt, you're going to create some scripts which will allow you to train a text generation model on some culturally significant data - comments on articles for The New York Times. You can find a link to the data [here](https://www.kaggle.com/datasets/aashita/nyt-comments).
You should create a collection of scripts which do the following:
-	Train a model on the Comments section of the data 
  -	Save the trained model
-	Load a saved model 
  -	Generate text from a user-suggested prompt
Language modelling is hard and training text generation models is doubly hard. For this course, we lack somewhat the computationl resources, time, and data to train top-quality models for this task. So, if your RNNs don't perform overwhelmingly, that's fine (and expected). Think of it more as a proof of concept.
-	Using TensorFlow to build complex deep learning models for NLP
-	Illustrating that you can structure repositories appropriately
-	Providing clear, easy-to-use documentation for your work. ”

## 3.	Methods
The ```train_model.py``` script loads the data sample specified before cleaning, tokenizing, and padding the data. The model is then trained and subsequently saved to the ```out``` folder. The ```prompt.py``` script uses ```argparse``` to take input from the terminal when it is being executed, generating text based on this input and printing it to the terminal.

## 4.	Usage
First, download the dataset from the link in the description above and unzip the resultant folder to the ```data``` folder. Before running the script, the relevant packages need to be installed. To do this, ensure that the current directory is ```assignment-3---rnns-for-text-generation-nikolaimh``` and run ```pip install –r requirements.txt``` from the terminal, which will install the required packages.

Then, move to the source folder with ```cd src/``` and run the training script with ```python train_model.py``` or the prompt script with ```prompt.py [‘seed text’] [number of next words]```. A trained model is already present for use in the ```out``` folder, but it has not been trained particularly well due to time constraints.

The training script is currently set to run on a sample of 200 rows of comments from each data file, for a total of 1800 comments. This can be changed on ```line 41```, if so desired, and the number of epochs trained for can be changed on ```line 76```, which is set to 20 epochs at present.

## 5.	Discussion
The final result is ultimately rather unimpressive and incoherent, rarely returning full words. While this may be due to errors in my training or prompt scripts, it might just as well be due to the model’s rather limited training.

Due to crashes, stalls, and server funds expiring at inopportune times, I did not have the chance to train the final model for any more than the number of epochs and comments set as the current default. That is to say, the model present in the ```out``` folder is, at the time of writing, poorly trained and barely functional, though I hope the code itself stands as proof that it could train a stronger model, if allowed the time.
