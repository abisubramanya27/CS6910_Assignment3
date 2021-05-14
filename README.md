# CS6910_Assignment3

## Index
- [Assignment3.ipynb](src/Assignment3.ipynb) -- Google colab notebook containing the code for the entire assignment
- [predictions_vanilla](predictions_vanilla) -- folder containing the [predictions_vanilla.csv](predictions_vanilla/predictions_vanilla.csv) file, which has the top K predictions for each input from test data (using the best non-attention/vanilla model) written to it
- [predictions_attention](predictions_attention) -- folder containing the [predictions_attention.csv](predictions_attention/predictions_attention.csv) file, which has the top K predictions for each input from test data (using the best attention based model) written to it
- [Report](https://wandb.ai/abisheks/assignment3/reports/Assignment-3-CS6910--Vmlldzo2MzA5OTA) - Link to WANDB report
- [Runs](https://wandb.ai/abisheks/assignment3?workspace=user-abisheks) - Link to WANDB runs page

## Requirements
All the python libraries required to run the program on a CPU (without Google colab) are listed in `requirements.txt` ([link](requirements.txt)).
They can be installed using 
```shell
pip install -r requirements.txt
```
**(Use Python 3.7 or lower versions of Python 3)**

The `plot_model()` function from `keras.utils.vis_utils` requires additionally the installation of **graphviz**, for which you can refer to [https://graphviz.gitlab.io/download/](https://graphviz.gitlab.io/download/). This again is needed only for running locally instead of Google colab.

The program is expected to run much faster on a GPU (in general) and the notebook can be run on Google colab without any additional installations than the ones done in the notebook itself. 

**NOTE :** Since we use recurrent dropout, some of the models may not meet the requirements for CuDNN. That warning can be safely ignored.


## Steps to run the program
**NOTE :** The program is written in a modular manner, with each logically separate unit of code written as functions.  

- The code is done in a Google colab notebook and stored in the path `src/Assignment3.ipynb` ([link](src/Assignment3.ipynb)). It can be opened and run in Google colab or jupyter server locally.
- The solution to each question is made in the form of a couple of function calls (which are clearly mentioned with question numbers in comments) and commented out in the program so that the user can choose which parts to run and evaluate.
- In order to run the solution for a particular question, uncomment that part and run the cell. Also check the comments in that cell for any cells that need to be run before.
- There are separate functions for creating training model and infefence model, training and sweeping for attention and vanilla models which are clearly mentioned as headings and in comments for the function as well.
- There is a **config** dictionary which is passed to WANDB during training. This is aslo used to create the model in the `seq2seq_no_attention()` and `seq2seq_attention()` functions. The python dictionary contains all hyperparameters and architectural informations for a model. It should contain the following keys :
  ```python
  "learning_rate" --  Learning rate used in gradient descent
  "epochs" --  Number of epochs to train the model
  "optimizer" --  Gradient descent algorithm used for the parameter updation
  "batch_size" --  Batch size used for the optimizer
  "loss_function" --  Loss function used in the optimizer
  "architecture" --  Type of neural network used
  "dataset" --  Name of dataset
  "inp_emb_size" -- Size of input embedding layer
  "no_enc_layers" -- Number of layers in the encoder
  "no_dec_layers" -- Number of layers in the decoder
  "hid_layer_size" -- Size of hidden layer
  "dropout" -- Value of dropout used in the normal and recurrent dropout
  "cell_type" -- Type of cell used in the encoder and decoder ('RNN' or 'GRU' or 'LSTM')
  "beam_width" -- Beam width used in beam decoder
  "attention" -- Whether or not attention is used
  ```
  
