Code is available at: https://github.com/dianasaur323/supervised_learning

* Requirements *
A working version of Python3 (I am using Python 3.7.1)

* Setup *
1) Create a virtual environment `python3 -m venv venv`
2) Activate the virtual environment `source venv/bin/activate`
3) Install necessary requirements `pip install -r requirements.txt`

* Accessing the Data *
Data is available at https://www.kaggle.com/areeves87/rscience-popular-comment-removal/version/3

* Running the program *
1) `python3 main.py`
2) Model name? prompt accepts the following: "N" for neural network, "K" for k-means, "EM" for gmm
3) Data set? prompt accepts the following: "reddit" for Kaggle reddit data, empty for news data
3) Feature selection? prompt accepts the following: "k" for k-means, "em" for gmm, "rp" for random projection,
"pca", "ica", "lda"
