Code is available at: https://github.com/dianasaur323/supervised_learning

* Requirements *
A working version of Python3 (I am using Python 3.7.1)

* Setup *
1) Create a virtual environment `python3 -m venv venv`
2) Activate the virtual environment `source venv/bin/activate`
3) Install necessary requirements `pip install -r requirements.txt`

* Accessing the Data *
Data is available at https://www.kaggle.com/areeves87/rscience-popular-comment-removal/version/3

* Resources *
https://pdfs.semanticscholar.org/8b69/7734036148e24fd151f825b6686d373edb4a.pdf
https://www.dataquest.io/blog/learning-curves-machine-learning/
http://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf

* Running the program *
--- 1) python -m textblob.download_corpora
1) `python3 main.py`
2) Model name? prompt accepts the following: "D" for decision tree, "N" for neural network, "B" for boosting, "S"
for SVM, "K" for KNN
3) Data set? prompt accepts the following: "reddit" for Kaggle reddit data, empty for news data
3) Manual? prompt accepts the following: "Y" for run of manually tuned model, empty for either grid-search
or a static model. Not all models have a manually tuned model - only those that took too long
to grid-search have it.

* Decision Tree implementation *
https://scikit-learn.org/stable/modules/tree.html#tree
