import models

if __name__ == "__main__":
    model_name = input("Model name?: ")
    if(model_name.upper() == "DECISION TREE"):
        model = models.DecisionTree()
        print("Decision Tree")
