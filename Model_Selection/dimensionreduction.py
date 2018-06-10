import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition

class PCR:
    def __init__(self, m_, reg_model_):
        self.m = m_                   # number of priciple components
        self.pca = None               # pca instance in sklearn
        self.reg_model = reg_model_   # regression model we used for PCA
        
    def fit(self, X_, y_):
        self.pca = decomposition.PCA(n_components=self.m)
        self.pca.fit(X_)
        Xpc = self.pca.transform(X_)
        self.reg_model.fit(Xpc, y_)
    
    def predict(self, X_):
        Xpc = self.pca.transform(X_)
        return self.reg_model.predict(Xpc)
    
    

def main():
    pass

if __name__ == "__main__":
    main()