import numpy as np ##
import time
##preprocessing:

##Idea N1: Convert to  1-D vector rep: each character is a utf-8 ord'ed integer. This is an input.
#preprocessing

def loss_prime(X, W, y_pred):
    return np.sum(2 * (X @ W - y_pred) @ W)

class Baseline_OLS:
    def __init__(self):
        self.W = []
        self.processing = self.Process_Into_256_Vector
    def fit(self, X_in, y): #labels as 1-5 star ratings: WE TEST THIS AND RETURN A CONTINUOUS VALUE HERE FIRST!
        X = self.processing(X_in)
        print(X)
        self.W = np.linalg.inv(X.T @ X) @ X.T @ y

    def fit_GD(self, X_in, y, epoch = 10**7, eta = 0.01):
        X= self.processing(X_in)
        for i in range(epoch):
            self.W = self.W - loss_prime(X, self.W, y) * eta

    def predict(self, X_in):
        X_as_array = np.array([X_in]) if type(X_in) == type("Ayo") else X_in #mild error fixing here.
        time_bfr = time.time()
        X = self.processing(X_as_array)
        pred = X @ self.W
        print(f'Time taken: {time.time() - time_bfr}')
        print(f'Predicted {pred} stars')
        return pred
    
    def Process_Into_256_Vector(self, vect_of_inputs, size_lim = 10**3): #length max is 10k. Keep y as reformat is easy.
        resulting_vec_rep_array = []
        for i in vect_of_inputs:
            if len(i) > size_lim:
                raise ValueError("Heyo, you have too much text. Surrender your master's program and give us kindergarten text.")
            vector_rep = np.array([ord(j) for j in i])
            # zero_vec = 0.0000000005 * np.random.randn(size_lim - len(vector_rep))
            zero_vec = np.zeros(size_lim - len(vector_rep))
            vector_rep = np.append(vector_rep, zero_vec)
            resulting_vec_rep_array = np.array([vector_rep]) if len(resulting_vec_rep_array) == 0 else np.append(resulting_vec_rep_array, np.array([vector_rep]), axis=0)
        return resulting_vec_rep_array
    

X, y = np.array(["I hate sushi!", "I hate life", "I hate ex", "Hate", "hat", "approve of milk."]), np.array([1, 1, 1, 1, 1,5])
model1 = Baseline_OLS()
model1.fit(X,y)
model1.predict("hat")