from sklearn.base import BaseEstimator
import numpy as np
import scipy.stats as stats
#from collections import Counter

# For this assignment we will implement the Naive Bayes classifier as a
# a class, sklearn style. You only need to modify the fit and predict functions.
# Additionally, implement the Disparate Impact measure as the evaluateBias function.
class NBC(BaseEstimator):
    '''
    (a,b) - Beta prior parameters for the class random variable
    alpha - Symmetric Dirichlet parameter for the features
    '''
    
    def __init__(self, a=1, b=1, alpha=1):
        self.a = a
        self.b = b
        self.alpha = alpha
        self.__params = None
    
    def get_a(self):
        return self.a
    
    def get_b(self):
        return self.b
    
    def get_alpha(self):
        return self.alpha
    
    # you need to implement this function
    #nbTrain
    def fit(self,X,y):
        '''
        This function does not return anything
            
        Inputs:
        X: Training data set (N x d numpy array)
        y: Labels (N length numpy array)
        '''
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        self.__classes = np.unique(y)
        
        params = None
        N = len(y)
        N1 = np.sum(y == 1)
        N2 = np.sum(y == 2)
        #Calculating theta (Eq 5)
        theta_bayes_1 = (N1+a)/(N+a+b) #P(Y=1)
        #Calcuating params for Naive Bayes (Eq 8 & 9)
        customer_good = np.where(y == 1) #returns all indices where y=1
        good_reviews = X[customer_good[0],:] #gets specific rows where Y=1
        theta_1_j = []
        for i in range(len(good_reviews[0])):
            unique_features, unique_counts = np.unique(good_reviews[:,i], return_counts=True) #returns unique features and their occurances
            Kj = len(unique_features) #Count of unqiue values for a given feature
            #unique_counts = np.unique(good_reviews[:,i], return_counts=True)[1]
            temp = []
            for j in range (len(unique_counts)): #Computing theta_1_j
                theta_1 = (unique_counts[j]+alpha)/(N1+Kj*alpha)
                temp.append (theta_1)
            theta_1_j.append(temp)
    
        customer_bad = np.where(y == 2) #returns all indecies where y=2
        bad_reviews = X[customer_bad[0],:] #creating matrix of rows where Y=2
        theta_2_j = []
        for i in range(len(bad_reviews[0])):
            unique_features_b, unique_counts_b = np.unique(bad_reviews[:,i], return_counts=True)
            Kj_b = len(unique_features_b)
            temp_b = []
            for j in range (len(unique_counts_b)): #Computing theta_2_j
                theta_2 = (unique_counts_b[j]+alpha)/(N2+Kj*alpha)
                temp_b.append (theta_2)
            theta_2_j.append(temp_b)
        
        params = [theta_bayes_1,theta_1_j,theta_2_j]
        # do not change the line below
        self.__params = params

    # you need to implement this function
    #nbPredict
    def predict(self,Xtest):
        '''
        This function returns the predicted class for a given data set
        
        Inputs:
        Xtest: Testing data set (N x d numpy array)
        
        Output:
        predictions: N length numpy array containing the predictions
        '''
        params = self.__params
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        #remove next line and implement from here
        predictions = np.random.choice(self.__classes,np.unique(Xtest.shape[0]))
        #do not change the line below
        return predictions

def evaluateBias(y_pred,y_sensitive):
    '''
        This function computes the Disparate Impact in the classification predictions (y_pred),
        with respect to a sensitive feature (y_sensitive).
        
        Inputs:
        y_pred: N length numpy array
        y_sensitive: N length numpy array
        
        Output:
        di (disparateimpact): scalar value
        '''
   
    n = len(y_pred)
    
    #P(Yˆ = 2|S != 1)
    count1 = 0
    s2 = np.sum(y_sensitive == 1)
    s1 = n - s2

    for i in range(n):
        if (y_sensitive[i]!=1):
            if(y_pred[i]==2):
                count1 = count1 + 1
    p1 = count1/s1

    #P(Yˆ = 2|S = 1)
    count2 = 0
    for x in range(n):
        if (y_sensitive[x]==1):
            if(y_pred[x]==2):
                count2 = count2 + 1
    p2 = count2/s2

    di = p1/p2
    #do not change the line below
    return di

def genBiasedSample(X,y,s,p,nsamples=1000):
    '''
        Oversamples instances belonging to the sensitive feature value (s != 1)
        
        Inputs:
        X - Data
        y - labels
        s - sensitive attribute
        p - probability of sampling unprivileged customer
        nsamples - size of the resulting data set (2*nsamples)
        
        Output:
        X_sample,y_sample,s_sample
        '''
    i1 = y == 1 # good
    i1 = i1[:,np.newaxis]
    i2 = y == 2 # bad
    i2 = i2[:,np.newaxis]
    
    sp = s == 1 #privileged
    sp = sp[:,np.newaxis]
    su = s != 1 #unprivileged
    su = su[:,np.newaxis]
    
    su1 = np.where(np.all(np.hstack([su,i1]),axis=1))[0]
    su2 = np.where(np.all(np.hstack([su,i2]),axis=1))[0]
    sp1 = np.where(np.all(np.hstack([sp,i1]),axis=1))[0]
    sp2 = np.where(np.all(np.hstack([sp,i2]),axis=1))[0]
    inds = []
    for i in range(nsamples):
        u = stats.bernoulli(p).rvs(1)
        if u == 1:
            #sample one bad instance with s != 1
            inds.append(np.random.choice(su2,1)[0])
            #sample one good instance with s == 1
            inds.append(np.random.choice(sp1,1)[0])
        else:
            #sample one good instance with s != 1
            inds.append(np.random.choice(su1,1)[0])
            #sample one bad instance with s == 1
            inds.append(np.random.choice(sp2,1)[0])
    X_sample = X[inds,:]
    y_sample = y[inds]
    s_sample = s[inds]
    
    return X_sample,y_sample,s_sample,inds
