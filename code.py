#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:35:29 2021

@author: lanlan
"""
import matplotlib.pyplot as plt
#from scipy.special import softmax
import numpy as np
import bonnerlib2D as bl2d
import pickle
import sklearn.linear_model as lin
from numpy.random import randn


print("Q1,a:")
with open('cluster_data.pickle','rb') as file:
    dataTrain,dataTest = pickle.load(file)
    Xtrain,Ttrain = dataTrain
    Xtest,Ttest = dataTest

clf = lin.LogisticRegression(multi_class='multinomial',solver='lbfgs')
clf.fit(Xtrain,Ttrain)

train_pred = clf.predict(Xtrain)
print("Accuracy of Train")
print(clf.score(Xtrain,Ttrain))
test_pred = clf.predict(Xtest)
print("Accuracy of Test")
print(clf.score(Xtest,Ttest))



print("Q1,b:")
plt.suptitle("Question 1(b): decision boundaries for linear classification.")
bl2d.plot_data(Xtrain,Ttrain)
bl2d.boundaries(clf)

W = clf.coef_
b = clf.intercept_

print("Q1,e:")
def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x,1,keepdims=True)
    return softmax_x

def predict(X,W,b):
    X_b = np.c_[np.ones((len(X),1)),X]
    W_b = np.c_[b,W]
    z = np.dot(X_b,W_b.T)

    x = softmax(z)
    
    return np.argmax(x,axis = 1)

k = predict(Xtest,W,b)

p = clf.predict(Xtest)

print(sum((k-p)**2))



print("Q1,f:")

def one_hot(Tint):
    n_class = np.max(Tint)+1
    length = len(Tint)
    one_h = np.zeros((length,n_class))
    one_h[np.arange(length),Tint] = 1
    return one_h

print(one_hot([4,3,2,3,2,1,2,1,0]) )



print("Q2, a:")

train_accuracys = []
test_accuracys = []
train_cross_entropys = []
test_cross_entropys = []



np.random.seed(7)




def GDlinear(I,lrate):
    print('learning rate =',lrate)
    Xtrain_1 = np.c_[np.ones(len(Xtrain)),Xtrain]
    Xtest_1 = np.c_[np.ones(len(Xtest)),Xtest]
    train_one_hot = one_hot(Ttrain)
    test_one_hot = one_hot(Ttest)

    
    Weight_train = np.random.randn(3,3)
    Weight_train = Weight_train/10000
    #Weight_test = np.random.randn(3,3)
    #Weight_test = Weight_test/10000

    for i in range(I):
        
        z_train = np.dot(Xtrain_1,Weight_train.T)#build z
      
        z_test = np.dot(Xtest_1,Weight_train.T)
        
        
        s_train = softmax(z_train)#build softmax y of train
        s_test = softmax(z_test)
        #print(s_train)

        n = len(Xtrain)
        #get cross entropy
     
        train_cross_entropy = -np.mean(np.log(s_train[np.arange(len(Ttrain)),Ttrain]))
        test_cross_entropy = -np.mean(np.log(s_test[np.arange(len(Ttest)),Ttest]))

        gradient_train_1 =  np.sum(1/n *(s_train - train_one_hot),0).T
        gradient_train_2 = 1/n * np.dot(Xtrain.T,(s_train - train_one_hot)).T

        gradient_train =  np.c_[gradient_train_1,gradient_train_2]

        train_cross_entropys.append(train_cross_entropy)
        test_cross_entropys.append(test_cross_entropy)
        
        train_pred = np.argmax(softmax(np.dot(Xtrain_1,Weight_train.T)),axis = 1)
        test_pred = np.argmax(softmax(np.dot(Xtest_1,Weight_train.T)),axis = 1)
        
        train_true = Ttrain
        test_true = Ttest
        
        
        train_accuracy = np.mean((train_pred==train_true))
        test_accuracy = np.mean((test_pred==test_true))
        train_accuracys.append(train_accuracy)
        test_accuracys.append(test_accuracy)
        

        Weight_train = Weight_train - lrate * gradient_train
 
    
    return Weight_train





    
a = GDlinear(10000,0.1)







#print(train_accuracys)


plt.figure(0)
plt.suptitle("Question 2(a):Training and test loss v.s. iterations.")
plt.xlabel("Iteartion number")
plt.ylabel("Cross entropy")
plt.semilogx(train_cross_entropys,color = "blue")
plt.semilogx(test_cross_entropys,color = "red")
#plt.ylim(0.4,1.1)


plt.figure(2)
plt.suptitle("Question 2(a):Training and test accuracy v.s. iterations.")
plt.xlabel("Iteartion number")
plt.ylabel("Accuracy")
plt.semilogx(train_accuracys,color = "blue")
plt.semilogx(test_accuracys,color = "red")
plt.ylim(0.8,0.84)


plt.figure(3)
plt.suptitle("Question 2(a): test loss from iteration 50 on.")
plt.xlabel("Iteartion number")
plt.ylabel("Cross entropy")
plt.semilogx(test_cross_entropys[:50],color = "red")
#plt.ylim(0.5,1.1)

plt.figure(4)
plt.suptitle("Question 2(a): train loss from iteration 50 on.")
plt.xlabel("Iteartion number")
plt.ylabel("Cross entropy")
plt.semilogx(train_cross_entropys[:50],color = "blue")

plt.figure(92)
plt.suptitle("Question 2(a): decision boundaries for linear classification.")
bl2d.plot_data(Xtrain,Ttrain)
bl2d.boundaries2(a[:,1:],a[:,0],predict)
plt.figure()

train_accuracys = []
test_accuracys = []
train_cross_entropys = []
test_cross_entropys = []

GDlinear(10000,10)
plt.figure(11)
plt.suptitle("Question 2(a):Training and test loss v.s. iterations,lrate = 10.")
plt.xlabel("Iteartion number")
plt.ylabel("Cross entropy")
plt.semilogx(train_cross_entropys,color = "blue")
plt.semilogx(test_cross_entropys,color = "red")

train_accuracys = []
test_accuracys = []
train_cross_entropys = []
test_cross_entropys = []

GDlinear(10000,1)
plt.figure(12)
plt.suptitle("Question 2(a):Training and test loss v.s. iterations,lrate = 1.")
plt.xlabel("Iteartion number")
plt.ylabel("Cross entropy")
plt.semilogx(train_cross_entropys,color = "blue")
plt.semilogx(test_cross_entropys,color = "red")

train_accuracys = []
test_accuracys = []
train_cross_entropys = []
test_cross_entropys = []

GDlinear(10000,0.001)
plt.figure(13)
plt.suptitle("Question 2(a):Training and test loss v.s. iterations,lrate = 0.001.")
plt.xlabel("Iteartion number")
plt.ylabel("Cross entropy")
plt.semilogx(train_cross_entropys,color = "blue")
plt.semilogx(test_cross_entropys,color = "red")

train_accuracys = []
test_accuracys = []
train_cross_entropys = []
test_cross_entropys = []

GDlinear(10000,0.00001)
plt.figure(14)
plt.suptitle("Question 2(a):Training and test loss v.s. iterations,lrate = 0.00001.")
plt.xlabel("Iteartion number")
plt.ylabel("Cross entropy")
plt.semilogx(train_cross_entropys,color = "blue")
plt.semilogx(test_cross_entropys,color = "red")

train_accuracys = []
test_accuracys = []
train_cross_entropys = []
test_cross_entropys = []

GDlinear(10000,0.1)
plt.figure(15)
plt.suptitle("Question 2(a):Training and test loss v.s. iterations,lrate = 0.1.")
plt.xlabel("Iteartion number")
plt.ylabel("Cross entropy")
plt.semilogx(train_cross_entropys,color = "blue")
plt.semilogx(test_cross_entropys,color = "red")





print("Q2,a xi:")
print("Training accuracy:")
print(train_accuracys[-1])
print(clf.score(Xtrain,Ttrain))
print(train_accuracys[-1]- clf.score(Xtrain,Ttrain))
print("Q2,a xii:")
print("Testing accuracy:")
print(test_accuracys[-1])
print(clf.score(Xtest,Ttest))
print(test_accuracys[-1] - clf.score(Xtest,Ttest))





print("Q2,d:")




import sklearn as sk
train_accuracyss = []
test_accuracyss = []
train_cross_entropyss = []
test_cross_entropyss = []

def SGDlinear(I,batch_size,lrate0,alpha,kappa):
    np.random.seed(7)
    print("batch size = " , batch_size)
    print('initial learning rate = ' ,lrate0)
    print('decay rate = ', alpha)
    print('burn in period = ' , kappa)
    print('learning rate =',lrate0)
    lrate0 = np.random.randn()

    Weight_train = np.random.randn(3,3)
    Weight_train = Weight_train/10000
    
    lrate = lrate0
    for i in range(I):#renew landa I = 500
        Xtdata_ = sk.utils.shuffle(np.c_[Xtrain,Ttrain])
        Xtrain_ = Xtdata_[:,:-1]
        Ttrain_ = Xtdata_[:,-1]
        Ttrain_ = Ttrain_.astype(int)
        n = int(len(Xtrain_)/batch_size)
        
        Xttdata_ = sk.utils.shuffle(np.c_[Xtest,Ttest])
        Xtest_ = Xttdata_[:,:-1]
        Ttest_ = Xttdata_[:,-1]
        Ttest_ = Ttest_.astype(int)
        
        
        #add one column
        Xtrain_0 = np.c_[np.ones(len(Xtrain_)),Xtrain_]
        Xtest_0 = np.c_[np.ones(len(Xtest)),Xtest_]
        #one hot
        train_one_hot = one_hot(Ttrain_)
        test_one_hot = one_hot(Ttest)
        

        for j in range(n):# renew weight
           
            z_train = np.dot(Xtrain_0[batch_size*j:batch_size*(j+1)],Weight_train.T)#build z
            s_train = softmax(z_train)
            
            gradient_train_1 =  np.sum(1/n *(s_train - train_one_hot[batch_size*j:batch_size*(j+1)]),0).T
            gradient_train_2 = 1/n * np.dot(Xtrain_[batch_size*j:batch_size*(j+1)].T,(s_train - train_one_hot[batch_size*j:batch_size*(j+1)])).T

            
            gradient_train =  np.c_[gradient_train_1,gradient_train_2]
            Weight_train = Weight_train - lrate*gradient_train

        lrate = lrate0/(1+alpha*(i-kappa))
        
        z_train = np.dot(np.c_[np.ones(len(Xtrain)),Xtrain],Weight_train.T)
        z_test = np.dot(np.c_[np.ones(len(Xtest)),Xtest],Weight_train.T)#build z
        s_train = softmax(z_train)
        s_test = softmax(z_test)
        

         
        train_cross_entropy = -np.mean(np.log(s_train[np.arange(len(Ttrain)),Ttrain]))
        train_cross_entropyss.append(train_cross_entropy)
        test_cross_entropy = -np.mean(np.log(s_test[np.arange(len(Ttest)),Ttest]))
        test_cross_entropyss.append(test_cross_entropy)

        train_pred = np.argmax(softmax(np.dot(np.c_[np.ones(len(Xtrain)),Xtrain],Weight_train.T)),axis = 1)
        test_pred = np.argmax(softmax(np.dot(np.c_[np.ones(len(Xtest)),Xtest],Weight_train.T)),axis = 1)

        train_accuracy = np.mean((train_pred==Ttrain))
        train_accuracyss.append(train_accuracy)
        test_accuracy = np.mean((test_pred==Ttest))
        test_accuracyss.append(test_accuracy)
 
    
    return Weight_train

a_ = SGDlinear(500,30,0.01,1,0.1)
#print("SGD")
print(train_accuracyss[-1])
print(test_accuracyss[-1])
#print(train_cross_entropyss[-1])

plt.figure(5)
plt.suptitle("Question 2(d):Training and test loss v.s. iterations.")
plt.xlabel("Iteartion number")
plt.ylabel("Cross entropy")
plt.semilogx(train_cross_entropyss,color = "blue")
plt.semilogx(test_cross_entropyss,color = "red")
#plt.ylim(0.4,1.1)

plt.figure(6)
plt.suptitle("Question 2(d):Training and test accuracy v.s. iterations.")
plt.xlabel("Iteartion number")
plt.ylabel("Accuracy")
plt.semilogx(train_accuracyss,color = "blue")
plt.semilogx(test_accuracyss,color = "red")
#plt.ylim(0.8,0.84)


plt.figure(7)
plt.suptitle("Question 2(d): test loss from iteration 50 on.")
plt.xlabel("Iteartion number")
plt.ylabel("Cross entropy")
plt.semilogx(test_cross_entropyss[:50],color = "red")
#plt.ylim(0.5,1.1)

plt.figure(8)
plt.suptitle("Question 2(d): train loss from iteration 50 on.")
plt.xlabel("Iteartion number")
plt.ylabel("Cross entropy")
plt.semilogx(train_cross_entropyss[:50],color = "blue")


plt.figure(93)
plt.suptitle("Question 2(d): decision boundaries for linear classification.")
bl2d.plot_data(Xtrain,Ttrain)
bl2d.boundaries2(a_[:,1:],a_[:,0],predict)
plt.figure()




print("Difference of training accuracy:")
print(train_accuracyss[-1]- clf.score(Xtrain,Ttrain))
print("Difference of training accuracy:")
print(clf.score(Xtest,Ttest) - test_accuracyss[-1])

    
    
print("Q3:")

print("Q3,a")
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf_2 = QuadraticDiscriminantAnalysis(store_covariance=True)  
clf_2.fit(Xtrain,Ttrain)
print(clf_2.score(Xtrain,Ttrain))
clf_2.fit(Xtest,Ttest)
print(clf_2.score(Xtest,Ttest))

plt.figure(16)
bl2d.plot_data(Xtrain,Ttrain)
plt.suptitle("Q3(a).decision boundaries of QDA")
bl2d.boundaries(clf_2)

plt.figure()


print("Q3,b")
from sklearn.naive_bayes import GaussianNB
clf_3 = GaussianNB()
clf_3.fit(Xtrain,Ttrain)
print(clf_3.score(Xtrain,Ttrain))
clf_3.fit(Xtest,Ttest)
print(clf_3.score(Xtest,Ttest))

bl2d.plot_data(Xtrain,Ttrain)
plt.suptitle("Q3(b).decision boundaries of QDA")
bl2d.boundaries(clf_3)
plt.figure()



print("Q3,f")
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
def EstMean(X,T):
    
    T_ = np.argmax(T,axis = 1)

    X_samples = len(X)
    X_features = len(X[0])
    
    classes, T_only = np.unique(T_,return_inverse = True)
    means = np.zeros((len(classes),X_features))
    np.add.at(means,T_only,X)
    mu_i = means/np.expand_dims((np.bincount(T_)),1)
    
    return mu_i



clf_Q = QuadraticDiscriminantAnalysis(store_covariance = True)
clf_Q.fit(Xtrain,Ttrain)

print("difference between clf and EstMean")
print(np.sum((clf_Q.means_-EstMean(Xtrain,one_hot(Ttrain)))**2))


print("Q3,g")
def EstCov(X,T):
    T_ = np.argmax(T,axis = 1)
    X_samples = len(X)
    X_features = len(X[0])
    classes, T_only = np.unique(T_,return_inverse = True)
    means = np.zeros((len(classes),X_features))
    np.add.at(means,T_only,X)
    mu_i = means/np.expand_dims((np.bincount(T_)),1)
    mu_i_len = len(mu_i)
    mu_i_cor = len(mu_i[1])

    X_ni = np.reshape(X,[X_samples,1,X_features])
    mu_ki = np.reshape(mu_i,[1,mu_i_len,mu_i_cor])
    A_nki = X_ni - mu_ki    
    
    A_nki = np.reshape(A_nki,[len(A_nki),len(A_nki[1]),len(A_nki[0][1]),1])
    A_nkj = np.reshape(A_nki,[len(A_nki),len(A_nki[1]),1,len(A_nki[0][1])])
        
    B_nkij = A_nki * A_nkj
    
    T_nk = np.reshape(T,[T.shape[0],T.shape[1],1,1])
    
    N = np.sum(T_nk,axis = 0)-1
    N_ = np.reshape(N,[N.shape[0],1,1])
    
    return np.sum(T_nk * B_nkij,axis = 0)/N_
print("difference between clf and EstCov")
print(np.sum(EstCov(Xtrain,one_hot(Ttrain))-clf_Q.covariance_)**2)



print("Q3,h")


def Estprior(T):
    X_samples = len(T)
    prior_p = np.bincount(T)/X_samples
    return prior_p

print("difference between clf and Estprior")

print(np.sum((clf_Q.priors_ - Estprior(Ttrain))**2))





print("Q3,i")
print("I dont know")
print("Q3,j")
print("I dont know")
print("Q3,k")
print("I dont know")



print("Q4,a:")
from sklearn.neural_network import MLPClassifier
from  matplotlib.pyplot import subplot
#10,000 iterations of stochastic gradi- ent descent (sgd)
#learning rate of 0.01
#an optimization tolerance of 10−6.
np.random.seed(7)   
clf_4 = MLPClassifier(hidden_layer_sizes = 5,activation= 'logistic', solver = 'sgd',learning_rate_init= 0.01,max_iter = 10000,tol=10**-6)
clf_4.fit(Xtrain,Ttrain)
print(clf_4.score(Xtrain,Ttrain))
print(clf_4.score(Xtest,Ttest))

plt.figure(9)
plt.suptitle("4(a): neural net with 5 hidden units.")
bl2d.plot_data(Xtrain,Ttrain)
bl2d.boundaries(clf_4)
plt.figure()



print("Q4,b:")
clf_layer_1 = MLPClassifier(hidden_layer_sizes = 1,activation= 'logistic', solver = 'sgd',learning_rate_init= 0.01,max_iter = 10000,tol=10**-6)
clf_layer_1.fit(Xtrain,Ttrain)

clf_layer_2 = MLPClassifier(hidden_layer_sizes = 2,activation= 'logistic', solver = 'sgd',learning_rate_init= 0.01,max_iter = 10000,tol=10**-6)
clf_layer_2.fit(Xtrain,Ttrain)

clf_layer_4 = MLPClassifier(hidden_layer_sizes = 4,activation= 'logistic', solver = 'sgd',learning_rate_init= 0.01,max_iter = 10000,tol=10**-6)
clf_layer_4.fit(Xtrain,Ttrain)

clf_layer_10 = MLPClassifier(hidden_layer_sizes = 10,activation= 'logistic', solver = 'sgd',learning_rate_init= 0.01,max_iter = 10000,tol=10**-6)
clf_layer_10.fit(Xtrain,Ttrain)




plt.figure(20)
plt.suptitle("Question 4(b): Neural net decsion boundaries.")
plt.subplot(2,2,1)
plt.title("1 hidden units")
bl2d.plot_data(Xtrain,Ttrain)
bl2d.boundaries(clf_layer_1)

plt.subplot(2,2,2)
plt.title("2 hidden units")
bl2d.plot_data(Xtrain,Ttrain)
bl2d.boundaries(clf_layer_2)

plt.subplot(2,2,3)
plt.title("4 hidden units")
bl2d.plot_data(Xtrain,Ttrain)
bl2d.boundaries(clf_layer_4)

plt.subplot(2,2,4)
plt.title("10 hidden units")
bl2d.plot_data(Xtrain,Ttrain)
bl2d.boundaries(clf_layer_10)
plt.figure()




    
    
    

    