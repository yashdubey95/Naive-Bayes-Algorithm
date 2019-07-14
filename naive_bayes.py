# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 19:52:37 2019

@author: yashd
"""
import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

def generate_train(size,mu1,cov1,mu2,cov2):
    # Generating 2D Gaussian Training Data
    label_one = [1 for i in range(size)]
    label_zero = [0 for i in range(size)]
    x,y= np.random.multivariate_normal(mu1,cov1,size).T
    G1 = np.array(list(zip(x,y,label_one)))
    a,b = np.random.multivariate_normal(mu2,cov2,size).T
    G2 = np.array(list(zip(a,b,label_zero)))
    X =np.concatenate((G1,G2),axis=0)
    train_set = pd.DataFrame(data = X)
    X_train = train_set[[0,1]]
    Y_train = train_set[2]
    return X_train,Y_train,train_set

def generate_train_4(mu1,cov1,mu2,cov2):
    # Generating 2D Gaussian Training Data for Part 4
    # No. of Examples in label_one = 300 and label_zero = 700
    label_one = [1 for i in range(300)]
    label_zero = [0 for i in range(700)]
    x,y= np.random.multivariate_normal(mu1,cov1,300).T
    G1 = np.array(list(zip(x,y,label_one)))
    a,b = np.random.multivariate_normal(mu2,cov2,700).T
    G2 = np.array(list(zip(a,b,label_zero)))
    X =np.concatenate((G1,G2),axis=0)
    train_set = pd.DataFrame(data = X)
    X_train = train_set[[0,1]]
    Y_train = train_set[2]
    return X_train,Y_train,train_set

def generate_test(size,mu1,cov1,mu2,cov2):
    # Generating 2d Gaussian Testing Data  
    label_one = [1 for i in range(size)]
    label_zero = [0 for i in range(size)]
    x1,y1= np.random.multivariate_normal(mu1,cov1,size).T
    H1 = np.array(list(zip(x1,y1,label_one)))
    a1,b1 = np.random.multivariate_normal(mu2,cov2,size).T
    H2 = np.array(list(zip(a1,b1,label_zero)))
    Y =np.concatenate((H1,H2),axis=0)
    test_set = pd.DataFrame(data = Y)
    x_test = test_set[[0,1]]
    y_test = test_set[2]
    return x_test,y_test,test_set
    
def naive_bayes(X_train,Y_train,x_test,y_test,train_set,test_set):
    
    # Calculating Prior for class 0 and class 1
    n_1 = train_set[2][train_set[2] == 1.0].count()
    n_0 = train_set[2][train_set[2] == 0.0].count()
    total = train_set[2].count()
    p_1 = n_1/total
    p_0 = n_0/total
    
    # Training model on train_set
    # Calculating mean and standard deviation and seperating data set based on class labels
    data_mean = train_set.groupby(2).mean()
    m_0 = data_mean.iloc[0]
    m_1 = data_mean.iloc[1]
    data_var = train_set.groupby(2).var()
    s_0 = np.sqrt(data_var.iloc[0])
    s_1 = np.sqrt(data_var.iloc[1])
    class0 = train_set[train_set[2]==0]
    del class0[2]
    class1 = train_set[train_set[2]==1]
    del class1[2]
    

    # Testing model on test_set
    label0=[]
    label1=[]
    for rows in x_test.iterrows():
        label0.append((sp.norm.pdf(rows[1][0],m_0[0],s_0[0]))*(sp.norm.pdf(rows[1][1],m_0[1],s_0[1]))*p_0)
        label1.append((sp.norm.pdf(rows[1][0],m_1[0],s_1[0]))*(sp.norm.pdf(rows[1][1],m_1[1],s_1[1])*p_1))
    x_test['class_0_prob']=label0
    x_test['class_1_prob']=label1
    
    posterior = pd.DataFrame(label0)
    posterior[1] = label1
    
    # Predicting class labels for test_set and comparing them with actual class labels for test_set
    # Creating a Confusion matrix and calculating Accuracy, Precision and Recall
    prediction=[]
    for rows in x_test.iterrows():
        if rows[1]['class_0_prob']>rows[1]['class_1_prob']:
            prediction.append(0)
        else:
            prediction.append(1)
            
    x_test['predict'] = prediction
    y_pred = x_test['predict']
    x_test['actual']= y_test
    
    tn=len([i for i in range(0,y_test.shape[0]) if y_test[i] == y_pred[i] and y_test[i]==0])
    fp=len([i for i in range(0,y_test.shape[0]) if y_test[i]==0 and y_pred[i]==1])
    fn=len([i for i in range(0,y_test.shape[0]) if y_test[i]==1 and y_pred[i]==0])
    tp=len([i for i in range(0,y_test.shape[0]) if y_test[i] == y_pred[i] and y_test[i]==1])
    
    confusion_matrix=np.array([[tp,fp],[fn,tn]])
    
    accuracy=((tp+tn)/(tp+tn+fn+fp))*100
    error = (1 - (accuracy/100))*100
    precision=(tp)/(tp+fp)*100
    recall=(tp)/(tp+fn)*100
    
    print("Accuracy:",accuracy)
    print("Precision:",precision)
    print("Recall:",recall)
    print("Confusion Matrix: \n",confusion_matrix)
    
    return prediction, posterior, error

def roc(df_test):
        tp,fp = 0, 0
        tpr,fpr=[],[]
        actual_p, actual_n = 0, 0
        fpr_prv=0
        auc=0
        #plt.clf()
        df_test=df_test.sort_values(by=1, ascending=False)
        for rows in df_test.iterrows():
            if rows[1][2]==1:
                actual_p+=1
            else:
                actual_n+=1
        #print(actual_p,actual_n) 
        for rows in df_test.iterrows():
            if rows[1][2]==1:
                tp+=1
            else: 
                fp+=1
            tpr.append((tp/(actual_p)))
            fpr.append((fp/(actual_n)))
            auc+=((tp/actual_p))*((fp/actual_n)-fpr_prv)
            fpr_prv=(fp/actual_n)
        print('Area Under Curve:',auc)       
        plt.plot(fpr,tpr)
        plt.show()

def plot_scatter(x_test):
    class_0_x, class_0_y = [], []
    class_1_x, class_1_y = [], []
    for rows in x_test.iterrows():
        if rows[1]['predict']==0:
            class_0_x.append(rows[1][0])
            class_0_y.append(rows[1][1])
        else:
            class_1_x.append(rows[1][0])
            class_1_y.append(rows[1][1])           
    plt.scatter(class_0_x,class_0_y,1,'blue')        
    plt.scatter(class_1_x,class_1_y,1,'red')
    plt.show()

if __name__ == '__main__':
    # Initializing 2D Gaussian Random Data    
    mu1 = [1,0]
    mu2 = [0,1.5]
    cov1=[[1,0.75],[0.75,1]]
    cov2=[[1,0.75],[0.75,1]]
   
    # Part 1 and 2:
    print('--------------------------------------------------------------------------------')
    print('Part 1 and 2')
    X_train,Y_train,train_set = generate_train(500,mu1,cov1,mu2,cov2)   # Generating a 2D Gaussian Training set of size 500.
    x_test,y_test, test_set = generate_test(500,mu1,cov1,mu2,cov2)      # Generating a 2D Gaussian Test set of size 500.
    pred,posterior,err = naive_bayes(X_train,Y_train,x_test,y_test,train_set,test_set)
    plot_scatter(x_test)
    print('--------------------------------------------------------------------------------')
    print('Part 5')
    post_prob = pd.DataFrame(posterior)
    post_prob[2] = y_test
    roc(post_prob)
    
    # Part 3:
    print('--------------------------------------------------------------------------------')
    print('Part 3')
    sample_size = [10, 20, 50, 100, 300, 500]
    Accuracy = []
    for i in sample_size:
        print('For Sample Size: ', i)
        X_train,Y_train,train_set = generate_train(i,mu1,cov1,mu2,cov2)
        x_test,y_test, test_set = generate_test(i,mu1,cov1,mu2,cov2)
        pred,posterior,err = naive_bayes(X_train,Y_train,x_test,y_test,train_set,test_set)
        Accuracy.append((1-(err/100))*100)
        
    print('Accuracy vs Sample Size')
    print(sample_size,Accuracy)
    plt.scatter(Accuracy,sample_size)
    plt.show()
    
    # Part 4:
    print('--------------------------------------------------------------------------------')
    print('Part 4')
    X_train,Y_train,train_set = generate_train_4(mu1,cov1,mu2,cov2)
    x_test,y_test, test_set = generate_test(500,mu1,cov1,mu2,cov2)
    pred,posterior,err = naive_bayes(X_train,Y_train,x_test,y_test,train_set,test_set)
    print('--------------------------------------------------------------------------------')
    print('Part 5')
    post_prob = pd.DataFrame(posterior)
    post_prob[2] = y_test
    roc(post_prob)
    
    

    
    
    
    