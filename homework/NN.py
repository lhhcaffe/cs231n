# -*- coding: utf-8 -*-
import cPickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
def load_data(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_datasets(file):
    Xtr_batch1 = unpickle(file + 'data_batch_1')['data']
    Xtr_batch2 = unpickle(file + 'data_batch_2')['data']
    Xtr_batch3 = unpickle(file + 'data_batch_3')['data']
    Xtr_batch4 = unpickle(file + 'data_batch_4')['data']
    Xtr_batch5 = unpickle(file + 'data_batch_5')['data']
    Xtr = np.vstack((Xtr_batch1, Xtr_batch2, Xtr_batch3, Xtr_batch4, Xtr_batch5))

    Ytr_batch1 = unpickle(file + 'data_batch_1')['labels']
    Ytr_batch2 = unpickle(file + 'data_batch_2')['labels']
    Ytr_batch3 = unpickle(file + 'data_batch_3')['labels']
    Ytr_batch4 = unpickle(file + 'data_batch_4')['labels']
    Ytr_batch5 = unpickle(file + 'data_batch_5')['labels']
    Ytr_tmp = [Ytr_batch1, Ytr_batch2, Ytr_batch3, Ytr_batch4, Ytr_batch5]
    Ytr = []
    for x in Ytr_tmp:
        Ytr += x

    Xte_sets = unpickle(file + 'test_batch')
    Xte = Xte_sets['data']
    Yte = Xte_sets['labels']

    return Xtr, Ytr, Xte, Yte

def cross_validation(validation_accuracies):

    tmp = np.array(validation_accuracies)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.plot(tmp[:,0], tmp[:,1], 'o')
    plt.xlim(-20, 120)
    plt.show()

class NearestNeighbor():
    def __init__(self):
        pass
    def train(self,X,y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr =y

    def predict(self,X,k):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        #Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        Ypred = np.zeros(num_test, dtype=type(self.ytr))
        k = k


        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)

            #KNN
            k_lable_index = np.argsort(distances)
            k_lable = self.ytr[k_lable_index[0:k]]
            y_pred = np.argmax(np.bincount(k_lable))

            #min_index = np.argmin(distances)   #NN min_index就是预测结果
            #Ypred[i] = self.ytr[y_pred]
            Ypred[i] = y_pred
        return Ypred

if __name__ == '__main__':

    # 加载训练数据集50000*32*32*3
    Xtr, Ytr, Xte, Yte = load_datasets('cifar-10-batches-py/')
    Ytr = np.array(Ytr)
    Yte = np.array(Yte)
    # 从训练数据集分出1000个验证集
    Xval = Xtr[:1000,:]
    Yval = Ytr[:1000]
    Xtr = Xtr[49000:,:]
    Ytr = Ytr[49000:]

    print 'start training-----'

    # find hyperparameters that work best on the validation set
    validation_accuracies = []
    #for k in [1, 3, 5, 10, 20, 50, 100]:
    for k in [1, 3, 5, 10]:
        print Xtr.shape, len(Ytr)
        nn = NearestNeighbor()
        nn.train(Xtr,Ytr)
        print 'start predict-----'
        print "k= %s" %(k)
        Yte_predict = nn.predict(Xte,k)
        acc = np.mean(Yte_predict == Yte)
        print 'accuracy: %f' %(acc)
        validation_accuracies.append((k,acc))
    print validation_accuracies
    #cross_validation(validation_accuracies)
