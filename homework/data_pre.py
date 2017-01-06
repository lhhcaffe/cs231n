import numpy as np
import cPickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def load_data_test(file):
    Xtr_batch1 = unpickle(file + 'data_batch_1')['data']
    Xtr_batch2 = unpickle(file + 'data_batch_2')['data']
    Xtr_batch3 = unpickle(file + 'data_batch_3')['data']
    Xtr_batch4 = unpickle(file + 'data_batch_4')['data']
    Xtr_batch5 = unpickle(file + 'data_batch_5')['data']
    Xtr = np.vstack((Xtr_batch1,Xtr_batch2,Xtr_batch3,Xtr_batch4,Xtr_batch5))


    Ytr_batch1 = unpickle(file + 'data_batch_1')['labels']
    Ytr_batch2 = unpickle(file + 'data_batch_2')['labels']
    Ytr_batch3 = unpickle(file + 'data_batch_3')['labels']
    Ytr_batch4 = unpickle(file + 'data_batch_4')['labels']
    Ytr_batch5 = unpickle(file + 'data_batch_5')['labels']
    Ytr_tmp = [Ytr_batch1,Ytr_batch2,Ytr_batch3,Ytr_batch4,Ytr_batch5]
    Ytr = []
    for x in Ytr_tmp:
        Ytr += x

    Xte_sets = unpickle(file + 'test_batch')
    Xte = Xte_sets['data']
    Yte = Xte_sets['labels']

    return Xtr, Ytr, Xte, Yte


if __name__ == '__main__':
    Xtr, Ytr, Xte, Yte = load_data_test('cifar-10-batches-py/')

    print Xtr.shape,len(Ytr),Xte.shape,len(Yte)
    #print  Xtr,Xtr.shape

    #labels = unpickle('cifar-10-batches-py/batches.meta')
    #print data_sets['data']
    #print 'num_cases_per_batch is %d' % (labels['num_cases_per_batch'], data_sets['data'].shape, len(data_sets['labels']))
    #print 'num_cases_per_batch is %d,label_names are %s,num_vis %d' %(labels['num_cases_per_batch'],labels['label_names'],labels['num_vis'])

