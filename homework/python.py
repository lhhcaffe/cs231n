import numpy as np
from scipy.spatial.distance import pdist,squareform
import matplotlib.pyplot as plt
from scipy.misc import imread,imresize

""" string
hello = 'hello'
world = "world"
print hello
print len(hello)

hw = hello + '' + world
print hw
hw2015 = '%s %s %d' %(hello, world,2015)
print hw2015

print hello.capitalize()
print hello.upper()
print hello.rjust(7)
print hello.center(7)
print hello.replace('o','mm')
"""


""" list
    xs = [3,1,2]
    print xs,xs[-1]
    xs[2] = 'foo'
    print xs
    xs.append('bar')
    print xs
    x = xs.pop()
    print xs
    print len(xs)
"""
""" list op
    nums = range(5)
    print nums
    print nums[1:-1]
    nums[2:4] = [8,9]
    print nums

    animals = ["cat","dog","monkey"]
    for animal in animals:
    print animal

    for idex,animal in enumerate(animals):
    print '#%d,%s' % (idex+1,animal)

    #for iteration
    nums = range(5)
    print nums
    squares = []
    for x in nums:
        squares.append(x ** 2)
    print squares
    #list comprehesion
    nums = range(5)
    print nums
    squares = [x ** 2 for x in nums]
    print squares
    even_squares = [x ** 2 for x in nums if x % 2 == 0]
    print even_squares
"""

""" dict
    d = {'cat':'cute','dog':'furry'}
    print d['cat']
    print 'cats' in d
    d['fish'] = 'wet'
    print d.get('monkey','N/A')
    print d.get('cat','N/A')
    d = {'person':2,'dog':4,'spider':8}
    for animal in d:
        legs = d[animal]
        print 'A %s has %d legs' % (animal,legs)
    for animal,legs in d.iteritems():
        print 'A %s has %d legs' % (animal, legs)
    nums = [0,1,2,3,4]
    even_num_to_squrar = {x:x ** 2 for x in nums if x % 2 == 0}
    print even_num_to_squrar


    d = {(x,x+1): x for x in range(10)}
    t = (5,6)
    print d[t],type(t),d[(1,2)]
"""

""" numpy
    a = np.array([1,2,3])
    print a,type(a),a.shape
    b = np.array([[1,2,3],[4,5,6]])
    print b,type(b),b.shape
    a = np.zeros((2,2))
    b = np.ones((1,2))
    c = np.full((2,2),7)
    d = np.eye(2)
    e = np.random.random((2,3))
    print e

    a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    b = a[:2,1:3]
    x = np.array([1,2],dtype=np.int64)
    b[0,0] = 77
    print a,b,x.dtype

    a = np.array([[1,2],[3,4]],dtype=np.float64)
    b = np.array([[5,6],[7,8]],dtype=np.float64)
    print a + b
    print np.add(a,b)
    print np.subtract(b,a)
    print np.multiply(b,a)
    print np.sqrt(a)

    #numpy op
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])

    v = np.array([9,10])
    w = np.array([11,12])

    print np.sum(x), np.sum(x,axis=0),np.sum(x,axis=1)
    print x.dot(v)
    print x.dot(y)

    # numpy broadcasting
    x = np.array([[1,2,3],[4,5,6],[7,8,9]])
    v = np.array([1,0,1])
    y = np.empty_like(x)

    vv = np.tile(v,(3,1))
    #for i in range(3):
    #    y[i,:] = x[i,:] + v
    print vv
    y = x + vv
    print y,x+v

    v = np.array([1,2,3])
    w = np.array([4,5])
    xb = np.array([[1,2,3],[5,6,7]])
    print xb,xb.shape,np.reshape(xb,(3,2))
"""


""" numpy
x = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print x
    d =squareform(pdist(x,'euclidean'))
    print d
"""


""" matplotlib
    x = np.arange(0,3 * np.pi,0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    plt.plot(x,y_sin)
    plt.plot(x,y_cos)
    plt.xlabel('x axis label')
    plt.ylabel('y axis label')
    plt.title('Sine and Cosine')
    plt.legend(['Sine','Cosine'])
    plt.show()

    #sub plot
    x = np.arange(0, 3 * np.pi, 0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    plt.subplot(2,1,1)
    plt.plot(x, y_sin)
    plt.title('Sine')

    plt.subplot(2,1,2)
    plt.plot(x, y_cos)
    plt.title('Cosine')

    plt.show()

    img = imread('zzy.jpg')
    img_tinted = img * [1,1,1]

    plt.subplot(1,2,1)
    plt.imshow(img)

    plt.subplot(1,2,2)
    plt.imshow(img_tinted)
    print img_tinted.shape,img.shape


    plt.show()
"""
"""k = 5
x = np.array([3, 1, 2, 2, 2, 3, 1,10,2,2,2,0,0,0])
index = np.argsort(x)

tmp = x[index[0:k]]
a = np.bincount(tmp)
result = np.argmax(np.bincount(t

print x,index,tmp,a,result
"""


validation_accuracies = [(1, 0.18770000000000001), (3, 0.16819999999999999), (5, 0.17199999999999999), (10, 0.1774)]


a = np.array(validation_accuracies)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')

plt.plot(a[:,0],a[:,1],'o')

plt.xlim(-20,120)
plt.show()

print a[:,0],a[:,1]
















































#plt.plot(validation_accuracies[1])
#plt.show()