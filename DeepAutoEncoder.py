## DeepAutoEncoder
# And its sklearn version : DeepAutoEncoder

#  11-5-15


import numpy as np
import theano as th
from theano import tensor as T
from numpy import random as rng
 
class AutoEncoder_theano(object):
    def __init__(self, X, list_hidden = [100,50], activation_function = T.nnet.sigmoid):
        """ X is the data, an n x p numpy matrix
        
        list_hidden is the successive hidden_size layers. (list_hidden = [100] is shallow)
        """
        
        assert type(X) is np.ndarray
        assert len(X.shape)==2
        self.X=X
        self.X=th.shared(name='X', value=np.asarray(self.X, 
                         dtype=th.config.floatX),borrow=True)

        
        self.p = X.shape[1]
        self.n = X.shape[0]
        #list_hidden decreasing hidden size
        assert type(list_hidden) is list

        self.list_hidden = list_hidden

        list_length = [self.p] + list_hidden #
        
        l_W = [] #list of W
        l_b_forward = [] #list bias for the forward prop
        l_b_backward = [] #list bias for the backward prop
        for (h_in,h_out) in zip(list_length[:-1], list_length[1:]):
            
            initial_W = np.asarray(rng.uniform(
                     low=-4 * np.sqrt(6. / (h_in + h_out)),
                     high=4 * np.sqrt(6. / (h_in + h_out)),
                     size=(h_in, h_out)), dtype=th.config.floatX)

            l_W.append(th.shared(value=initial_W, name='W'+str(h_in)+'_'+str(h_out), borrow=True))

            l_b_forward.append(th.shared(name='b'+str(h_out), value=np.zeros(shape=(h_out,),
                                dtype=th.config.floatX),borrow=True))
                       
            l_b_backward.append(th.shared(name='b'+str(h_in), value=np.zeros(shape=(h_in,),
                                dtype=th.config.floatX),borrow=True))
                       

        self.l_W = l_W
        self.l_b_forward = l_b_forward
        self.l_b_backward = l_b_backward[::-1]
                       
        self.activation_function=activation_function

                

    def pre_train(self, n_epochs = 50, mini_batch_size = 1, learning_rate = 0.1):
        # train every layers independently (useless?)
        
        index = T.lscalar()
        x=T.matrix('x')
        
        start_time = time.clock()
        
        X_c = self.X     
        for W,b_in,b_out,i in zip(self.l_W, self.l_b_forward, self.l_b_backward[::-1],range(len(self.l_W))):
            
            params = [W, b_in, b_out]
            hidden = self.activation_function(T.dot(x,W) + b_in)
            output = self.activation_function(T.dot(hidden,T.transpose(W)) + b_out)
            
            L = -T.sum(x*T.log(output) + (1-x)*T.log(1-output), axis=1)
            cost=L.mean()       
            updates=[]

            #Return gradient with respect to W, b1, b2. and so on
            gparams = T.grad(cost,params)

            #Create a list of 2 tuples for updates.
            for param, gparam in zip(params, gparams):
                updates.append((param, param-learning_rate*gparam))

            #Train1 given a mini-batch of the data.
            train = th.function(inputs=[index], outputs=[cost], updates=updates,
                                givens={x:X_c[index:index+mini_batch_size,:]})


            import time
            
            for epoch in xrange(n_epochs):
                print("pre_train layer, Epoch:",i,epoch)
                for row in xrange(0,self.n, mini_batch_size):
                    
                    train(row)
            
            
        
            #transfo X by W
            one_step_transfo = th.function(inputs=[], outputs=[hidden],
                             givens={x:X_c})
            X_cc = one_step_transfo()[0]         
            X_c = th.shared(name='Xc', value=np.asarray(X_cc, 
                         dtype=th.config.floatX),borrow=True)
            
        end_time = time.clock()
        print "Average time per epoch=", (end_time-start_time)/n_epochs

    def train(self, n_epochs=20, mini_batch_size=1, learning_rate=0.1):
        # Finetune.
        
        index = T.lscalar()
        x=T.matrix('x')
        params = self.l_W + self.l_b_forward + self.l_b_backward
                  
        hidden = {}
        hidden[0] = x
        hh = 0
        #Forward pass
        for W,b_forward in zip(self.l_W, self.l_b_forward):
            hh += 1
            hidden[hh] = self.activation_function(T.dot(hidden[hh-1],W) + b_forward)
            
                  
        #Backward pass
        for W,b_backward in zip(self.l_W[::-1], self.l_b_backward):
            hh += 1
            hidden[hh] = self.activation_function(T.dot(hidden[hh-1],T.transpose(W)) + b_backward)
            
          
        output = hidden[hh]

        #Use cross-entropy loss.
        L = -T.sum(x*T.log(output) + (1-x)*T.log(1-output), axis=1)
        cost=L.mean()       
        updates=[]
         
        #Return gradient with respect to W, b1, b2. and so on
        gparams = T.grad(cost,params)
         
        #Create a list of 2 tuples for updates.
        for param, gparam in zip(params, gparams):
            updates.append((param, param-learning_rate*gparam))
         
        #Train given a mini-batch of the data.
        train = th.function(inputs=[index], outputs=[cost], updates=updates,
                            givens={x:self.X[index:index+mini_batch_size,:]})
                             
 
        import time
        start_time = time.clock()
        for epoch in xrange(n_epochs):
            print "Epoch:",epoch
            for row in xrange(0,self.n, mini_batch_size):
                train(row)
        end_time = time.clock()
        print "Average time per epoch=", (end_time-start_time)/n_epochs
       
    
    def get_hidden(self,data):
        x=T.dmatrix('x')
        
        hidden = {}
        hidden[0] = x
        hh = 0
        #Forward pass
        for W,b_forward in zip(self.l_W, self.l_b_forward):
            hh += 1
            hidden[hh] = self.activation_function(T.dot(hidden[hh-1],W) + b_forward)
            
                  
        transformed_data = th.function(inputs=[x], outputs=[hidden[hh]])
        return transformed_data(data)[0]
    
    def get_list_hidden(self,data):
        x=T.dmatrix('x')
        
        hidden = {}
        hidden[0] = x
        hh = 0
        l_res = []
        #Forward pass
        for W,b_forward in zip(self.l_W, self.l_b_forward):
            hh += 1
            hidden[hh] = self.activation_function(T.dot(hidden[hh-1],W) + b_forward)
            
            l_res.append(transformed_data = th.function(inputs=[x], outputs=[hidden]))
        
        return l_res
     
    def get_weights(self):
        return [self.l_W[0].get_value(), self.l_b_forward[0].get_value(), self.l_b_backward[0].get_value()]

    
class DeepAutoEncoder():
    
    #scikit learn style
    
    def __init__(self, list_hidden = [100, 50]):
        self.list_hidden = list_hidden
        
    def fit(self, M, n_epochs = 10, mini_batch_size=1, learning_rate=0.1, normalize = True):
        
        if normalize:
            M = [x / np.max(x) for x in M] #get probabilities
            M = np.array(M)
           
        print('start fit with ' + str(n_epochs) + ' epochs.')
        self.AE = DeepAutoEncoder(M,self.list_hidden)
        self.AE.train(n_epochs, mini_batch_size, learning_rate)
        
    def transform(self, M):
        
        return self.AE.get_hidden(M)
    
    def fit_transformM(self,M, n_epochs = 10, mini_batch_size=1, learning_rate=0.1):
        
        self.fit(M, n_epochs, mini_batch_size, learning_rate)
        return self.transform(M)
