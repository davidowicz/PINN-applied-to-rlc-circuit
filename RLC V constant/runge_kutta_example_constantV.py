import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import RNN, Layer
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.optimizers import RMSprop
from tensorflow import float32, concat, convert_to_tensor, linalg
from tensorflow.keras.callbacks import ModelCheckpoint

class RungeKuttaIntegratorCell(Layer):
    def __init__(self, m, c, k, dt, initial_state, **kwargs):
        super(RungeKuttaIntegratorCell, self).__init__(**kwargs)
        self.M = m
        self._c   = c
        self.K    = k
        self.initial_state = initial_state
        self.state_size    = 2*len(m)
        self.A  = np.array([0., 0.5, 0.5, 1.0], dtype='float32')
        self.B  = np.array([[1/6, 2/6, 2/6, 1/6]], dtype='float32')
        self.dt = dt

    def build(self, input_shape, **kwargs):
        self.kernel = self.add_weight("C", shape = self._c.shape, trainable = True, initializer = lambda shape, dtype: self._c, **kwargs)
        self.built  = True

    def call(self, inputs, states):
        C    = self.kernel
        
        
        y    = states[0][:,0]     
        ydot = states[0][:,1:]
        
        
        yddoti = self._fun(self.M, self.K, C, inputs, y, ydot)
        yi     = y + self.A[0] * ydot * self.dt
        ydoti  = ydot + self.A[0] * yddoti * self.dt
        fn     = self._fun(self.M, self.K, C, inputs, yi, ydoti)
        for j in range(1,4):
            yn    = y + self.A[j] * ydot * self.dt
            ydotn = ydot + self.A[j] * yddoti * self.dt
            ydoti = concat([ydoti, ydotn], axis=0)
            fn    = concat([fn, self._fun(self.M, self.K, C, inputs, yn, ydotn)], axis=0)

        y    = y + linalg.matmul(self.B, ydoti) * self.dt
        ydot = ydot + linalg.matmul(self.B, fn) * self.dt
        return y, [concat(([y, ydot]), axis=-1)]

    def _fun(self, M, K, C, u, y, ydot):
        return (u-C*ydot-K*y)/M

    #def _getCKmatrix(self, a):
        #return convert_to_tensor([[a[0]+a[1],-a[1]],[-a[1],a[1]+a[2]]], dtype=float32)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.initial_state


def create_model(m, c, k, dt, initial_state, batch_input_shape, return_sequences = True, unroll = False):
    rkCell = RungeKuttaIntegratorCell(m=m, c=c, k=k, dt=dt, initial_state=initial_state)
    PINN   = RNN(cell=rkCell, batch_input_shape=batch_input_shape, return_sequences=return_sequences, return_state=False, unroll=unroll)
    model  = Sequential()
    model.add(PINN)
    model.compile(loss='mse', optimizer=RMSprop(0.1), metrics=['mae'])
    return model

if __name__ == "__main__":
    # masses, spring coefficients, and damping coefficients
    m = np.array([1.0], dtype='float32')
    c = np.array([20.0], dtype='float32') # initial guess
    k = np.array([9.0], dtype='float32')

    # data
    df = pd.read_csv('./data/dataV_noise2_35_and_7_85.csv')
    t  = df[['t']].values
    dt = (t[1] - t[0])[0]
    utrain = df[['u']].values[np.newaxis, :, :]
    ytrain = df[['yT']].values[np.newaxis, :, :]

    # fitting physics-informed neural network
    
    mckp = ModelCheckpoint(filepath="./savedmodels/cp.ckpt", monitor='loss', verbose=1,
                           save_best_only=True, mode='min', save_weights_only=True)
    # initial_state = np.zeros((1,2*len(m)), dtype='float32')
    initial_state = np.array([[0.0,0.0]], dtype='float32' )
    model = create_model(m, c, k, dt, initial_state=initial_state, batch_input_shape=utrain.shape)
    yPred_before = model.predict_on_batch(utrain)[0, :,:]
    history = model.fit(utrain, ytrain, epochs=120, steps_per_epoch=1, verbose=1, callbacks=[mckp])
    #model.fit(utrain, ytrain, epochs=100, steps_per_epoch=1, verbose=2)
    #model.fit(
        #utrain, 
        #ytrain, 
        #epochs=100, 
        #steps_per_epoch=1, 
        #verbose=2, 
        #callbacks = [
            #callbacks.EarlyStopping(monitor='loss', mode='min', patience=10, restore_best_weights=True),
        #]
    #)
    yPred = model.predict_on_batch(utrain)[0, :,:]

    # plotting prediction results
    plt.plot(t, ytrain[0, :,:], 'gray')
    plt.plot(t, yPred_before[:, :], 'r', label='before training')
    plt.plot(t, yPred[:, :], 'b', label='after training')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid('on')
    plt.legend()
    plt.show()
    
    #Predicted parameters
    weights = model.trainable_weights[0].numpy()
    weights = weights[np.newaxis, :]
    
    

    #Loss plot
    plt.plot(np.array(history.history['loss']))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(which='both')
    plt.show()
    