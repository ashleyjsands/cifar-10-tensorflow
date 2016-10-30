import numpy as np
import tensorflow as tf

def analyse_weights(model):
    with tf.Session(graph=model.graph) as session:
        init_op = tf.initialize_all_variables()
        #saver = tf.train.Saver()
        session.run(init_op) # All variables must be initialised before the saver potentionally restores the checkpoint below.
        print("Initialized")
        model_name = 'model'
        ckpt = tf.train.get_checkpoint_state("./")
        if ckpt and ckpt.model_checkpoint_path:
            model.saver.restore(session, ckpt.model_checkpoint_path)
            print("model loaded")
        else:
            raise Error("No checkpoint.")
        for weights in model.layer_weights:
            array = weights.eval()
            #print(array)
            print("min:", np.min(array), "max:", np.max(array), "std:", np.std(array))
            num_of_zeros = 0
            for x in  np.nditer(array):
                if x == 0:
                    num_of_zeros += 1
            print("percentage of weights are zero:", num_of_zeros / array.size)
