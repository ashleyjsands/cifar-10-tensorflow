import unittest
import numpy as np
import tensorflow as tf

import neural_network

class NeuralNetworkTests(unittest.TestCase):

    def testSimpleDepthConcat(self):
        with tf.Session() as session:
            t1 = tf.Variable(np.array([[[[1]]]]).reshape((1, 1, 1, 1)))
            t2 = tf.Variable(np.arange(1 * 3 * 3 * 1).reshape((1, 3, 3, 1)))
            depth_concat_tensor = neural_network.depth_concat([t1, t2])
            tf.initialize_all_variables().run()
            self.assertEqual(depth_concat_tensor.get_shape().as_list(), [1, 3, 3, 2])
            print "depthconcat shape %s" % depth_concat_tensor.get_shape().as_list()
            expected_tensor = [[[[0, 0], [0, 1], [0, 2]], [[0, 3], [1, 4], [0, 5]], [[0, 6], [0, 7], [0, 8]]]]
            self.assertEqual(depth_concat_tensor.eval().tolist(), expected_tensor)

if __name__ == "__main__":
    unittest.main()
