import tensorflow as tf
from Layer import Layer
class DenseLayer(Layer):
    
    def __call__(self,inputs):
            
        with tf.variable_scope(self.scope) as scope:
            self.check_reuse(scope)
            inputs_shape = inputs.get_shape().as_list()
            output_shape = inputs_shape[:-1] + [self.output_dim]
            if output_shape[0] ==None:
                output_shape[0] = -1

            weight_shape = [inputs_shape[-1],self.output_dim]
            #print "weight shape: ",weight_shape
            weight = tf.get_variable("weight",weight_shape,initializer = tf.contrib.layers.xavier_initializer(inputs.dtype))
            bias =  tf.get_variable("bias",[self.output_dim],initializer = tf.zeros_initializer(inputs.dtype) )
            #print "weight: ",weight.get_shape().as_list()
            #print "bias: ",bias.get_shape().as_list()
            #if len(inputs_shape) >2:
            #    values = tf,batch_matmul(inputs,weight) + bias
            #else:
            values = tf.matmul(tf.reshape(inputs,[-1,inputs_shape[-1]]),weight) + bias
            values = tf.reshape(values,output_shape)
            return values


if __name__ =="__main__":
    l = DenseLayer("Dense",3)
    b = tf.Variable([[[1.0,2],[3,4]],[[5,6],[7,8]]])
    a = tf.placeholder(tf.float32,[10,2])
    c = l(a)
    d = l(b)
