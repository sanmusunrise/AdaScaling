import tensorflow as tf
from tfnlp.embedding.GloveEmbeddings import *
from tfnlp.layer.Conv1DLayer import *
from tfnlp.layer.NegativeMaskLayer import *
from tfnlp.layer.MaskLayer import *
from tfnlp.layer.DenseLayer import *
from tfnlp.layer.MaskedSoftmaxLayer import *


def weight_loss(logits,label_ids,positive_idx = None,negative_idx = None,correct_class_weight =None,wrong_confusion_matrix = None, label_size = 19):
    class_weight = [1.0] * label_size
    class_weight[9] = 0.1
    class_weight = tf.Variable(class_weight,dtype = tf.float32,name = "cls_weight",trainable = False)
    
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    loss_weight = tf.gather(class_weight,label_ids)
    
    loss = - loss_weight * tf.log(golden_prob + 1e-8)
    return loss
    
def f1_reweight_loss(logits,label_ids,positive_idx,negative_idx,correct_class_weight =None,wrong_confusion_matrix = None, label_size = 19):
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    m = tf.reduce_sum(positive_idx)
    n = tf.reduce_sum(negative_idx)
    p1 = tf.reduce_sum(positive_idx * golden_prob)
    p2 = tf.reduce_sum(negative_idx * golden_prob)
    beta2 = 1
    neg_weight = p1 / ((beta2 *m)+n-p2 + 1e-8)
    all_one = tf.ones(tf.shape(golden_prob))
    loss_weight = all_one * positive_idx + all_one * neg_weight * negative_idx
    
    loss = - loss_weight * tf.log(golden_prob +1e-8)
    return loss

def f1_reweight_stopped(logits,label_ids,positive_idx,negative_idx,correct_class_weight =None,wrong_confusion_matrix = None, label_size = 19):
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    m = tf.reduce_sum(positive_idx)
    n = tf.reduce_sum(negative_idx)
    p1 = tf.reduce_sum(positive_idx * golden_prob)
    p2 = tf.reduce_sum(negative_idx * golden_prob)
    neg_weight = p1 / (m+n-p2 + 1e-8)
    all_one = tf.ones(tf.shape(golden_prob))
    loss_weight = all_one * positive_idx + all_one * neg_weight * negative_idx
    loss_weight_stopped = tf.stop_gradient(loss_weight)
    loss = - loss_weight_stopped * tf.log(golden_prob +1e-8)
    return loss
    


def likelihood_loss(logits,label_ids,positive_idx = None,negative_idx = None,correct_class_weight =None,wrong_confusion_matrix = None, label_size = 19):
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    loss = - tf.log(golden_prob +1e-8)
    return loss

def focal_loss(logits,label_ids,positive_idx = None,negative_idx = None,correct_class_weight =None,wrong_confusion_matrix = None, label_size = 19):
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    
    loss_weight = tf.pow(1.03 - golden_prob,2)
    loss = - loss_weight * tf.log(golden_prob + 1e-8)
    return loss