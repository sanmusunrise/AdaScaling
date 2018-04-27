import tensorflow as tf
from tfnlp.embedding.GloveEmbeddings import *
from tfnlp.layer.Conv1DLayer import *
from tfnlp.layer.NegativeMaskLayer import *
from tfnlp.layer.MaskLayer import *
from EEData import *
from tfnlp.layer.DenseLayer import *
from tfnlp.layer.MaskedSoftmaxLayer import *
from scorer.event_scorer import *
import numpy as np
import shelve
import loss as loss_functions
import sys
from Configure import *

class CNNModel:

    def __init__(self,conf):
    
        self.configure = conf
        
        self.data_dir = None
        self.max_len = None
        self.position_embedding_dim = None
        self.word_embedding_dim = None
        self.feature_map_size = None
        self.label_size = None
        self.batch_size = None
        self.sample_negative = None
        self.negative_ratio =None
        self.word_embeddings_file = None
        self.loss_function_name = None
        
        self.model_save_path = None
        self.dev_rst_path = None
        
        self.best_dev_epoch = 0
        self.best_dev_f1 = 0.0

        self.placeholders = {}
        self.variables = {}
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        #self.sess = create_session()

        self.epoch = 0
        self.word2cnt = None
        self.word_embeddings = None

        self.set_configure()
        self.loss_function = eval(self.loss_function_name)
         
        self.wrong_confusion_matrix = [[0.0] * self.label_size] *self.label_size
        self.correct_class_weight = [1.0] * self.label_size
        
    
    def set_configure(self):
        for key in self.configure.confs:
            self.__dict__[key] = self.configure[key]
            
    def save_parameters(self):
        f = shelve.open(self.model_save_path + "model_obj.shv")
        f['max_len'] = self.max_len
        f['position_embedding_dim'] = self.position_embedding_dim
        f['word_embedding_dim'] = self.word_embedding_dim
        f['feature_map_size'] = self.feature_map_size
        f['label_size'] = self.label_size
        f['model_save_path'] = self.model_save_path
        f['dev_rst_path'] = self.dev_rst_path
        f['best_dev_epoch'] = self.best_dev_epoch
        f['best_dev_f1'] = self.best_dev_f1
        f['batch_size'] = self.batch_size
        f['epoch'] = self.epoch
        f['word2cnt'] = self.word2cnt
        f['word_embeddings'] = self.word_embeddings
        f.close()
        

    def restore_parameters(self):
        f = shelve.open(self.model_save_path + "model_obj.shv")
        self.max_len = f['max_len']
        self.position_embedding_dim = f['position_embedding_dim']
        self.word_embedding_dim = f['word_embedding_dim']
        self.feature_map_size = f['feature_map_size']
        self.label_size = f['label_size']
        self.model_save_path = f['model_save_path']
        self.dev_rst_path = f['dev_rst_path']
        self.best_dev_epoch = f['best_dev_epoch']
        self.best_dev_f1 = f['best_dev_f1']
        self.batch_size = f['batch_size']
        self.epoch = f['epoch']
        self.word2cnt = f['word2cnt']
        self.word_embeddings = f['word_embeddings']
        f.close()
        
        
    def load_previous_model(self):
        self.restore_parameters()
        if self.best_dev_epoch !=0:
            self.create_model()
            self.restore_tf_model(self.best_dev_epoch)
            print "load trained model at epoch: ",self.epoch,self.best_dev_epoch
            print "current model dev F1: ",self.best_dev_f1
        else:
            print "No previous model available."
            exit(-1)
        

    def create_model(self):
        word_embeddings = tf.Variable( self.word_embeddings.get_embeddings() ,dtype = tf.float32,name = "word_embeddings" )
        position_embeddings = tf.Variable( tf.random_uniform([self.max_len *2, self.position_embedding_dim], -1,1), dtype = tf.float32,name = "position_embeddings")
        
        correct_class_weight = tf.placeholder(tf.float32,[self.label_size])
        #correct_class_weight = tf.Variable([1.0]*self.label_size,dtype = tf.float32,trainable = False)
        wrong_confusion_matrix = tf.placeholder(tf.float32,[self.label_size,self.label_size])
        #wrong_confusion_matrix = tf.zeros([self.label_size,self.label_size],dtype = tf.float32)
        word_ids = tf.placeholder(tf.int32, [None,self.max_len])
        position_ids = tf.placeholder(tf.int32, [None,self.max_len])
        label_ids = tf.placeholder(tf.int32, [None])
        lexical_ids = tf.placeholder(tf.int32,[None,3])
        is_negative = tf.placeholder(tf.float32,[None])
        is_train = tf.placeholder(tf.bool,[])

        seq_len = tf.placeholder(tf.int32, [None])
        trigger_position = tf.placeholder(tf.int32,[None])

        embed_word = self.embed(word_embeddings,word_ids)   #B*T*d1
        embed_position = self.embed(position_embeddings,position_ids)   #b*t*d2
        embed_lexicals = self.embed(word_embeddings,lexical_ids)
        lexical_features = tf.reshape(embed_lexicals,shape = [-1,3* self.word_embedding_dim])
    
        concat_embedding = tf.concat([embed_word,embed_position],axis = 2) #B*T*(d1+d2)
        
        input_mask_layer = MaskLayer("input_mask")
        input_mask_layer.set_extra_parameters({"mask_value":0})
        masked_concat_embedding = input_mask_layer(concat_embedding,seq_len)

        conv_layer = Conv1DLayer("conv1d",self.feature_map_size)

        feature_maps = conv_layer(masked_concat_embedding) #B*T*feature_map_size
        feature_maps = tf.tanh(feature_maps)

        left_trigger_mask = MaskLayer("left_feature_mask")
        right_trigger_mask = MaskLayer("right_feature_mask")

        left_trigger_mask.set_extra_parameters({"mask_value":-2,"mask_from_right":False})
        right_trigger_mask.set_extra_parameters({"mask_value":-2,"mask_from_right":True})

        left_feature_map = left_trigger_mask(feature_maps,trigger_position)
        right_feature_map = right_trigger_mask(feature_maps,trigger_position)

        left_pooled = tf.reduce_max(left_feature_map,axis = 1)
        right_pooled = tf.reduce_max(right_feature_map,axis = 1)
        max_pooled_feature_maps = tf.concat( [left_pooled,right_pooled] , axis = 1 )
        
        features = tf.concat([max_pooled_feature_maps,lexical_features],axis =1)
        
        #dropout_features = features
        dropout_features = tf.layers.dropout(features,training = is_train)
        
        output_layer = DenseLayer("output_layer",self.label_size)
        logits = output_layer(dropout_features)
        
        positive_idx = is_negative
        negative_idx = 1- is_negative

        loss_every_token = self.loss_function(logits,label_ids,positive_idx,negative_idx, correct_class_weight,wrong_confusion_matrix,self.label_size)
        
        loss = tf.reduce_mean(loss_every_token)
        train_step = tf.train.AdadeltaOptimizer(learning_rate=1,epsilon=1e-06).minimize(loss)

        self.variables['loss'] = loss
        self.variables['train_step'] = train_step
        self.variables['output'] = logits

        self.placeholders['sents'] = word_ids
        self.placeholders['seq_len'] = seq_len
        self.placeholders['position_idx'] = position_ids
        self.placeholders['labels'] = label_ids
        self.placeholders['lexical_ids'] = lexical_ids
        self.placeholders['trigger_pos'] = trigger_position
        self.placeholders['is_train'] = is_train
        self.placeholders['is_negative'] = is_negative

        self.placeholders['correct_class_weight'] = correct_class_weight
        self.placeholders['wrong_confusion_matrix'] = wrong_confusion_matrix

    def save_tf_model(self):
        saver = tf.train.Saver()
        saver.save(self.sess, "model/model.ckpt",global_step=self.epoch)


    def restore_tf_model(self,epoch):
        self.epoch = epoch
        saver = tf.train.Saver()
        saver.restore(self.sess,"model/model.ckpt-" + str(epoch))

    def train_model(self):
       
        #self.restore_model(40)
        init = tf.global_variables_initializer()
        sess = self.sess
        if self.epoch == 0:
            sess.run(init)
        while(1):
            total_instance = 0.0
            correct_typed_instance = 0.0
            correct_instance = 0.0
            total_typed_instance = 0.0
            self.epoch +=1
            epoch = self.epoch
            epoch_loss = 0.0
            for batch in self.train_data.next_batch(self.batch_size):
                batch['is_train'] = True
                batch['correct_class_weight'] = self.correct_class_weight
                batch['wrong_confusion_matrix'] = self.wrong_confusion_matrix
                feed_dict = {}
                for key in self.placeholders:
                    feed_dict[self.placeholders[key]] = batch[key]
                #feed_dict[self.placeholders['is_train']] = True
                loss_var = self.variables['loss']
                train_step_var = self.variables['train_step']
                probs = self.variables['output']
                loss,probs,_ = sess.run([loss_var,probs,train_step_var],feed_dict = feed_dict)
                epoch_loss += loss * self.batch_size
                
                for label,prob in zip(batch['labels'],probs):

                    if label ==np.argmax(prob):
                        correct_instance +=1
                    total_instance +=1
                    
                    if self.train_data.id2label[label] == "other":
                        continue
                    if label == np.argmax(prob):
                        correct_typed_instance +=1
                    total_typed_instance +=1

            print "epoch ", epoch, "loss: ", epoch_loss
            if epoch !=1:
                print
                print "Dev data performance at epoch %d" % (epoch)
                results = self.decode(self.dev_data,epoch,"dev")
                s_p,s_r,s_f = results[0]
                t_p,t_r,t_f = results[1]
                print "Span result: ",s_p,s_r,s_f
                print "Type Result: ",t_p,t_r,t_f
                print
                print "Test data performance at epoch %d" %(epoch)
                results = self.decode(self.test_data,epoch,"test")
                s_p,s_r,s_f = results[0]
                t_p,t_r,t_f = results[1]
                print "Span result: ",s_p,s_r,s_f
                print "Type Result: ",t_p,t_r,t_f
                print "---------------------"
            sys.stdout.flush()       

            if self.epoch > 25:
                print "Finish Training"
                break
    def embed(self,params,ids):
        return tf.nn.embedding_lookup(params,ids)

    
    def decode(self,corpus,epoch = None,prefix = "typed",do_eval = True):
        sess = self.sess
    
        key2label = {}
        key2label_output = {}
        key2label_confusion = {}
        precision = recall = f1 = 0.0
        for batch in corpus.next_batch(self.batch_size):
            feed_dict = {}
            #feed_dict[self.placeholders['is_train']] = False
            batch['is_train'] = False
            batch['correct_class_weight'] = self.correct_class_weight
            batch['wrong_confusion_matrix'] = self.wrong_confusion_matrix
            for key in self.placeholders:
                if key !='labels':
                    feed_dict[self.placeholders[key]] = batch[key]
            probs = self.variables['output']
            probs = sess.run(probs,feed_dict = feed_dict)
        
            for key,prob in zip(batch['keys'],probs):
                
                #prob = np.exp(prob) / np.sum(np.exp(prob))
                #other_id = corpus.label2id['other']
                #if prob[other_id] < 0.5:
                #    prob[other_id] = 0.0

                top_3 = np.argsort(prob).tolist()[-3:]
                top_3.reverse()
                top_3 = [(corpus.id2label[k],prob[k]) for k in top_3]
                
                key = (key[0],key[1],key[2],key[3],key[4])
                key_g = (key[0],key[3],key[4])
                
                output_label = top_3[0][0]
                key2label_confusion[key_g] = output_label

                                
                if output_label =="other":
                    continue
                
                key2label[key_g] = output_label
                key2label_output[key] = top_3
        if do_eval:
            for key in key2label:
                key2label[key] = [key2label[key]]
            
            output_strs = transform_to_score_list(key2label)
            eval_rst = score(corpus.golden_strs,output_strs)[2]
            r1 = eval_rst['plain']['micro']
            r2 = eval_rst['mention_type']['micro']
            (p_span,r_span,f_span),(p_typed,r_typed,f_typed)  = r1,r2
            
            #(p_span,r_span,f_span),(p_typed,r_typed,f_typed) = self.score(corpus,key2label)

        if epoch:
            self.output_result(key2label_output,epoch,prefix)
        
        return (p_span,r_span,f_span),(p_typed,r_typed,f_typed)
        
    def score(self,data,rst):
        tp_span = 0.0
        tp_typed = 0.0
        if len(rst) <1:
            return(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        for doc_id, offset, length in rst:
            label = rst[(doc_id,offset,length)]
            key = (doc_id, offset, length)
            if key in data.golden:
                tp_span += 1
                if label in data.golden[key]:
                    tp_typed += 1
        if tp_span <1 or tp_typed <1:
            return (0.0,0.0,0.0),(0.0,0.0,0.0)
        p_span = tp_span / len(rst)
        r_span = tp_span / len(data.golden)
        f_span = p_span * r_span * 2 / (p_span + r_span)

        p_typed = tp_typed / len(rst)
        r_typed = tp_typed / len(data.golden)
        f_typed = p_typed * r_typed * 2 / (p_typed + r_typed)

        return (p_span, r_span, f_span), (p_typed, r_typed, f_typed)

    
    def output_result(self,key2label,epoch,prefix):
        output = open("tmp/" + prefix + "_result." + str(epoch) + ".rst","w")
        for key in key2label:
            output.write( str(key[0]) + "\t" + str(key[1]) + "\t" + str(key[2]) + "\t" + str(key[3]) + "\t" + str(key[4]) + "\t "  )
            for label,prob in key2label[key]:
                output.write(label + "\t" + str(prob) + "\t")
            output.write("\n")
        output.close()
        
    def load_train_dev_data_and_embeddings(self):

        self.train_data = EEData(self.data_dir,"train",self.data_dir +"label2id.dat")
        self.dev_data = EEData(self.data_dir,"dev",self.data_dir +"label2id.dat")
        self.test_data = EEData(self.data_dir,"test",self.data_dir +"label2id.dat")


        if self.sample_negative:
            self.train_data.sample_negative(self.negative_ratio)
        
        if not self.word2cnt:
            self.word2cnt = self.train_data.get_word_to_cnt()
        
        if not self.word_embeddings:
            self.load_init_word_embeddings()

        self.train_data.translate_sentence(self.word_embeddings)
        self.dev_data.translate_sentence(self.word_embeddings )
        self.test_data.translate_sentence(self.word_embeddings )
    def load_init_word_embeddings(self):
        self.word_embeddings = GloveEmbeddings(self.word_embeddings_file,self.word2cnt)

    def do_test(self):
        self.test_data = EEData("test")

        self.test_data.translate_sentence(self.word_embeddings)
        print "start to decode on test set"
        (p_span,r_span,f_span),(p_typed,r_typed,f_typed) = self.decode(self.test_data,epoch = self.epoch,prefix = "test",do_eval = True)
        print (p_span,r_span,f_span),(p_typed,r_typed,f_typed)

if __name__ =="__main__":
    
    confs = Configure(sys.argv[1])
    model = CNNModel(confs)
    model.load_train_dev_data_and_embeddings()
    model.create_model()
    #model.save_parameters()
    model.train_model()
