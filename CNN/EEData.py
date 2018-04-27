from tfnlp.embedding.GloveEmbeddings import *
import random
from scorer.event_scorer import *

class EEData:

    def __init__(self,data_dir,split,label2id_file):
        self.data_dir = data_dir
        self.candidates= [] #(doc_id,sent_id,token_id,offset,length,train_label_id)
        self.word_sents = {}    #(doc_id,sent_id) ->[token0,token1.....]
        self.id_sents = {}  #(doc_id,sent_id) ->([token0_id,token1_id.....],length)
        self.golden = {}    #(doc_id,offset,length) -> [label1,label2,...]
        self.golden_strs = None


        self.id2label = {}
        self.label2id = {}
        self.label2id = {}

        self.max_len = 100
        self.used_embedding = None

        self.load_label2id(label2id_file)
        if split ==None or split == "train":
            self.is_train = True
        else:
            self.is_train = False
        if split:
            self.load(split)
        self.golden_strs = transform_to_score_list(self.golden)

        #print score(self.golden_strs,self.golden_strs[:200])[2]

    def next_batch(self,batch_size = 128):
        batch = self.empty_batch()
        random.shuffle(self.candidates)
        for tri in self.candidates:
            doc_id,sent_id,token_id,offset,length,train_label_id = tri
            if token_id >= self.max_len:
                continue

            sent = self.id_sents[(doc_id,sent_id)]
            batch['sents'].append(sent[0] ) #B*max_len
            batch['keys'].append( (doc_id,sent_id,token_id,offset,length) )
            batch['seq_len'].append( sent[1] )  #B
            batch['position_idx'].append( self.get_relative_position(token_id) )    #B*max_len
            batch['trigger_pos'].append( token_id)

            ctx = []
            if token_id ==0:
                ctx.append( self.used_embedding.get_padding_id() )  #B
            else:
                ctx.append( sent[0][token_id-1] )
            if token_id >= self.max_len -1:
                ctx.append( self.used_embedding.get_padding_id() )  #B
            else:
                ctx.append( sent[0][token_id+1] )

            ctx.append( sent[0][token_id] )#B
            batch['labels'].append(train_label_id)  #B
            batch['lexical_ids'].append(ctx)

            if train_label_id ==9:
                batch['other_cnt'] +=1
                batch['is_negative'].append(0)
            else:
                batch['is_negative'].append(1)

            batch['cnt'] +=1
            if batch['cnt'] >= batch_size:
                yield batch
                batch = self.empty_batch()

        if batch['cnt'] != 0:
            yield batch

    def get_relative_position(self,token_id):
        anchor = [i+self.max_len- token_id -1 for i in xrange(0,self.max_len)]
        return anchor
        
    def sample_negative(self,ratio):
        samples = []
        total_positive = 0
        for tri in self.candidates:
            doc_id,sent_id,token_id,offset,length,train_label_id = tri
            if train_label_id != self.label2id['other']:
                total_positive +=1

        random.shuffle(self.candidates)
        total_negative = 0
        for tri in self.candidates:
            doc_id,sent_id,token_id,offset,length,train_label_id = tri
            if train_label_id != self.label2id['other']:
                samples.append(tri)
            else:
                if total_negative > total_positive * ratio:
                    continue
                total_negative +=1
                samples.append(tri)

        print total_positive,len(samples),len(self.candidates)
        self.candidates = samples


    def empty_batch(self):
        batch = {}

        batch['cnt'] = 0
        batch['keys'] = []
        batch['labels'] = []
        batch['sents'] = []
        batch['seq_len'] = []
        batch['position_idx'] = []
        batch['is_negative'] = []
        #batch['lt'] = []
        #batch['ct'] = []
        #batch['rt'] = []
        batch['lexical_ids'] = []
        batch['other_cnt'] = 0
        batch['trigger_pos'] = []

        return batch


    def candidate_size(self):
        return len(self.candidates)
    def sent_size(self):
        return len(self.word_sents)
    def golden_size(self):
        total =0
        for key in self.golden:
            total += len(self.golden[key])
        return total

    def size(self):
        return self.candidate_size(),self.sent_size(),self.golden_size()

    def load_from_files(self,ids_file,sent_file,golden_file):

        for line in open(ids_file):
            line = line.strip().split("\t")
            if len(line) !=7:
                print line
                print len(line)
                continue
            doc_id,sent_id,token_id,offset,length,train_label,_ = line
            if self.is_train:
                self.candidates.append( (doc_id,int(sent_id),int(token_id),int(offset),int(length),self.label2id[train_label] ) )
            else:
                self.candidates.append( (doc_id,int(sent_id),int(token_id),int(offset),int(length),-1 ) )

        for line in open(sent_file):
            line = line.strip().split("\t")
            doc_id,sent_id,words = line
            words = [w.decode("utf-8").lower() for w in words.split()]
            self.word_sents[( doc_id, int(sent_id) )] = words

        for line in open(golden_file):
            line = line.strip().split("\t")
            key = line[0],int(line[1]),int(line[2])
            if not key in self.golden:
                self.golden[key] = []
            self.golden[key].append(line[4])

    def load_label2id(self,file_name = "data/label2id.dat"):
        self.id2label = {}
        self.label2id = {}
        for line in open(file_name):
            label,i = line.strip().split()
            self.id2label[int(i)] = label
            self.label2id[label] = int(i)

        return len(self.id2label),len(self.label2id)
    
    def load(self,data_split = "train"):
        data_dir = self.data_dir    #"../trigger_data/"

        ids_file = data_dir + data_split + "/"  + data_split + ".ids.dat"
        sent_file = data_dir + data_split + "/" + data_split + ".sents.dat"
        golden_file = data_dir + data_split + "/" +  data_split + ".golden.dat"

        self.load_from_files(ids_file,sent_file,golden_file)

    def translate_sentence(self,word_embeddings,padding = True):
        self.used_embedding = word_embeddings
        self.id_sents = {}
        for sent_key in self.word_sents:
            sent = self.word_sents[sent_key]
            word_ids,length = word_embeddings.words_to_ids(sent,self.max_len,padding)
            self.id_sents[sent_key] = (word_ids,length)
            #print word_ids,length
        return len(self.id_sents),len(self.word_sents)

    def get_word_to_cnt(self,word2cnt = None):
        '''
        return: [(word1,cnt1),(word2,cnt2).....] ordered by cnt
        '''
        if not word2cnt:
            word2cnt = {}
        for sent_key in self.word_sents:
            sent = self.word_sents[sent_key]
            for word in sent:
                if not word in word2cnt:
                    word2cnt[word] = 0
                word2cnt[word] +=1
        
        return word2cnt #sorted(word2cnt.items(),key = lambda x:x[1],reverse = True)



if __name__ =="__main__":
    train_data = EEData("train")
    test_data = EEData("test")
    dev_data = EEData("dev")

    print train_data.size()
    print test_data.size()
    print dev_data.size()

    word2cnt = train_data.get_word_to_cnt()
    word2cnt = test_data.get_word_to_cnt(word2cnt)
    word2cnt = dev_data.get_word_to_cnt(word2cnt)

    embed = GloveEmbeddings("word_word2vec.dat",word2cnt)
    print train_data.translate_sentence(embed)
    print test_data.translate_sentence(embed)
    print dev_data.translate_sentence(embed)

    for batch in train_data.next_batch(170):
        for lex_ids,label in zip(batch['lexical_ids'], batch['labels']):
            if label !=9:
                print lex_ids,train_data.id2label[label]

    #for batch in train_data.next_batch(128):
    #    print batch['cnt']
