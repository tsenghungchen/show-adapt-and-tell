from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from coco_spice.pycocoevalcap.eval import COCOEvalCap
import pdb

class G_pretrained():
    def __init__(self, sess, dataset, conf=None):
        self.sess = sess
        self.batch_size = conf.batch_size
        self.max_iter = conf.max_iter
        self.num_train = dataset.num_train
        self.hidden_size = conf.G_hidden_size   # 512
        self.dict_size = dataset.dict_size
        self.max_words = dataset.max_words
        self.dataset = dataset
        self.load_ckpt = conf.load_ckpt
	self.is_train = conf.is_train
	if self.is_train:
	    self.drop_out_rate = conf.drop_out_rate
	else:
	    self.drop_out_rate = 0

	self.init_lr = conf.init_lr
	self.lr_decay = conf.lr_decay
	self.lr_decay_every = conf.lr_decay_every
	self.ss_ascent = conf.ss_ascent
	self.ss_ascent_every = conf.ss_ascent_every
	self.ss_max = conf.ss_max
        # train pretrained model -> no need to add START_TOKEN
        #                        -> need to add END_TOKEN
	self.img_dims = self.dataset.img_dims
        self.lstm_steps = self.max_words+1
        self.global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)
        #self.optim = tf.train.AdamOptimizer(conf.learning_rate)
        self.checkpoint_dir = conf.checkpoint_dir
        self.START = self.dataset.word2ix[u'<BOS>']
        self.END = self.dataset.word2ix[u'<EOS>']
	self.UNK = self.dataset.word2ix[u'<UNK>']
	self.NOT = self.dataset.word2ix[u'<NOT>']

	self.coins = tf.placeholder('bool', [self.batch_size, self.max_words-1])
	self.images_one = tf.placeholder('float32', [100, self.img_dims])
	self.images = tf.placeholder('float32', [self.batch_size, self.img_dims])
        self.target_sentence = tf.placeholder('int32', [self.batch_size, self.max_words])
        self.mask = tf.placeholder('float32', [self.batch_size, self.max_words])       # mask out the loss
        self.build_Generator(name='G')
	self._predict_words_argmax = []
        self._predict_words_sample = []
        self._predict_words_argmax = self.build_Generator_test(100, self._predict_words_argmax, type='max', name='G')
	self._predict_words_sample = self.build_Generator_test(100, self._predict_words_sample, type='sample', name='G')

	self.lr = tf.Variable(self.init_lr, trainable=False)
        self.optim = tf.train.AdamOptimizer(self.lr)    

        params = tf.trainable_variables()
        self.G_params_dict = {}
        for param in params:
            self.G_params_dict.update({param.name:param})

    def build_Generator_test(self, batch_size=100, predict_words=None, type='max', name='generator'):
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(name):
            tf.get_variable_scope().reuse_variables()
	    with tf.variable_scope("images"):
                # "generator/images"
                images_W = tf.get_variable("images_W", [self.img_dims, self.hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("lstm"):
                # WONT BE CREATED HERE
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True)
#		lstm1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, output_keep_prob=1-self.drop_out_rate)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                # "generator/embedding"
                word_emb_W = tf.get_variable("word_emb_W", [self.dict_size, self.hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("output"):
                # "generator/output"
                output_W = tf.get_variable("output_W", [self.hidden_size, self.dict_size], "float32", random_uniform_init)

	    start_token = tf.constant(self.START, dtype=tf.int32, shape=[batch_size])
            state = lstm1.zero_state(batch_size, 'float32')
            for j in range(self.lstm_steps):
		tf.get_variable_scope().reuse_variables()
		if j == 0:
		    images_emb = tf.matmul(self.images_one, images_W)       # B,H
                    lstm1_in = images_emb
		elif j == 1:
		    with tf.device("/cpu:0"):
		    	lstm1_in = tf.nn.embedding_lookup(word_emb_W, start_token)
                else:
                    with tf.device("/cpu:0"):
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W, sample_words)
                with tf.variable_scope("lstm"):
                    # "generator/lstm"
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())     # output: B,H
		if j > 0:
		    logits = tf.matmul(output, output_W)        # B,D
		    #log_probs = tf.log(tf.nn.softmax(logits))   # B,D
		    # word drawn from the multinomial distribution
		    #sample_words = tf.reshape(tf.multinomial(log_probs,1), [batch_size])
		    sample_words = tf.argmax(logits, 1)
                    predict_words.append(sample_words)
		   
            predict_words = tf.pack(predict_words)
            predict_words = tf.transpose(predict_words, [1,0])
	return predict_words

    def build_Generator(self, name='generator'):
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(name):
	    with tf.variable_scope("images"):
                # "generator/images"
                images_W = tf.get_variable("images_W", [self.img_dims, self.hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("lstm"):
		# "generator/lstm"
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True)
		lstm1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, output_keep_prob=1-self.drop_out_rate)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                # "generator/embedding"
                word_emb_W = tf.get_variable("word_emb_W", [self.dict_size, self.hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("output"):
                # "generator/output"
                output_W = tf.get_variable("output_W", [self.hidden_size, self.dict_size], "float32", random_uniform_init)
	
	    start_token = tf.constant(self.START, dtype=tf.int32, shape=[self.batch_size])
            state = lstm1.zero_state(self.batch_size, 'float32')
            self.pretrained_loss = 0.
            for j in range(self.lstm_steps):
		if j == 0:
		    images_emb = tf.matmul(self.images, images_W)   	# B,H
		    lstm1_in = images_emb
		else:
		    tf.get_variable_scope().reuse_variables()
                    with tf.device("/cpu:0"):
			if j == 1:
			    # <BOS>
			    lstm1_in = tf.nn.embedding_lookup(word_emb_W, start_token)
			else:
			    # schedule sampling
			    word = tf.select(self.coins[:,j-2], self.target_sentence[:,j-2], tf.stop_gradient(word_predict))
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, word)

                with tf.variable_scope("lstm"):
                    # "generator/lstm"
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())     # output: B,H

		if j > 0:
                    logits = tf.matmul(output, output_W)       		# B,D
		    # calculate loss
		    pretrained_loss_t = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.target_sentence[:,j-1])
		    pretrained_loss_t = tf.reduce_sum(tf.mul(pretrained_loss_t, self.mask[:,j-1]))
		    self.pretrained_loss += pretrained_loss_t
		    word_predict = tf.to_int32(tf.argmax(logits, 1))	# B		    


            self.pretrained_loss /= tf.reduce_sum(self.mask)
            self.pretrained_loss_sum = tf.scalar_summary("pretrained_loss", self.pretrained_loss)

    def train(self):
	'''
	Train a caption generator with XE
	with learning rate decay and schedule sampling
	'''

        self.train_op = self.optim.minimize(self.pretrained_loss, global_step=self.global_step)
        self.writer = tf.train.SummaryWriter("./logs/G_pretrained", self.sess.graph)
        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver(var_list=self.G_params_dict, max_to_keep=30)
        try:
            self.saver.restore(self.sess, self.load_ckpt)
            print "[#] Restore", self.load_ckpt
        except:
            print "[#] Fail to restore"

	self.current_lr = self.init_lr
	self.current_ss = 0.
        self.tr_count = 0
        for idx in range(self.max_iter//3000):
	    print "Evaluate source test set..."
	    self.evaluate('test', self.tr_count)
	    print "Evaluate target test set..."
	    self.evaluate('target_test', self.tr_count)
            self.evaluate('train', self.tr_count, eval_algo='max')
            self.evaluate('train', self.tr_count, eval_algo='sample')
            self.save(self.checkpoint_dir, self.tr_count)
            for k in tqdm(range(3000)):
		tgt_text = self.dataset.flickr_caption_sequential_sample(self.batch_size)
                image_feature, target, img_idx = self.dataset.sequential_sample(self.batch_size)
#		dummy_feature = np.zeros(image_feature.shape)
                nonENDs = np.array(map(lambda x: (x != self.NOT).sum(), target))
                mask = np.zeros([self.batch_size, self.max_words])
		tgt_mask = np.zeros([self.batch_size, self.max_words])
                for ind, row in enumerate(mask):
		    # mask out the <BOS>
                    row[0:nonENDs[ind]] = 1

		for ind, row in enumerate(tgt_mask):
		    row[0:nonENDs[ind]] = 1
 	        # schedule sampling condition
	        coins = np.zeros([self.batch_size, self.max_words-1])	
	        for (x,y), value in np.ndenumerate(coins):
		    if y==0:
		        coins[x][y] = True
		    elif np.random.rand() < self.current_ss:
		        coins[x][y] = False
		    else:
		        coins[x][y] = True


                _, loss, summary_str = self.sess.run([self.train_op, self.pretrained_loss, self.pretrained_loss_sum],{
				self.images: image_feature, 
                                self.target_sentence: target,
                                self.mask: mask, 
				self.coins: coins
                                })
#		_, dummy_loss, _ = self.sess.run([self.train_op, self.pretrained_loss, self.pretrained_loss_sum],{
#			self.images: dummy_feature, 
#			self.target_sentence: tgt_text,
#			self.mask: tgt_mask, 
#			self.coins: coins
#			})
	
                self.writer.add_summary(summary_str, self.tr_count)
                self.tr_count += 1

		#if k%1000 == 0:
		#    print " [*] Iter {}, lr={}, ss={}, loss={}".format(self.tr_count, self.current_lr, self.current_ss, loss)

		if idx == 0 and k != 0  and k%1000 == 0:
                    self.evaluate('train', self.tr_count, eval_algo='max')
                    self.evaluate('train', self.tr_count, eval_algo='sample')
                    self.evaluate('test', self.tr_count)
		    self.evaluate('target_test', self.tr_count)
		# schedule sampling
		if (self.tr_count+1)%self.ss_ascent_every == 0 and self.current_ss<self.ss_max:
		    self.current_ss = self.current_ss+self.ss_ascent
	        # learning rate decay
	        if (self.tr_count+1)%self.lr_decay_every == 0:
		    self.current_lr = self.current_lr*self.lr_decay
		    self.lr.assign(self.current_lr).eval()

    def evaluate(self, split, count, eval_algo='max'):
	if not self.is_train:
	    self.saver = tf.train.Saver(var_list=self.G_params_dict)
            self.saver.restore(self.sess, self.load_ckpt)
            print "[#] Restore", self.load_ckpt

        if split == 'test':
            image_feature, image_id, test_annotation = self.dataset.get_source_test_for_eval()
            num_eval = len(test_annotation)
	elif split == 'target_test':
	    image_feature, image_id, test_annotation = self.dataset.get_test_for_eval()
	    num_eval = len(test_annotation)
        elif split == 'train':
            image_feature, img_name =  self.dataset.get_train_for_eval(50000)
            num_eval = 50000

        samples = []
        samples_index = np.full([len(image_feature), self.max_words], self.NOT)
        if eval_algo == 'max': 
            prediction = self._predict_words_argmax
        elif eval_algo == 'sample':
            prediction = self._predict_words_sample
        for i in range(num_eval//100):
            image_feature_one = image_feature[i*100:(i+1)*100]
            predict_words = self.sess.run(prediction,{
                                self.images_one: image_feature_one,
                                })
            for j in range(100):
                samples.append([self.dataset.decode(predict_words[j, :], type='string', remove_END=True)[0]])
                sample_index = self.dataset.decode(predict_words[j, :], type='index', remove_END=False)[0]
                samples_index[i*100+j][:len(sample_index)] = sample_index
                #samples_index.append(self.dataset.decode(predict_words[j, :], type='index')[0])
        if split == 'train':
            # save for negative sample  
            samples = np.asarray(samples)
            sample_dir = os.path.join("./negative_samples", self.dataset.dataset_name+'_'+eval_algo)
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            file_name = "%s_%s" % (self.dataset.dataset_name, str(count))
            #file_name = "%s_%s_%s" % (self.dataset.dataset_name, "final", str(count))
            np.savez(os.path.join(sample_dir, file_name), string=samples, index=samples_index, img_name=img_name)
        else:
            meteor_pd = {}
            meteor_id = []
            for j in range(len(samples)):
                if image_id[j] == 0:
                    break
                meteor_pd[str(int(image_id[j]))] = [{'image_id':str(int(image_id[j])), 'caption':samples[j][0]}]
                meteor_id.append(str(int(image_id[j])))
            #np.savez("result_%s"%str(count), meteor_pd=meteor_pd, meteor_id=meteor_id)
            scorer = COCOEvalCap(test_annotation, meteor_pd, meteor_id)
            scorer.evaluate(verbose=True)		

    def save(self, checkpoint_dir, step):
        model_name = "G_pretrained"
        model_dir = "%s" % (self.dataset.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir, "G_pretrained")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s" % (self.dataset.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

