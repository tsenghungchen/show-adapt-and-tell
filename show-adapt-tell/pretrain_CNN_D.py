from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from highway import *
import pdb

class D_pretrained():
    def __init__(self, sess, dataset, negative_dataset, D_info, conf=None, l2_reg_lambda=0.2):

        self.sess = sess
        self.batch_size = conf.batch_size
        self.max_iter = conf.max_iter
        self.num_train = dataset.num_train
        self.hidden_size = conf.D_hidden_size     # 512
        self.dict_size = dataset.dict_size
        self.max_words = dataset.max_words
        self.dataset = dataset
	self.negative_dataset = negative_dataset
	self.checkpoint_dir = conf.checkpoint_dir
	self.global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)
        self.optim = tf.train.AdamOptimizer(conf.learning_rate)
	self.filter_sizes = D_info['filter_sizes']
	self.num_filters = D_info['num_filters']
	self.num_filters_total = sum(self.num_filters)
	self.num_classes = D_info['num_classes']
	self.l2_reg_lambda = l2_reg_lambda
	self.START = self.dataset.word2ix[u'<BOS>']
        self.END = self.dataset.word2ix[u'<EOS>']
        self.UNK = self.dataset.word2ix[u'<UNK>']
        self.NOT = self.dataset.word2ix[u'<NOT>']
	# placeholder
	self.text = tf.placeholder(tf.int32, [None, self.max_words], name="text")
        self.label = tf.placeholder(tf.float32, [None, self.num_classes], name="label")
	self.images = tf.placeholder(tf.float32, [None, self.dataset.img_dims], name="images")

	self.loss, self.pred = self.build_Discriminator(self.images, self.text, self.label, name='D')
	self.loss_sum = tf.scalar_summary("loss", self.loss)

	params = tf.trainable_variables()
        self.D_params_dict = {}
	self.D_params_train = []
        for param in params:
            self.D_params_dict.update({param.name:param})
	    if "embedding" in param.name:
		embedding_matrix = np.load("embedding-42000.npy")
		self.embedding_assign_op = param.assign(tf.Variable(embedding_matrix, trainable=False))
	    else:
		self.D_params_train.append(param)

    def build_Discriminator(self, images, text, label, name="discriminator", reuse=False):

        ### sentence: B, S
        hidden_size = self.hidden_size
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with tf.device('/cpu:0'), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self.dict_size, hidden_size], "float32", random_uniform_init)
                embedded_chars = tf.nn.embedding_lookup(word_emb_W, text) # B,S,H
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)      # B,S,H,1
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [hidden_size, self.num_classes],
                                                "float32", random_uniform_init)
                output_b = tf.get_variable("output_b", [self.num_classes], "float32", random_uniform_init)
	    with tf.variable_scope("images"):
                images_W = tf.get_variable("images_W", [self.dataset.img_dims, hidden_size],
                                                "float32", random_uniform_init)
                images_b = tf.get_variable("images_b", [hidden_size], "float32", random_uniform_init)
	    with tf.variable_scope("text"):
                text_W = tf.get_variable("text_W", [self.num_filters_total, hidden_size],
                                                "float32", random_uniform_init)
                text_b = tf.get_variable("text_b", [hidden_size], "float32", random_uniform_init)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)
            for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, hidden_size, 1, num_filter]
                    W = tf.get_variable("W", filter_shape, "float32", random_uniform_init)
                    b = tf.get_variable("b", [num_filter], "float32", random_uniform_init)
                    conv = tf.nn.conv2d(
                        embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.max_words - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            h_pool = tf.concat(3, pooled_outputs)      # B,1,1,total filters
            h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])        # b, total filters
            # Add highway
            with tf.variable_scope("highway"):
                h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)
            with tf.variable_scope("text"):
		text_emb = tf.nn.xw_plus_b(h_highway, text_W, text_b, name="text_emb")
	    with tf.variable_scope("images"):
                images_emb = tf.nn.xw_plus_b(images, images_W, images_b, name="images_emb")
	    with tf.variable_scope("output"):
		fusing_vec = tf.mul(text_emb, images_emb)
                l2_loss += tf.nn.l2_loss(output_W)
                l2_loss += tf.nn.l2_loss(output_b)
                logits = tf.nn.xw_plus_b(fusing_vec, output_W, output_b, name="logits")
		ypred_for_auc = tf.nn.softmax(logits)
		predictions = tf.argmax(logits, 1, name="predictions")
                #predictions = tf.nn.sigmoid(logits, name="predictions")
	    # Calculate Mean cross-entropy loss
	    with tf.variable_scope("loss"):
		losses = tf.nn.softmax_cross_entropy_with_logits(logits, label)
        	#losses = tf.nn.sigmoid_cross_entropy_with_logits(tf.squeeze(logits), self.input_y)
        	loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

	    return loss, predictions

    def train(self):

	self.train_op = self.optim.minimize(self.loss, global_step=self.global_step, var_list=self.D_params_train)
	#self.train_op = self.optim.minimize(self.loss, global_step=self.global_step)
        self.writer = tf.train.SummaryWriter("./logs/D_CNN_pretrained_sample", self.sess.graph)
        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver(var_list=self.D_params_dict, max_to_keep=30)
	# assign the G matrix to D pretrain
	self.sess.run(self.embedding_assign_op)
        count = 0
	for idx in range(self.max_iter//3000):
            self.save(self.checkpoint_dir, count)
            self.evaluate('test', count)
	    self.evaluate('train', count)
            for k in tqdm(range(3000)):
		right_images, right_text, _ = self.dataset.sequential_sample(self.batch_size)
                fake_images, fake_text, _ = self.negative_dataset.sequential_sample(self.batch_size)
                wrong_text = self.dataset.get_wrong_text(self.batch_size)

		images = np.concatenate((right_images, right_images, fake_images), axis=0)
		text = np.concatenate((right_text, wrong_text, fake_text.astype('int32')), axis=0)
		label = np.zeros((text.shape[0], self.num_classes))
		# right -> first entry
		# wrong -> second entry
		# fake -> third entry
		label[:self.batch_size, 0] = 1
		label[self.batch_size:2*self.batch_size, 1] = 1
		label[2*self.batch_size:, 2] = 1
                _, loss, summary_str = self.sess.run([self.train_op, self.loss, self.loss_sum],{
                                self.text: text.astype('int32'),
				self.images: images, 
                                self.label: label
                                })
                self.writer.add_summary(summary_str, count)
                count += 1

    def evaluate(self, split, count):

        if split == 'test':
            num_test_pair = -1
        elif split == 'train':
            num_test_pair = 5000
        right_images, right_text, _ = self.dataset.get_paired_data(num_test_pair, phase=split)
        # the true paired data we get
        num_test_pair = len(right_images)
        fake_images, fake_text, _ = self.negative_dataset.get_paired_data(num_test_pair, phase=split)
        random_idx = range(num_test_pair)
        np.random.shuffle(random_idx)
        wrong_text = np.squeeze(right_text[random_idx, :])
        count = 0.
	loss_t = []
	right_acc_t = []
        wrong_acc_t = []
        fake_acc_t = []
        for i in range(num_test_pair//self.batch_size):
            right_images_batch = right_images[i*self.batch_size:(i+1)*self.batch_size,:]
            fake_images_batch = fake_images[i*self.batch_size:(i+1)*self.batch_size,:]
            right_text_batch = right_text[i*self.batch_size:(i+1)*self.batch_size,:]
            fake_text_batch = fake_text[i*self.batch_size:(i+1)*self.batch_size,:]
            wrong_text_batch = wrong_text[i*self.batch_size:(i+1)*self.batch_size,:]
	    text_batch = np.concatenate((right_text_batch, wrong_text_batch, fake_text_batch.astype('int32')), axis=0)
	    images_batch = np.concatenate((right_images_batch, right_images_batch, fake_images_batch), axis=0)
 	    label = np.zeros((text_batch.shape[0], self.num_classes))
            # right -> first entry
            # wrong -> second entry
            # fake -> third entry
            label[:self.batch_size, 0] = 1
            label[self.batch_size:2*self.batch_size, 1] = 1
            label[2*self.batch_size:, 2] = 1
	    feed_dict = {self.images:images_batch, self.text:text_batch, self.label:label}
	    loss, pred, loss_str = self.sess.run([self.loss, self.pred, self.loss_sum], feed_dict)
	    loss_t.append(loss)
	    right_acc_t.append(np.sum((np.argmax(label[:self.batch_size],1)==pred[:self.batch_size])+0))
	    wrong_acc_t.append(np.sum((np.argmax(label[self.batch_size:2*self.batch_size],1)==pred[self.batch_size:2*self.batch_size])+0))
	    fake_acc_t.append(np.sum((np.argmax(label[2*self.batch_size:],1)==pred[2*self.batch_size:])+0))
            count += self.batch_size
	print "Phase =", split.capitalize()
        print "======================= Loss ====================="
	print '[$] Loss =', np.mean(loss_t)
        print "======================= Acc ======================"
        print '[$] Right Pair Acc. =', sum(right_acc_t)/count
        print '[$] Wrong Pair Acc. =', sum(wrong_acc_t)/count
        print '[$] Fake Pair Acc. =', sum(fake_acc_t)/count

    def save(self, checkpoint_dir, step):
        model_name = "D_Pretrained"
        model_dir = "%s" % (self.dataset.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir, "D_CNN_pretrained_sample")
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

