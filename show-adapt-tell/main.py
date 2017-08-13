import os
import scipy.misc
import numpy as np
import tensorflow as tf
from pretrain_G import G_pretrained
from pretrain_CNN_D import D_pretrained
from model import SeqGAN
from data_loader import mscoco, mscoco_negative
import pprint
import pdb

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 5e-5, "Learning rate of for adam [0.0003]")
flags.DEFINE_float("drop_out_rate", 0.3, "Drop out rate fro LSTM")
flags.DEFINE_float("discount", 0.95, "discount factor in RL")
flags.DEFINE_string("model_name", "cub_no_scheduled", "")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")	# 128:G, 32:D
flags.DEFINE_integer("G_hidden_size", 512, "")		# 512:G, 64:D
flags.DEFINE_integer("D_hidden_size", 512, "")
flags.DEFINE_integer("max_iter", 100000, "")
flags.DEFINE_integer('max_to_keep', 40, '')
flags.DEFINE_string("method", "ROUGE_L", "")
flags.DEFINE_string("load_ckpt", './checkpoint/mscoco/G_pretrained/G_Pretrained-39000', "Directory name to loade the checkpoints [checkpoint]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("G_is_pretrain", False, "Do the G pretraining")
flags.DEFINE_boolean("D_is_pretrain", False, "Do the D pretraining")
flags.DEFINE_boolean("load_pretrain", True, "Load the pretraining")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")

# Setting from Self-critical Sequence Training for Image Captioning
tf.app.flags.DEFINE_float('init_lr', 5e-5, '')	# follow IBM's paper
tf.app.flags.DEFINE_float('lr_decay', 0.8, 'learning rate decay factor')
tf.app.flags.DEFINE_float('lr_decay_every', 6600, 'every 3 epoch 3*2200')
tf.app.flags.DEFINE_float('ss_ascent', 0.05, 'schedule sampling')
tf.app.flags.DEFINE_float('ss_ascent_every', 11000, 'every 5 epoch 5*2200')
tf.app.flags.DEFINE_float('ss_max', 0.25, '0.05*5=0.25')

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()
def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    dataset = mscoco(FLAGS)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1/10
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
	filter_sizes = [1,2,3,4,5,6,7,8,9,10,16,24,dataset.max_words]
        num_filters = [100,200,200,200,200,100,100,100,100,100,160,160,160]
        num_filters_total = sum(num_filters)
        info={'num_classes':3, 'filter_sizes':filter_sizes, 'num_filters':num_filters,
                        'num_filters_total':num_filters_total, 'l2_reg_lambda':0.2}
	if FLAGS.G_is_pretrain:
	    G_pretrained_model = G_pretrained(sess, dataset, conf=FLAGS)
	    if FLAGS.is_train:
		G_pretrained_model.train()
	    G_pretrained_model.evaluate('test', 0, )
	if FLAGS.D_is_pretrain:
	    negative_dataset = mscoco_negative(dataset, FLAGS)
	    D_pretrained_model = D_pretrained(sess, dataset, negative_dataset, info, conf=FLAGS)
            D_pretrained_model.train()
	if FLAGS.is_train:
	    model = SeqGAN(sess, dataset, info, conf=FLAGS)
	    model.train()

if __name__ == '__main__':
    tf.app.run()
