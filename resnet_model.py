import tensorflow as tf
import numpy as np
import utils
class Model(object):


    def __init__(self):
        pass

    def build(self, size, num_classes, logdir, learning_rate):

        self.x, self.y = self.init_placeholders(size, num_classes)
        self.logits = self.resnet18_model(num_classes)
        self.summary_writer = self.init_summary_writer(logdir)
        self.saver = self.init_saver()
        self.loss = self.init_loss()
        self.optimizer = self.init_optimizer(learning_rate)

    def run(self, train_data, train_label, batch_size, num_classes, size, mean):

        merged_summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(20):
                tf.summary.scalar('epoch', epoch)
                if epoch % 10 == 0:
                    perm = np.random.permutation(len(train_data))
                    valid_id = 1
                train_set, validation_set = utils.separate_validation(np.copy(perm), 10, valid_id)
                cond = True
                idx = 0
                while cond:
                    if idx + batch_size >= len(train_set):
                        index = tarin_set[idx:]
                        cond = False
                    else:
                        index = train_set[idx:idx+batch_size]
                    train_imgs = utils.get_batch(train_data[index], mean)
                    print(train_set.shape, validation_set.shape)
                    batch_label = np.eye(num_classes)[train_label[index].astype(np.int32)].astype(np.int32)
                    feed_dict = {self.x: train_imgs, self.y: batch_label}
                    _, l, summary = sess.run([self.optimizer, self.loss, merged_summary_op], feed_dict=feed_dict)
                    idx += batch_size
                    self.summary_writer.add_summary(summary)
                    print('epoch: {}, iteration : {}, loss: {}'.format(epoch, idx//batch_size, l))


                valid_imgs = utils.get_batch(train_data[validation_set])
                valid_labels = np.eye(num_classes)[test_label[validation_set].astype(np.int32)]
                feed_dict = {self.x: valid_imgs, self.y: valid_labels}
                logits = sess.run(logits, feed_dict=feed_dict)
                pred = np.argmax(logits, axis=1)
                result = valid_labels == pred
                print('Accuracy : {}'.format(result))

                valid_id += 1
                save_path = saver.save(sess, "/tmp/model.ckpt")

    def resnet18_model(self, num_classes):

        conv1 = tf.layers.conv2d(self.x, 64, (7, 7), padding='same', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='conv1')


        bn1 = tf.layers.batch_normalization(conv1, name='bn1')

        mp1 = tf.layers.max_pooling2d(bn1, (3, 3), (2, 2), padding='same', name='maxp1')
        res_block1 = self._res_block(mp1, 64, (3, 3), 1)
        res_block2 = self._res_block(res_block1, 64, (3, 3), 2)

        res_block3 = self._res_block_proj(res_block2, 128, (3, 3), (2, 2), 3)
        res_block4 = self._res_block(res_block3, 128, (3, 3), 4)

        res_block5 = self._res_block_proj(res_block4, 256, (3, 3), (2, 2), 5)
        res_block6 = self._res_block(res_block5, 256, (3, 3), 6)

        res_block7 = self._res_block_proj(res_block6, 512, (3, 3), (2, 2), 7)
        res_block8 = self._res_block(res_block7, 512, (3, 3), 8)



        avg = tf.layers.average_pooling2d(res_block8, (7, 7), (1, 1), name='avg1')
        logits = tf.layers.dense(avg, num_classes, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='fc')
        return logits

    def init_summary_writer(self, logdir):
        summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        return summary_writer


    def init_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits, name='loss'))
        tf.summary.scalar('loss', loss)
        tf.summary.histogram('loss_hist', loss)
        return loss

    def init_saver(self):
        saver = tf.train.Saver()
        return saver

    def init_optimizer(self, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        return optimizer

    def init_placeholders(self, size, classes):
        x = tf.placeholder(np.float32, shape=[None, size, size, 3], name='x')
        y = tf.placeholder(np.int32, shape=[None, classes], name='y')
        return x, y

    def _res_block(self, x, num_filter, kernel_size, block_nmb):
        cv1 = self._conv_bn_relu(x, num_filter, kernel_size, block_nmb, 1)
        cv2 = self._conv_bn(cv1, num_filter, kernel_size, block_nmb, 2)
        p = cv2 + x
        return tf.nn.relu(p)

    def _res_block_proj(self, x, num_filter, kernel_size, strides, block_nmb):
        cv1 = self._conv_bn_relu(x, num_filter, kernel_size, block_nmb, 1, strides=strides)
        cv2 = self._conv_bn(cv1, num_filter, kernel_size, block_nmb, 2)
        projection = self._conv_bn(x, num_filter, (1, 1), block_nmb, 3, strides)
        p = cv2 + projection
        return tf.nn.relu(p)

    def _conv_bn(self, x, num_filter, kernel_size, block_nmb, layer_nmb, strides=(1, 1)):
        cv = tf.layers.conv2d(x, num_filter, kernel_size, strides=strides, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='conv{}_{}'.format(block_nmb, layer_nmb))
        bn = tf.layers.batch_normalization(cv, name='bn{}_{}'.format(block_nmb, layer_nmb))
        return bn

    def _conv_bn_relu(self, x, num_filter, kernel_size, block_nmb, layer_nmb, strides=(1, 1)):
        cv_relu = self._conv_bn(x, num_filter, kernel_size, block_nmb, layer_nmb, strides=strides)
        return tf.nn.relu(cv_relu)
