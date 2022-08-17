from utils import *

def sarcnn(input, output_channels=1):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 19):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=0))
    with tf.variable_scope('block%d' % (layers+1)):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return input - output

class denoiser(object):
    def __init__(self, sess, real_sar, input_c_dim=1):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        self.real_sar = real_sar
        if not self.real_sar:
            "Testing on images with simulated speckle"
            self.X = tf_simulate_speckle(self.Y_)
        else:
            "Testing on real data"
            self.X = self.Y_
        self.Y = sarcnn(self.X)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, test_files, ckpt_dir, save_dir, dataset_dir):
        """Test SAR-CNN"""
        tf.initialize_all_variables().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        print("[*] start testing...")
        for idx in range(len(test_files)):
            clean_image = load_sar_images(test_files[idx]).astype(np.float32) / 255.0
            if self.real_sar:
                "downsampling real image to reduce correlation"
                clean_image_downsampled = clean_image[:,::2,::2,:]
                output_clean_image, noisy_image = self.sess.run([self.Y, self.X],
                                                            feed_dict={self.Y_: clean_image_downsampled})
                noisy_image = clean_image
            else:
                output_clean_image, noisy_image = self.sess.run([self.Y, self.X],
                                                                feed_dict={self.Y_: clean_image})
            groundtruth = denormalize_sar(clean_image)
            noisyimage = denormalize_sar(noisy_image)
            outputimage = denormalize_sar(output_clean_image-cn)

            imagename = test_files[idx].replace(dataset_dir+"/", "")
            print("Denoised image %s" % imagename)

            save_sar_images(groundtruth, outputimage, noisyimage, imagename, save_dir, self.real_sar)
