from ops import *
from utils import *
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np
from tqdm import tqdm


class FUNIT(object):
    def __init__(self, sess, args):

        self.phase = args.phase
        self.model_name = 'FUNIT'

        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.gpu_num = args.gpu_num

        self.iteration = args.iteration // args.gpu_num

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.ch = args.ch
        self.ema_decay = args.ema_decay

        self.K = args.K

        self.gan_type = args.gan_type


        """ Weight """
        self.adv_weight = args.adv_weight
        self.recon_weight = args.recon_weight
        self.feature_weight = args.feature_weight


        """ Generator """
        self.latent_dim = args.latent_dim

        """ Discriminator """
        self.sn = args.sn

        self.img_height = args.img_height
        self.img_width = args.img_width

        self.img_ch = args.img_ch

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.dataset_path = os.path.join('./dataset', self.dataset_name, 'train')
        self.class_dim = len(glob(self.dataset_path + '/*'))

        print()

        print("##### Information #####")
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# max iteration : ", self.iteration)
        print("# gpu num : ", self.gpu_num)

        print()

        print("##### Generator #####")
        print("# latent_dim : ", self.latent_dim)

        print()

        print("##### Discriminator #####")
        print("# spectral normalization : ", self.sn)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# feature_weight : ", self.feature_weight)
        print("# recon_weight : ", self.recon_weight)

        print()

    ##################################################################################
    # Generator
    ##################################################################################

    def content_encoder(self, x_init, reuse=tf.AUTO_REUSE, scope='content_encoder'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x_init, channel, kernel=7, stride=1, pad=3, pad_type='reflect',scope='conv')
            x = instance_norm(x, scope='ins_norm')
            x = relu(x)

            for i in range(3) :
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_' + str(i))
                x = instance_norm(x, scope='ins_norm_' + str(i))
                x = relu(x)

                channel = channel * 2

            for i in range(2) :
                x = resblock(x, channel, scope='resblock_' + str(i))

            return x

    def class_encoder(self, x_init, reuse=tf.AUTO_REUSE, scope='class_encoder'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x_init, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv')
            x = relu(x)

            for i in range(2) :
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_' + str(i))
                x = relu(x)

                channel = channel * 2

            for i in range(2) :
                x = conv(x, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='fix_conv_' + str(i))
                x = relu(x)

            x = global_avg_pooling(x)
            x = conv(x, channels=self.latent_dim, kernel=1, stride=1, scope='style_logit')

            return x

    def generator(self, content, style, reuse=tf.AUTO_REUSE, scope="generator"):
        channel = self.ch * 8 # 512
        with tf.variable_scope(scope, reuse=reuse):
            x = content

            mu, var = self.MLP(style, channel // 2, scope='MLP')

            for i in range(2) :
                idx = 2 * i
                x = adaptive_resblock(x, channel, mu[idx], var[idx], mu[idx + 1], var[idx + 1], scope='ada_resbloack_' + str(i))

            for i in range(3) :

                x = up_sample(x, scale_factor=2)
                x = conv(x, channel//2, kernel=5, stride=1, pad=2, pad_type='reflect', scope='up_conv_' + str(i))
                x = instance_norm(x, scope='ins_norm_' + str(i))
                x = relu(x)

                channel = channel // 2

            x = conv(x, channels=self.img_ch, kernel=7, stride=1, pad=3, pad_type='reflect', scope='g_logit')
            x = tanh(x)

            return x

    def MLP(self, style, channel, scope='MLP'):
        with tf.variable_scope(scope):
            x = style

            for i in range(2) :
                x = fully_connected(x, channel, scope='FC_' + str(i))
                x = relu(x)

            mu_list = []
            var_list = []

            for i in range(4) :
                mu = fully_connected(x, channel * 2, scope='FC_mu_' + str(i))
                var = fully_connected(x, channel * 2, scope='FC_var_' + str(i))

                mu = tf.reshape(mu, shape=[-1, 1, 1, channel * 2])
                var = tf.reshape(var, shape=[-1, 1, 1, channel * 2])

                mu_list.append(mu)
                var_list.append(var)


            return mu_list, var_list


    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, class_onehot, reuse=tf.AUTO_REUSE, scope="discriminator"):
        channel = self.ch
        class_onehot = tf.reshape(class_onehot, shape=[self.batch_size, 1, 1, -1])

        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x_init, channel, kernel=7, stride=1, pad=3, pad_type='reflect', sn=self.sn, scope='conv')

            for i in range(4) :
                x = pre_resblock(x, channel * 2, sn=self.sn, scope='front_resblock_0_' + str(i))
                x = pre_resblock(x, channel * 2, sn=self.sn, scope='front_resblock_1_' + str(i))
                x = down_sample_avg(x, scale_factor=2)

                channel = channel * 2

            for i in range(2) :
                x = pre_resblock(x, channel, sn=self.sn, scope='back_resblock_' + str(i))

            x_feature = x
            x = lrelu(x, 0.2)

            x = conv(x, channels=self.class_dim, kernel=1, stride=1, sn=self.sn, scope='d_logit')
            x = tf.reduce_sum(x * class_onehot, axis=-1, keepdims=True) # [1, 0, 0, 0, 0]

            return x, x_feature

    ##################################################################################
    # Model
    ##################################################################################


    def build_model(self):

        if self.phase == 'train' :
            """ Input Image"""
            img_data_class = Image_data(self.img_height, self.img_width, self.img_ch, self.dataset_path, self.augment_flag)
            img_data_class.preprocess()

            self.dataset_num = len(img_data_class.image_list)


            img_and_class = tf.data.Dataset.from_tensor_slices((img_data_class.image_list, img_data_class.class_list))

            gpu_device = '/gpu:0'
            img_and_class = img_and_class.apply(shuffle_and_repeat(self.dataset_num)).apply(
                map_and_batch(img_data_class.image_processing, batch_size=self.batch_size * self.gpu_num, num_parallel_batches=16,
                              drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))


            img_and_class_iterator = img_and_class.make_one_shot_iterator()

            self.content_img, self.content_class = img_and_class_iterator.get_next()
            self.style_img, self.style_class = img_and_class_iterator.get_next()

            self.content_img = tf.split(self.content_img, num_or_size_splits=self.gpu_num)
            self.content_class = tf.split(self.content_class, num_or_size_splits=self.gpu_num)
            self.style_img = tf.split(self.style_img, num_or_size_splits=self.gpu_num)
            self.style_class = tf.split(self.style_class, num_or_size_splits=self.gpu_num)

            self.fake_img = []

            d_adv_losses = []
            g_adv_losses = []
            g_recon_losses = []
            g_feature_losses = []


            for gpu_id in range(self.gpu_num):
                with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                        """ Define Generator, Discriminator """
                        content_code = self.content_encoder(self.content_img[gpu_id])
                        style_class_code = self.class_encoder(self.style_img[gpu_id])
                        content_class_code = self.class_encoder(self.content_img[gpu_id])

                        fake_img = self.generator(content_code, style_class_code)
                        recon_img = self.generator(content_code, content_class_code)

                        real_logit, style_feature_map = self.discriminator(self.style_img[gpu_id], self.style_class[gpu_id])
                        fake_logit, fake_feature_map = self.discriminator(fake_img, self.style_class[gpu_id])

                        recon_logit, recon_feature_map = self.discriminator(recon_img, self.content_class[gpu_id])
                        _, content_feature_map = self.discriminator(self.content_img[gpu_id], self.content_class[gpu_id])

                        """ Define Loss """
                        d_adv_loss = self.adv_weight * discriminator_loss(self.gan_type, real_logit, fake_logit, self.style_img[gpu_id])
                        g_adv_loss = 0.5 * self.adv_weight * (generator_loss(self.gan_type, fake_logit) + generator_loss(self.gan_type, recon_logit))

                        g_recon_loss = self.recon_weight * L1_loss(self.content_img[gpu_id], recon_img)

                        content_feature_map = tf.reduce_mean(tf.reduce_mean(content_feature_map, axis=2), axis=1)
                        recon_feature_map = tf.reduce_mean(tf.reduce_mean(recon_feature_map, axis=2), axis=1)
                        fake_feature_map = tf.reduce_mean(tf.reduce_mean(fake_feature_map, axis=2), axis=1)
                        style_feature_map = tf.reduce_mean(tf.reduce_mean(style_feature_map, axis=2), axis=1)

                        g_feature_loss = self.feature_weight * (L1_loss(recon_feature_map, content_feature_map) + L1_loss(fake_feature_map, style_feature_map))

                        d_adv_losses.append(d_adv_loss)
                        g_adv_losses.append(g_adv_loss)
                        g_recon_losses.append(g_recon_loss)
                        g_feature_losses.append(g_feature_loss)

                        self.fake_img.append(fake_img)

            self.g_loss = tf.reduce_mean(g_adv_losses) + \
                          tf.reduce_mean(g_recon_losses) + \
                          tf.reduce_mean(g_feature_losses) + regularization_loss('encoder') + regularization_loss('generator')

            self.d_loss = tf.reduce_mean(d_adv_losses) + regularization_loss('discriminator')


            """ Training """
            t_vars = tf.trainable_variables()
            G_vars = [var for var in t_vars if 'encoder' in var.name or 'generator' in var.name]
            D_vars = [var for var in t_vars if 'discriminator' in var.name]

            if self.gpu_num == 1 :
                prev_G_optim = tf.train.RMSPropOptimizer(self.lr, decay=0.99, epsilon=1e-8).minimize(self.g_loss, var_list=G_vars)
                self.D_optim = tf.train.RMSPropOptimizer(self.lr, decay=0.99, epsilon=1e-8).minimize(self.d_loss, var_list=D_vars)
                # Pytorch : decay=0.99, epsilon=1e-8

            else :
                prev_G_optim = tf.train.RMSPropOptimizer(self.lr, decay=0.99, epsilon=1e-8).minimize(self.g_loss, var_list=G_vars, colocate_gradients_with_ops=True)
                self.D_optim = tf.train.RMSPropOptimizer(self.lr, decay=0.99, epsilon=1e-8).minimize(self.d_loss, var_list=D_vars, colocate_gradients_with_ops=True)
                # Pytorch : decay=0.99, epsilon=1e-8

            self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
            with tf.control_dependencies([prev_G_optim]):
                self.G_optim = self.ema.apply(G_vars)


            """" Summary """
            self.summary_g_loss = tf.summary.scalar("g_loss", self.g_loss)
            self.summary_d_loss = tf.summary.scalar("d_loss", self.d_loss)

            self.summary_g_adv_loss = tf.summary.scalar("g_adv_loss", tf.reduce_mean(g_adv_losses))
            self.summary_g_recon_loss = tf.summary.scalar("g_recon_loss", tf.reduce_mean(g_recon_losses))
            self.summary_g_feature_loss = tf.summary.scalar("g_feature_loss", tf.reduce_mean(g_feature_losses))


            g_summary_list = [self.summary_g_loss,
                              self.summary_g_adv_loss,
                              self.summary_g_recon_loss, self.summary_g_feature_loss
                              ]

            d_summary_list = [self.summary_d_loss]

            self.summary_merge_g_loss = tf.summary.merge(g_summary_list)
            self.summary_merge_d_loss = tf.summary.merge(d_summary_list)

        else :
            """ Test """
            self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
            self.test_content_img = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch])
            self.test_class_img = tf.placeholder(tf.float32, [self.K, self.img_height, self.img_width, self.img_ch])

            test_content_code = self.content_encoder(self.test_content_img)
            test_style_class_code = tf.reduce_mean(self.class_encoder(self.test_class_img), axis=0, keepdims=True)

            self.test_fake_img = self.generator(test_content_code, test_style_class_code)

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=20)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_batch_id = checkpoint_counter
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")

        else:
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for idx in range(start_batch_id, self.iteration):

            # Update D
            _, d_loss, summary_str = self.sess.run([self.D_optim, self.d_loss, self.summary_merge_d_loss])
            self.writer.add_summary(summary_str, counter)

            # Update G
            content_images, style_images, fake_x_images, _, g_loss, summary_str = self.sess.run(
                [self.content_img[0], self.style_img[0], self.fake_img[0],
                 self.G_optim,
                 self.g_loss, self.summary_merge_g_loss])

            self.writer.add_summary(summary_str, counter)


            # display training status
            counter += 1
            print("iter: [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (idx, self.iteration, time.time() - start_time, d_loss, g_loss))

            if np.mod(idx + 1, self.print_freq) == 0:
                content_images = np.expand_dims(content_images[0], axis=0)
                style_images = np.expand_dims(style_images[0], axis=0)
                fake_x_images = np.expand_dims(fake_x_images[0], axis=0)

                merge_images = np.concatenate([content_images, style_images, fake_x_images], axis=0)

                save_images(merge_images, [1, 3],
                            './{}/merge_{:07d}.jpg'.format(self.sample_dir, idx + 1))

                # save_images(content_images, [1, 1],
                #             './{}/content_{:07d}.jpg'.format(self.sample_dir, idx + 1))
                #
                # save_images(style_images, [1, 1],
                #             './{}/style_{:07d}.jpg'.format(self.sample_dir, idx + 1))
                #
                # save_images(fake_x_images, [1, 1],
                #             './{}/fake_{:07d}.jpg'.format(self.sample_dir, idx + 1))


            if np.mod(counter - 1, self.save_freq) == 0:
                self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        return "{}_{}_{}_{}adv_{}feature_{}recon{}".format(self.model_name, self.dataset_name, self.gan_type,
                                                           self.adv_weight, self.feature_weight, self.recon_weight,
                                                           sn)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        content_images = glob('./dataset/{}/{}/{}/*.*'.format(self.dataset_name, 'test', 'content'))
        class_images = glob('./dataset/{}/{}/{}/*.*'.format(self.dataset_name, 'test', 'class'))

        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'encoder' in var.name or 'generator' in var.name]

        shadow_G_vars_dict = {}

        for g_var in G_vars :
            shadow_G_vars_dict[self.ema.average_name(g_var)] = g_var

        self.saver = tf.train.Saver(shadow_G_vars_dict)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>content</th><th>style</th><th>output</th></tr>")

        for sample_content_image in tqdm(content_images):
            sample_image = load_test_image(sample_content_image, self.img_width, self.img_height)

            random_class_images = np.random.choice(class_images, size=self.K, replace=False)
            sample_class_image = np.concatenate([load_test_image(x, self.img_width, self.img_height) for x in random_class_images])

            fake_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_content_image)))
            class_path = os.path.join(self.result_dir, 'style_{}'.format(os.path.basename(sample_content_image)))

            fake_img = self.sess.run(self.test_fake_img, feed_dict={self.test_content_img : sample_image, self.test_class_img : sample_class_image})

            save_images(fake_img, [1, 1], fake_path)
            save_images(sample_class_image, [1, self.K], class_path)

            index.write("<td>%s</td>" % os.path.basename(sample_content_image))
            index.write(
                "<td><img src='%s' width='%d' height='%d'></td>" % (sample_content_image if os.path.isabs(sample_content_image) else (
                        '../..' + os.path.sep + sample_content_image), self.img_width, self.img_height))

            index.write(
                "<td><img src='%s' width='%d' height='%d'></td>" % (class_path if os.path.isabs(class_path) else (
                        '../..' + os.path.sep + class_path), self.img_width * self.K, self.img_height))
            index.write(
                "<td><img src='%s' width='%d' height='%d'></td>" % (fake_path if os.path.isabs(fake_path) else (
                        '../..' + os.path.sep + fake_path), self.img_width, self.img_height))
            index.write("</tr>")

        index.close()