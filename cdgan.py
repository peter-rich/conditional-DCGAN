#       
#	Conditional DCGAN
#
#	Author: Zhanfu Yang
#	email: yang1676@purdue.edu
#
#	Class 0:  No bed, No lamp
#	Class 1:  Bed, No lamp
#	Class 2:  No bed, lamp
#	Class 3:  bed, lamp
#		

import os, time, itertools, imageio, pickle, random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
#from skimage.io import imread
import scipy.misc

__author__ = 'Zhanfu Yang'
__email__ = 'yang1676@purdue.edu'
__license__ = 'MIT'

# Define the leaky_relu
def l_relu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

# Define the Generator
def generator(x, y_label, isTrain=True, reuse=False):
	with tf.variable_scope('generator', reuse=reuse):
		# initializer
        	w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        	b_init = tf.constant_initializer(0.0)

        	# concat layer
        	cat1 = tf.concat([x, y_label], 3)
		# 0 hidden layer
		deconv0 = tf.layers.conv2d_transpose(cat1, 1024, [16, 16], strides=(2, 2), padding='valid', kernel_initializer=w_init, bias_initializer=b_init)
                lrelu1 = l_relu(tf.layers.batch_normalization(deconv0, training=isTrain), 0.2)
        	# 1st hidden layer
        	deconv1 = tf.layers.conv2d_transpose(deconv0, 512, [8, 8], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        	lrelu1 = l_relu(tf.layers.batch_normalization(deconv1, training=isTrain), 0.2)

        	# 2nd hidden layer
        	deconv2 = tf.layers.conv2d_transpose(lrelu1, 256, [8, 8], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        	lrelu2 = l_relu(tf.layers.batch_normalization(deconv2, training=isTrain), 0.2)
		# 3rd hidden layer
		
                deconv3 = tf.layers.conv2d_transpose(lrelu2, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
                lrelu3 = l_relu(tf.layers.batch_normalization(deconv3, training=isTrain), 0.2)
        	# output layer
        	deconv4 = tf.layers.conv2d_transpose(lrelu3, 3, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        	output = tf.nn.tanh(deconv4)
		
		return output

# Define the Discriminator
def discriminator(x, y_fill, isTrain=True, reuse=False):
	with tf.variable_scope('discriminator', reuse=reuse):
		# Initializer
		w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
		b_init = tf.constant_initializer(0.0)
		
		# concat layer
		print("D->0->layer->D")
		cat1 = tf.concat([x, y_fill], 3)
		print("D->1->layer->D")
		# 1st hidden layer
		cov1 = tf.layers.conv2d(cat1, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
		relu1 = l_relu(cov1, 0.2)
		print("D->2->layer->D")
		# 2nd hidden layer
		cov2 = tf.layers.conv2d(relu1, 256, [7, 7], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
		relu2 = l_relu(tf.layers.batch_normalization(cov2, training=isTrain), 0.2)
		print("D->3->layer->D")
		# output layer
		cov3 = tf.layers.conv2d(relu2, 1, [7, 7], strides=(1, 1), padding='valid', kernel_initializer=w_init)
		output = tf.nn.sigmoid(cov3)
		return output, cov3

############################################
#   Main function()
############################################
# preprocess
img_size = 256
onehot = np.eye(4)
temp_z_ = np.random.normal(0, 1, (4, 1, 1, 16))
fixed_z_ = temp_z_
fixed_y_ = np.zeros((4, 1))
for i in range(3):
	fixed_z_ = np.concatenate([fixed_z_, temp_z_], 0)
	temp = np.ones((4, 1)) + i
	fixed_y_ = np.concatenate([fixed_y_, temp], 0)

fixed_y_ = onehot[fixed_y_.astype(np.int32)].reshape((16, 1, 1, 4))
def show_result(num_epoch, show = False, save = False, path = 'result.png'):
	test_images = sess.run(G_z, {z: fixed_z_, y_label: fixed_y_, isTrain: False})
	size_figure_grid = 4
	fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))	
	for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
		ax[i, j].get_xaxis().set_visible(False)
		ax[i, j].get_yaxis().set_visible(False)
	for k in range(16):
		i = k // 4
		j = k % 4
		ax[i, j].cla()
		ax[i, j].imshow(np.reshape(test_images[k], (img_size, img_size)), cmap='gray')

	label = 'Epoch {0}'.format(num_epoch)
    	fig.text(0.5, 0.04, label, ha='center')

    	if save:
        	plt.savefig(path)

    	if show:
        	plt.show()
    	else:
        	plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    	x = range(len(hist['D_losses']))

    	y1 = hist['D_losses']
    	y2 = hist['G_losses']

    	plt.plot(x, y1, label='D_loss')
    	plt.plot(x, y2, label='G_loss')

    	plt.xlabel('Epoch')
    	plt.ylabel('Loss')

    	plt.legend(loc=4)
    	plt.grid(True)
    	plt.tight_layout()

    	if save:
        	plt.savefig(path)

    	if show:
        	plt.show()
    	else:
        	plt.close()

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)


# training parameters
batch_size = 32
input_height = 256
output_hight = 256

# lr = 0.0002
train_epoch = 400
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.0002, global_step, 500, 0.95, staircase=True)
# load MNIST
#### DataSet
data_dir = "../../scratch/"
dataset_name_0 = "EmptyRoom"
dataset_name_1 = "OnlyOneBed"
dataset_name_2 = "RoomWithLamp"
dataset_name_3 = "bedroom"
## Data path:
data_path_0 = os.path.join(data_dir, dataset_name_0, "*.jpg")
data_path_1 = os.path.join(data_dir, dataset_name_1, "*.jpg")
data_path_2 = os.path.join(data_dir, dataset_name_2, "*.jpg")
data_path_3 = os.path.join(data_dir, dataset_name_3, "*.jpg")

data_0 = glob(data_path_0)
data_1 = glob(data_path_1)
data_2 = glob(data_path_2)
data_3 = glob(data_path_3)
# Length
len_0 = len(data_0)
len_1 = len(data_1)
len_2 = len(data_2)
len_3 = len(data_3)

# Give some labels to it.
label_0 = np.zeros((1,4))
label_1 = np.zeros((1,4))
label_2 = np.zeros((1,4))
label_3 = np.zeros((1,4))
label_0[0][0]=1.0
label_1[0][1]=1.0
label_2[0][2]=1.0
label_3[0][3]=1.0
train_label = label_0

for i in range(0,len_0-1):
	train_label=np.concatenate((train_label,label_0), axis=0)
for i in range(0,len_1):
        train_label=np.concatenate((train_label,label_1), axis=0)
for i in range(0,len_2):
        train_label=np.concatenate((train_label,label_2), axis=0)
for i in range(0,len_3):
        train_label=np.concatenate((train_label,label_3), axis=0)

print(label_0)

files_set = data_0 + data_1 + data_2 + data_3
print(train_label.shape)
train_set = sample = [
          get_image(	sample_file,
                    	input_height=img_size,
                    	input_width=img_size,
                    	resize_height=img_size,
                    	resize_width=img_size,
			crop = False,
			grayscale=False) for sample_file in files_set]
# shuffle the data together with the same order.
combined = list(zip(train_set, train_label))
random.shuffle(combined)

train_set[:], train_label[:] = zip(*combined)

# dimension

c_dim = imread(data_0[0]).shape[-1]
##############
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])
###
# variables : input
x = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 3))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 16))
y_label = tf.placeholder(tf.float32, shape=(None, 1, 1, 4))
y_fill = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 4))
isTrain = tf.placeholder(dtype=tf.bool)

print("Step 1")
# networks : generator
G_z = generator(z, y_label, isTrain)
print(G_z.get_shape())
# networks : discriminator
D_real, D_real_logits = discriminator(x, y_fill, isTrain)
D_fake, D_fake_logits = discriminator(G_z, y_fill, isTrain, reuse=True)
print(str("real and fake:"))
print(D_real_logits.get_shape())
print(D_fake_logits.get_shape())
print("Step 2")
# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 58, 58, 1])))
print("Step 3")
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 58, 58, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 58, 58, 1])))

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
	optim = tf.train.AdamOptimizer(lr, beta1=0.5)
	D_optim = optim.minimize(D_loss, global_step=global_step, var_list=D_vars)
	# D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
	G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# MNIST resize and normalization
# train_set = tf.image.resize_images(mnist.train.images, [img_size, img_size]).eval()
# train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1

# results save folder
root = 'Bedroom_cDCGAN_results/'
model = 'Bedroom_cDCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
	G_losses = []
	D_losses = []
	epoch_start_time = time.time()
	ned = list(zip(train_set, train_label))
	random.shuffle(combined)
	train_set[:], train_label[:] = zip(*combined)
	shuffled_set=np.array(train_set)
	shuffled_label=np.array(train_label)
	print(shuffled_label.ndim)
	for iter in range(len(shuffled_set) // batch_size):
        	# update discriminator
        	x_ = shuffled_set[iter*batch_size:(iter+1)*batch_size]
        	y_label_ = shuffled_label[iter*batch_size:(iter+1)*batch_size].reshape([batch_size, 1, 1, 4])
        	y_fill_ = y_label_ * np.ones([batch_size, img_size, img_size, 4])
        	print(str(iter)+"-->Initial")
		z_ = np.random.normal(0, 1, (batch_size, 1, 1, 16))

        	loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, y_fill: y_fill_, y_label: y_label_, isTrain: True})
		print(str(iter)+"-->start trainning")
        # update generator
        	z_ = np.random.normal(0, 1, (batch_size, 1, 1, 16))
        	y_ = np.random.randint(0, 4, (batch_size, 1))
        	y_label_ = onehot[y_.astype(np.int32)].reshape([batch_size, 1, 1, 4])
        	y_fill_ = y_label_ * np.ones([batch_size, img_size, img_size, 4])
        	loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, y_fill: y_fill_, y_label: y_label_, isTrain: True})

        	errD_fake = D_loss_fake.eval({z: z_, y_label: y_label_, y_fill: y_fill_, isTrain: False})
        	errD_real = D_loss_real.eval({x: x_, y_label: y_label_, y_fill: y_fill_, isTrain: False})
        	errG = G_loss.eval({z: z_, y_label: y_label_, y_fill: y_fill_, isTrain: False})

        	D_losses.append(errD_fake + errD_real)
        	G_losses.append(errG)

    	epoch_end_time = time.time()
    	per_epoch_ptime = epoch_end_time - epoch_start_time
    	print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    	fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    	show_result((epoch + 1), save=True, path=fixed_p)
    	train_hist['D_losses'].append(np.mean(D_losses))
    	train_hist['G_losses'].append(np.mean(G_losses))
    	train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    	pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    	img_name = root + 'Fixed_results/' + model + str(e + 1) + '.jpg'
    	images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

sess.close()
