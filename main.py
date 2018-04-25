import os.path
import tensorflow as tf
import numpy as np
import helper
import time
import datetime
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import imageio
import scipy
import argparse


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'),\
       'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    lyr3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    lyr4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    lyr7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep, lyr3, lyr4, lyr7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, l2_regularization_rate=0.001):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    #conv_transpose_init_stddev = 0.001
    conv_init_stddev = 0.01

    # apply 1x1 convolution to layers so that number of classes is reduced

    with tf.name_scope('decoder') as scope:
        l7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out,
                                   filters=num_classes,
                                   kernel_size=(1,1),
                                   padding='same',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_regularization_rate),
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=conv_init_stddev),
                                   name='l7_conv_1x1')

    # upscale layer 7 by X2 to match layer 4 then add together
        l7_upscaledx2 = tf.layers.conv2d_transpose(l7_conv_1x1,
                                   filters=num_classes,
                                   kernel_size=(4,4),
                                   strides=(2, 2),
                                   padding='same',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_regularization_rate),
                                   kernel_initializer=tf.zeros_initializer,
                                   name='l7_upscaledx2')

        pool4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')

        l4_conv_1x1 = tf.layers.conv2d(pool4_out_scaled,
                                   filters=num_classes,
                                   kernel_size=(1,1),
                                   padding='same',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_regularization_rate),
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=conv_init_stddev),
                                   name='l4_conv_1x1')

        skipsum_7_4 = tf.add(l7_upscaledx2, l4_conv_1x1, name='skipsum_7_4')

    # upscale sum of 7 and 4 by X2 to match layer 3 then add together
        output_2XSS_7_4 = tf.layers.conv2d_transpose(skipsum_7_4,
                                   filters=num_classes,
                                   kernel_size=(4,4),
                                   strides=(2, 2),
                                   padding='same',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_regularization_rate),
                                   kernel_initializer=tf.zeros_initializer,
                                   name='output_2XSS_7_4')

        pool3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')

        l3_conv_1x1 = tf.layers.conv2d(pool3_out_scaled,
                                   filters=num_classes,
                                   kernel_size=(1,1),
                                   padding='same',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_regularization_rate),
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=conv_init_stddev),
                                   name='l3_conv_1x1')

        skipsum_7_4_3 = tf.add(output_2XSS_7_4, l3_conv_1x1, name='skipsum_7_4_3')
    
    # upscale sum of 7 4 and 3 by X8 to match original
        ss_7_4_3_conv2d = tf.layers.conv2d_transpose(skipsum_7_4_3,
                                   filters=num_classes,
                                   kernel_size=(16, 16),
                                   strides=(8, 8),
                                   padding='same',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_regularization_rate),
                                   kernel_initializer=tf.zeros_initializer,
                                   name='ss_7_4_3_conv2d')

        last_layer = tf.identity(ss_7_4_3_conv2d, name='last_layer')
    return last_layer 
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes, iou_test=False):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, combined_loss, iou_obj)
    """
    with tf.name_scope('combined_loss'):

        logits = tf.reshape(nn_last_layer, (-1, num_classes))
        labels = tf.reshape(correct_label, (-1, num_classes))
  
        # Compute the regularization loss.
        # This is a list of the individual loss values, so we still need to sum them up.
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) 
        regularization_loss = tf.reduce_sum(regularization_losses, name='regularization_loss')

        # Compute the cross entropy loss.
        ce_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        ce_loss = tf.reduce_mean(ce_losses, name='approximation_loss')

        # sum the losses
        # combined_loss = regularization_loss + ce_loss
        combined_loss = tf.add(regularization_loss, ce_loss, name='combined_loss')
        cmb_loss = regularization_loss + ce_loss

    # Add loss to TensorBoard summary logging
    tf.summary.scalar('regularization loss', regularization_loss)
    tf.summary.scalar('cross entropy loss', ce_loss)
    tf.summary.scalar('combined loss', combined_loss)

    # set up global step
    global_step = tf.Variable(0, trainable=False, name='global_step')

    with tf.name_scope('train'): 

        # Compute the optimizer as Adam and minimize combine ce and reg losses.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(combined_loss, global_step=global_step, name='train_op')  
    
    tf.summary.scalar('learning rate', learning_rate)

    if iou_test is True:
        # Intersection over Union
        with tf.name_scope('I_o_U'):

            prediction = tf.argmax(nn_last_layer, axis=3)
            ground_truth = correct_label[:, :, :, 1]
            mean_iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
            iou_obj = (mean_iou, iou_op)

        tf.summary.scalar('I_o_U', mean_iou)

        return logits, train_op, combined_loss, iou_obj
    else:
        return logits, train_op, combined_loss


    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # train_op = optimizer.minimize(cross_entropy_loss)

    # return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, combined_loss, input_image,
             correct_label, keep_prob, learning_rate, l2_regularization_rate=0.001, iou_obj=None, lr=0.0001):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param combined_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param l2_regularization_rate: TF Placeholder for regularization rate
    :param iou_obj: [0]: TF mean intersection-over-union [1]: TF operation for confusion matrix.
    """

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    tb_out_dir = os.path.join('tb/', str(time.time()))
    tb_merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(tb_out_dir + '/train', graph=sess.graph)
    test_writer = tf.summary.FileWriter(tb_out_dir + '/test')

    saver = tf.train.Saver()

    beginTime = time.time()


    print("Training...")

    for epoch in range(epochs):
        print("EPOCH {} ...".format(epoch + 1))
        loss = -1.0
        total_iou = 0.0
        image_count = 0
        counter = 0
        for images, labels in get_batches_fn(batch_size):
            # run tests on alternate batches
            if counter % 2 == 0:

                feed_dict = {input_image: images,
                            correct_label: labels,
                            keep_prob: 0.5,
                            learning_rate: lr}

                # dropout parameter 50% is from the original paper
                _, loss, summary = sess.run([train_op, combined_loss, tb_merged], feed_dict=feed_dict)
                # Log loss for each global step
                global_step = tf.train.get_or_create_global_step()
                step = tf.train.global_step(sess, global_step)
                train_writer.add_summary(summary, step)
                print("  Step: {}, Combined_Loss ={:3.3f}".format(step, loss), end = '')

                image_count += len(images)
            else: # alternate images
                if iou_obj is not None:
                    iou = iou_obj[0]
                    iou_op = iou_obj[1]

                    feed_dict={input_image: images,
                            correct_label: labels,
                            keep_prob: 1.0}

                    sess.run(iou_op, feed_dict=feed_dict)
                    mean_iou = sess.run(iou)
                    total_iou += mean_iou * len(images)
                    avg_iou = total_iou / image_count
                    print(" per batch mean_iou = {:3.3f}, per epoch avg_iou = {:3.3f}".format(mean_iou, avg_iou) )
            counter += 1
            # just do 500 images training per epoch, not whole set
            if image_count % 500 == 0 and counter % 2 == 1:
                break
        avg_iou = total_iou / image_count
        print("Epoch {} / {}, Combined Loss {:0.5f}, Avg IoU {:0.5f}".format(epoch+1, epochs, loss, avg_iou))
        saver.save(sess, 'checkpoints/teeekay', global_step=step)
 

    endTime = time.time()
    print('Training time: {:5.2f}s'.format(endTime - beginTime))

def parse_args():
  """
  Set up argument parser for command line operation of main.py program
  """

  parser = argparse.ArgumentParser(description='Semantic Segmentation of Road Scenes')

  parser.add_argument('-md', '--mode',
    help='mode [1]: 0=Train, 1=Test, 2=Video', type=int, default=1)

  parser.add_argument('-ep', '--epochs',
    help='epochs [5]', type=int, default=5)

  parser.add_argument('-bs', '--batch_size',
    help='batch size [1]', type=int, default=1)

  parser.add_argument('-lr', '--learn_rate',
    help='learning rate [0.0001]', type=float, default=0.0001)

  parser.add_argument('-l2r', '--l2_regularization_rate',
    help='l2 regularization rate [0.00001]', type=float, default=0.00001)

  parser.add_argument('-crit', '--softmax_criteria',
    help='softmax criteria [0.9]', type=float, default=0.90)

  # args = parser.parse_args()
  args = parser.parse_known_args()
  return args


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    model_path = './teeekay/'  
    model = 'ss_mdl8'
    # mdl8 20 * 1,000 images at lr = 0.000025 
    # mdl7 10 * 1,000 images at lr = 0.00001 
    # mdl6 with cityscapes data 
    #mdl4 100 at 0.00001  
    #mdl3 50 at .00003
    #mdl2 25 at .00005
 
    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    Kepochs = FLAGS.epochs  # set to reasonable value
    Kbatch_size = FLAGS.batch_size
    KLearningRate = FLAGS.learn_rate
    Kl2_regularization_rate = FLAGS.l2_regularization_rate

    print("Kepochs ={}, Kbatch_size= {}, KLearningRate={:3.6f}, Kl2_regularization_rate ={:3.6f} Model name ={}"
          .format(Kepochs, Kbatch_size, KLearningRate, Kl2_regularization_rate, model_path+model))

    # Download pretrained vgg model
    print("helper.maybe_download_pretrained_vgg({})".format(data_dir))
    helper.maybe_download_pretrained_vgg(data_dir)

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
    training_data_dir = os.path.join(data_dir, 'data_road/training')

    # Create function to get batches
    # get_batches_fn = helper.gen_batch_function(training_data_dir, image_shape)
    # test with cityscapes data
    get_batches_fn = helper.gen_batch_function(data_dir, image_shape)
    print("get_batches_fn = {}".format(get_batches_fn))

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/


    config = tf.ConfigProto()
    tf.log_device_placement=True
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    with tf.Session(config = config) as sess:
        
        # OPTIONAL: Augment Images for better results 
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        if FLAGS.mode == 0:  # Train the model

            print("load_vgg")
            input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

            print("layers")
            last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

            print("optimize")

            logits, train_op, combined_loss, iou_obj = optimize(last_layer, correct_label, learning_rate, num_classes, iou_test=True)

            print("Train!")
            initialized = tf.global_variables_initializer()
            sess.run(initialized)
            
            train_nn(sess, Kepochs, Kbatch_size, get_batches_fn, train_op, combined_loss, input_image, correct_label, 
                    keep_prob, learning_rate, Kl2_regularization_rate, iou_obj=iou_obj, lr=KLearningRate)

            # Save model result
            saver = tf.train.Saver()
            save_path = saver.save(sess, model_path+model)
            print("\nSaved model at {}.".format(save_path))

            # TODO: Save inference data using helper.save_inference_samples
            print("saving samples")
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        elif FLAGS.mode == 1: # run inference on images from kitti dataset
            # Load saved model
            saver = tf.train.import_meta_graph(model_path+model+'.meta')
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            graph = tf.get_default_graph()
            img_input = graph.get_tensor_by_name('image_input:0')
            keep_prob = graph.get_tensor_by_name('keep_prob:0')
            last_layer = graph.get_tensor_by_name('decoder/last_layer:0')
            logits = tf.reshape(last_layer, (-1, num_classes))

            # Process test images
            print("saving samples")
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                                            logits, keep_prob, img_input)
            exit()
            return()

        elif FLAGS.mode == 2: # Run inference on Video file
            def process_frame(sess, logits, keep_prob, image_pl, frame, frame_shape, image_shape):
                """
                Generate output using the video frames
                :param sess: TF session
                :param logits: TF Tensor for the logits
                :param keep_prob: TF Placeholder for the dropout keep robability
                :param image_pl: TF Placeholder for the image placeholder
                :param frame: image frame in 
                :param frame_shape: Tuple - Shape of frame coming in  and going out
                :param image_shape: Tuple - Shape of image used in TF model
                :return: np.array of video frame image with superimposed semantic segmentation
                """
                softmax_criteria = 0.90
                softmax_criteria1 = 0.50
                softmax_criteria2 = 0.25

                # resize to shape used in model
                img_resized = scipy.misc.imresize(frame, image_shape, interp='lanczos')

                # inference with no dropout
                im_softmax = sess.run(
                    [tf.nn.softmax(logits)],
                    {keep_prob: 1.0, img_input: [img_resized]})

                # reshape to image dimensions
                im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
                # apply mask anywhere softmax is > softmax_criteria
                segmentation = (im_softmax > softmax_criteria).reshape(image_shape[0], image_shape[1], 1)
                segmentation1 = (np.logical_and(im_softmax <= softmax_criteria, im_softmax > softmax_criteria1)).reshape(image_shape[0], image_shape[1], 1)
                segmentation2 = (np.logical_and(im_softmax <= softmax_criteria1, im_softmax > softmax_criteria2)).reshape(image_shape[0], image_shape[1], 1)

                # create mask as green and semitransparent
                mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
                mask1 = np.dot(segmentation1, np.array([[0, 225, 0, 63]]))
                mask2 = np.dot(segmentation2, np.array([[0, 200, 0, 31]]))

                mask = scipy.misc.toimage(mask, mode="RGBA")
                mask1 = scipy.misc.toimage(mask1, mode="RGBA")
                mask2 = scipy.misc.toimage(mask2, mode="RGBA")

                mask_resized = scipy.misc.imresize(mask, frame_shape, mode="RGBA")
                mask_resized1 = scipy.misc.imresize(mask1, frame_shape, mode="RGBA")
                mask_resized2 = scipy.misc.imresize(mask2, frame_shape, mode="RGBA")

                mask_resized = scipy.misc.toimage(mask_resized, mode="RGBA")
                mask_resized1 = scipy.misc.toimage(mask_resized1, mode="RGBA")
                mask_resized2 = scipy.misc.toimage(mask_resized2, mode="RGBA")

                frame_im = scipy.misc.toimage(frame)
                frame_im.paste(mask_resized2, box=None, mask=mask_resized2)

                frame_im.paste(mask_resized1, box=None, mask=mask_resized1)
                frame_im.paste(mask_resized, box=None, mask=mask_resized)
                return np.array(frame_im)


            cap = imageio.get_reader('./video/harder_challenge_video.mp4')

            md = cap.get_meta_data()
            fps = float(md['fps'])
            framewidth = int(md['size'][0])
            frameheight = int(md['size'][1])
            framecount = int(md['nframes'])
            frame_shape = (frameheight, framewidth)
            
            print("Video opened with framecount of {:4,d}, dimensions ({:4d},{:4d}), and speed of {:3.03f} fps."
                 .format(framecount, framewidth, frameheight, fps))

            # Load saved model
            saver = tf.train.import_meta_graph(model_path+model+'.meta')
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            graph = tf.get_default_graph()
            img_input = graph.get_tensor_by_name('image_input:0')
            keep_prob = graph.get_tensor_by_name('keep_prob:0')
            last_layer = graph.get_tensor_by_name('decoder/last_layer:0')
            logits = tf.reshape(last_layer, (-1, num_classes))

            fileruntime = datetime.datetime.now().strftime("%Y%m%d%H:%M:%S")
            outfilename = './video/ss_video_output_' + fileruntime + '.mp4'

            out = imageio.get_writer(outfilename, fps=fps)

            frames = 0

            for frame in cap:
                frames += 1
               
                #uncomment for early stop
                #framecount = 150
                
                if frames > framecount:
                    print("\nClosed video after passing expected framecount of {}".format(frames-1))
                    break
                
                out_frame = process_frame(sess, logits, keep_prob, img_input, frame, frame_shape, image_shape)

                out.append_data(out_frame)
                print("Frames: {0:02d}, Seconds: {1:03.03f}".format(frames, frames/fps), end='\r')
                
            print("finished processing video - output video is {}".format(outfilename))
            cap.close()
            out.close()

if __name__ == '__main__':
    FLAGS, _ = parse_args()

    run()
