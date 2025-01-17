import matplotlib as mpl
mpl.use('Agg')      # training mode, no screen should be open. (It will block training loop)

import argparse
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from pose_dataset import get_dataflow_batch, DataFlowToQueue, CocoPose
from pose_augment import set_network_input_wh, set_network_scale
from common import get_sample_images
from networks import get_network

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

logger = logging.getLogger('train')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='mobilenet_v2_1.4', help='Model architecture name')
    parser.add_argument('--datapath', type=str, default='/data/public/rw/coco/annotations', help='Path to the "annotations" folder from coco dataset, default "/data/public/rw/coco/annotations"')
    parser.add_argument('--imgpath', type=str, default='/data/public/rw/coco/', help='Path to the "coco" main folder from coco dataset, default "/data/public/rw/coco"')
    parser.add_argument('--batchsize', type=int, default=16, help='Batchsize per GPU, default 16')
    parser.add_argument('--gpus', type=int, default=4, help='Count of GPUs to use, default 4')
    parser.add_argument('--max-epoch', type=int, default=600, help='Maximum number of epochs to continue training, default 600')
    parser.add_argument('--lr', type=str, default='0.0001', help='Start lerning rate, default 0.0001')
    parser.add_argument('--tag', type=str, default='test', help='Tag to append to the models save path, default "test"')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint folder for resuming training from checkpoint, default no checkpoint')
    parser.add_argument('--modelpath', type=str, default='./models/train/', help='Path to save model checkpoints and summarys, default "./models/train/"')

    parser.add_argument('--input-width', type=int, default=432, help='Image input width for training, default 432')
    parser.add_argument('--input-height', type=int, default=368, help='Image input height for training, default 368')
    parser.add_argument('--quant-delay', type=int, default=-1, help='How many steps till quantisation is started, default -1 for no quantisation')
    parser.add_argument('--max_summarys', type=int, default=100, help='How many checkpoints should be kept, default 100')
    parser.add_argument('--save_interval', type=int, default=2000, help='How many steps to save a checkpoint and validate the current model, default 2000')
    parser.add_argument('--log_interval', type=int, default=500, help='How many steps to log statistics from current model, default 500')
    
    args = parser.parse_args()

    modelpath = logpath = args.modelpath

    if args.gpus <= 0:
        raise Exception('gpus <= 0')

    # define input placeholder
    set_network_input_wh(args.input_width, args.input_height)
    scale = 4

    if args.model in ['cmu', 'vgg'] or 'mobilenet' in args.model:
        scale = 8

    set_network_scale(scale)
    output_w, output_h = args.input_width // scale, args.input_height // scale

    logger.info('define model+')
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        #TODO Change to variable batch size
        #for multy gpu set preprocessing batchsize to gpu_count * batchsize
        input_node_queue = tf.placeholder(tf.float32, shape=(args.batchsize*args.gpus, args.input_height, args.input_width, 3), name='image_queue')
        vectmap_node_queue = tf.placeholder(tf.float32, shape=(args.batchsize*args.gpus, output_h, output_w, 38), name='vectmap_queue')
        heatmap_node_queue = tf.placeholder(tf.float32, shape=(args.batchsize*args.gpus, output_h, output_w, 19), name='heatmap_queue')

        #TODO for multy gpu set preprocessing batchsize to gpu_count * batchsize
        #so preprocessed batches split for each gpu = batchsize
        # prepare data
        df = get_dataflow_batch(args.datapath, True, args.batchsize*args.gpus, img_path=args.imgpath)
        enqueuer = DataFlowToQueue(df, [input_node_queue, heatmap_node_queue, vectmap_node_queue], queue_size=100)
        q_inp, q_heat, q_vect = enqueuer.dequeue()

    df_valid = get_dataflow_batch(args.datapath, False, args.batchsize, img_path=args.imgpath)
    df_valid.reset_state()
    validation_cache = []

    val_image = get_sample_images(args.input_width, args.input_height)
    logger.debug('tensorboard val image: %d' % len(val_image))
    logger.debug(q_inp)
    logger.debug(q_heat)
    logger.debug(q_vect)

    # define model for multi-gpu
    # TODO split only works if batchsize dividable by gpu count (4)
    # no works for every count till preprocessing batchsize = gpu_count * batchsize
    input_node = tf.placeholder(tf.float32, shape=(None, args.input_height, args.input_width, 3), name='image')
    vectmap_node = tf.placeholder(tf.float32, shape=(None, output_h, output_w, 38), name='vectmap')
    heatmap_node = tf.placeholder(tf.float32, shape=(None, output_h, output_w, 19), name='heatmap')

    q_inp_split, q_heat_split, q_vect_split = tf.split(input_node, args.gpus), tf.split(heatmap_node, args.gpus), tf.split(vectmap_node, args.gpus)

    output_vectmap = []
    output_heatmap = []
    losses = []
    last_losses_l1 = []
    last_losses_l2 = []
    outputs = []
    for gpu_id in range(args.gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                net, pretrain_path, last_layer = get_network(args.model, q_inp_split[gpu_id])
                if args.checkpoint:
                    pretrain_path = args.checkpoint
                vect, heat = net.loss_last()
                output_vectmap.append(vect)
                output_heatmap.append(heat)
                outputs.append(net.get_output())

                l1s, l2s = net.loss_l1_l2()
                for idx, (l1, l2) in enumerate(zip(l1s, l2s)):
                    loss_l1 = tf.nn.l2_loss(tf.concat(l1, axis=0) - q_vect_split[gpu_id], name='loss_l1_stage%d_tower%d' % (idx, gpu_id))
                    loss_l2 = tf.nn.l2_loss(tf.concat(l2, axis=0) - q_heat_split[gpu_id], name='loss_l2_stage%d_tower%d' % (idx, gpu_id))
                    losses.append(tf.reduce_mean([loss_l1, loss_l2]))

                last_losses_l1.append(loss_l1)
                last_losses_l2.append(loss_l2)

    outputs = tf.concat(outputs, axis=0)

    with tf.device(tf.DeviceSpec(device_type="GPU")):
        # define loss
        total_loss = tf.reduce_sum(losses) / (args.batchsize * args.gpus)
        total_loss_ll_paf = tf.reduce_sum(last_losses_l1) / (args.batchsize * args.gpus)
        total_loss_ll_heat = tf.reduce_sum(last_losses_l2) / (args.batchsize * args.gpus)
        total_loss_ll = tf.reduce_sum([total_loss_ll_paf, total_loss_ll_heat])

        # define optimizer
        #TODO hardcodierte datensatz größe
        step_per_epoch = 121745 // (args.batchsize * args.gpus)
        global_step = tf.Variable(0, trainable=False)
        if ',' not in args.lr:
            starter_learning_rate = float(args.lr)
            # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
            #                                            decay_steps=10000, decay_rate=0.33, staircase=True)
            learning_rate = tf.train.cosine_decay(starter_learning_rate, global_step, args.max_epoch * step_per_epoch, alpha=0.0)
        else:
            lrs = [float(x) for x in args.lr.split(',')]
            boundaries = [step_per_epoch * 5 * i for i, _ in range(len(lrs)) if i > 0]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, lrs)

    if args.quant_delay >= 0:
        logger.info('train using quantized mode, delay=%d' % args.quant_delay)
        g = tf.get_default_graph()
        tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=args.quant_delay)

    # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, momentum=0.9, epsilon=1e-10)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.8, use_locking=True, use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step, colocate_gradients_with_ops=True)
    logger.info('define model-')

    # define summary
    tf.summary.scalar("loss", total_loss)
    tf.summary.scalar("loss_lastlayer", total_loss_ll)
    tf.summary.scalar("loss_lastlayer_paf", total_loss_ll_paf)
    tf.summary.scalar("loss_lastlayer_heat", total_loss_ll_heat)
    tf.summary.scalar("queue_size", enqueuer.size())
    tf.summary.scalar("lr", learning_rate)
    merged_summary_op = tf.summary.merge_all()

    valid_loss = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_paf = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_heat = tf.placeholder(tf.float32, shape=[])
    #TODO hardcoded size
    # replaced with None vor variable size
    sample_train = tf.placeholder(tf.float32, shape=(None, 640, 640, 3))
    sample_valid = tf.placeholder(tf.float32, shape=(None, 640, 640, 3))
    train_img = tf.summary.image('training sample', sample_train, 4)
    valid_img = tf.summary.image('validation sample', sample_valid, 12)
    valid_loss_t = tf.summary.scalar("loss_valid", valid_loss)
    valid_loss_ll_t = tf.summary.scalar("loss_valid_lastlayer", valid_loss_ll)
    merged_validate_op = tf.summary.merge([train_img, valid_img, valid_loss_t, valid_loss_ll_t])

    saver = tf.train.Saver(max_to_keep=args.max_summarys)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        logger.info('model weights initialization')
        sess.run(tf.global_variables_initializer())

        if args.checkpoint and os.path.isdir(args.checkpoint):
            logger.info('Restore from checkpoint...')
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            logger.info('Restore from checkpoint...Done')
        elif pretrain_path:
            logger.info('Restore pretrained weights... %s' % pretrain_path)
            if '.npy' in pretrain_path:
                net.load(pretrain_path, sess, False)
            else:
                try:
                    loader = tf.train.Saver(net.restorable_variables(only_backbone=False))
                    loader.restore(sess, pretrain_path)
                except:
                    logger.info('Restore only weights in backbone layers.')
                    loader = tf.train.Saver(net.restorable_variables())
                    loader.restore(sess, pretrain_path)
            logger.info('Restore pretrained weights...Done')

        logger.info('prepare file writer')
        file_writer = tf.summary.FileWriter(os.path.join(logpath, args.tag), sess.graph)

        logger.info('prepare coordinator')
        coord = tf.train.Coordinator()
        enqueuer.set_coordinator(coord)
        enqueuer.start()

        logger.info('Training Started.')
        time_started = time.time()
        last_gs_num = last_gs_num2 = 0
        initial_gs_num = sess.run(global_step)

        last_log_epoch1 = last_log_epoch2 = -1
        while True:
            q_inp_data, q_heat_data, q_vect_data = sess.run([q_inp, q_heat, q_vect])
            _, gs_num = sess.run([train_op, global_step],
                feed_dict={input_node: q_inp_data, vectmap_node: q_vect_data, heatmap_node: q_heat_data}
            )
            curr_epoch = float(gs_num) / step_per_epoch

            if gs_num > step_per_epoch * args.max_epoch:
                break

            if gs_num - last_gs_num >= args.log_interval:
                train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat, lr_val, summary = sess.run([total_loss, total_loss_ll, total_loss_ll_paf, total_loss_ll_heat, learning_rate, merged_summary_op], 
                    feed_dict={input_node: q_inp_data, vectmap_node: q_vect_data, heatmap_node: q_heat_data}
                )

                # log of training loss / accuracy
                batch_per_sec = (gs_num - initial_gs_num) / (time.time() - time_started)
                logger.info('epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g, loss_ll=%g, loss_ll_paf=%g, loss_ll_heat=%g' % (gs_num / step_per_epoch, gs_num, batch_per_sec * args.batchsize, lr_val, train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat))
                last_gs_num = gs_num

                if last_log_epoch1 < curr_epoch:
                    file_writer.add_summary(summary, curr_epoch)
                    last_log_epoch1 = curr_epoch

            if gs_num - last_gs_num2 >= args.save_interval:
                # save weights
                saver.save(sess, os.path.join(modelpath, args.tag, 'model_latest'), global_step=global_step)

                average_loss = average_loss_ll = average_loss_ll_paf = average_loss_ll_heat = 0
                total_cnt = 0

                if len(validation_cache) == 0:
                    for images_test, heatmaps, vectmaps in tqdm(df_valid.get_data()):
                        validation_cache.append((images_test, heatmaps, vectmaps))
                    df_valid.reset_state()
                    del df_valid
                    df_valid = None

                # log of test accuracy
                for images_test, heatmaps, vectmaps in validation_cache:
                    lss, lss_ll, lss_ll_paf, lss_ll_heat, vectmap_sample, heatmap_sample = sess.run(
                        [total_loss, total_loss_ll, total_loss_ll_paf, total_loss_ll_heat, output_vectmap, output_heatmap],
                        feed_dict={input_node: images_test, vectmap_node: vectmaps, heatmap_node: heatmaps}
                    )
                    average_loss += lss * len(images_test)
                    average_loss_ll += lss_ll * len(images_test)
                    average_loss_ll_paf += lss_ll_paf * len(images_test)
                    average_loss_ll_heat += lss_ll_heat * len(images_test)
                    total_cnt += len(images_test)

                logger.info('validation(%d) %s loss=%f, loss_ll=%f, loss_ll_paf=%f, loss_ll_heat=%f' % (total_cnt, args.tag, average_loss / total_cnt, average_loss_ll / total_cnt, average_loss_ll_paf / total_cnt, average_loss_ll_heat / total_cnt))
                last_gs_num2 = gs_num

                #TODO somthing wrong here max(1, (args.batchsize // 16) -> batchgröße vielfaches von 16
                # 4 sample_image + 12 val_image = 16 images 
                # n mal um batchsize zu erreichen -> aufgerunded batchgröße/bildzahl -> mehr als batchgröße wegen ceil
                # die ersten batchsize elemente nutzen
                ### Change to variable batch size
                # only use 16 image batches
                sample_image = [enqueuer.last_dp[0][i] for i in range(4)]
                test_image = np.array((sample_image + val_image)*args.gpus)
                outputMat = sess.run(
                    outputs,
                    feed_dict={input_node: test_image}
                )
                pafMat, heatMat = outputMat[:, :, :, 19:], outputMat[:, :, :, :19]
                
                #TODO not right size
                #take the min length batchsize vs sample images
                ## solved with var batchsize
                test_results = []
                for image,heat,paf in zip(test_image, heatMat, pafMat):
                    test_result = CocoPose.display_image(image, heat, paf, as_numpy=True)
                    test_result = cv2.resize(test_result, (640, 640))
                    test_result = test_result.reshape([640, 640, 3]).astype(float)
                    test_results.append(test_result)
                
                train_results = test_results[:4]
                val_results = test_results[4:16]

                # save summary
                #TODO not right size
                ## solved var batchsize
                summary = sess.run(merged_validate_op, feed_dict={
                    valid_loss: average_loss / total_cnt,
                    valid_loss_ll: average_loss_ll / total_cnt,
                    valid_loss_ll_paf: average_loss_ll_paf / total_cnt,
                    valid_loss_ll_heat: average_loss_ll_heat / total_cnt,
                    sample_valid: val_results,
                    sample_train: train_results
                })
                if last_log_epoch2 < curr_epoch:
                    file_writer.add_summary(summary, curr_epoch)
                    last_log_epoch2 = curr_epoch

        saver.save(sess, os.path.join(modelpath, args.tag, 'model'), global_step=global_step)
    logger.info('optimization finished. %f' % (time.time() - time_started))
