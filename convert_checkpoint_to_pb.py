import os
import argparse
import logging
import tensorflow as tf
from tf_pose.networks import get_network, model_wh, load_variables

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


if __name__ == '__main__':
    """
    Use this script to just save graph and checkpoint.
    While training, checkpoints are saved. You can test them with this python code.
    """
    parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0')
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--cp_path', type=str, default='', help='path to checkpoint file')
    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w <= 0 or h <= 0:
        w = h = None
    print(w, h)
    input_node = tf.placeholder(tf.float32, shape=(None, h, w, 3), name='image')

    net, pretrain_path, last_layer = get_network(args.model, input_node, None, trainable=False)
    
    if args.cp_path:
        pretrain_path = args.cp_path
    if args.quantize:
        g = tf.get_default_graph()
        tf.contrib.quantize.create_eval_graph(input_graph=g)
    
    output_node_names = ["Openpose/concat_stage7"]

    with tf.Session(config=config) as sess:
        load_variables(sess, net, pretrain_path)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names # The output node names are used to select the usefull nodes
        )

        output_graph_file = os.path.join('./converted/'+args.model,args.model+args.resize+'.pb')

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph_file, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        flops = tf.profiler.profile(None, cmd='graph', options=tf.profiler.ProfileOptionBuilder.float_operation())
        print('FLOP = ', flops.total_float_ops / float(1e6))