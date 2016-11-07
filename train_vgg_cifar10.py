import tensorflow as tf

from models.slim.deployment import model_deploy
from models.slim.preprocessing import preprocessing_factory
from models.slim.datasets import dataset_factory
from tensorflow.python.ops import control_flow_ops

#import vgg_cifar10
#import vgg_cifar10_vgg9
#import vgg_cifar10_vgg13
import vgg_cifar10_vgg16

slim = tf.contrib.slim
# tf.reset_default_graph()

EXPERIMENT_NAME = 'VGG11B_pretrain'
DATA_DIR = '/home/ziba/DataSets/cifar10'
TRAIN_DIR = '/home/ziba/WVU/tfResults/vgg_cifar10_results/Structures/VGG16/' + EXPERIMENT_NAME + '/train'
BATCH_SIZE = 256
INIT_LEARNING_RATE = 0.0005
LR_DECAY = 0.99
NUM_EPOCHS_PER_DECAY = 2
MAX_STEPS = 14000
NUM_CLONES = 1
EXCLUDED_SCOPES = (#'vgg_16/conv2/conv2_2',
                   #'vgg_16/conv3/conv3_2',
                   'vgg_16/conv3/conv3_3',
                   #'vgg_16/conv4/conv4_2',
                   'vgg_16/conv4/conv4_3',
                   'vgg_16/conv5/conv5_3',
                   'vgg_16/fc6',
                   'vgg_16/fc7',
                   'vgg_16/fc8')
CHECKPOINT_PATH = '/home/ziba/WVU/tfmodels/mine/vgg-cifar10/pretrained_weights/vgg_16.ckpt'

tf.logging.set_verbosity(tf.logging.INFO)
deploy_config = model_deploy.DeploymentConfig(num_clones=NUM_CLONES)

with tf.Graph().as_default() as g:
    with tf.device(deploy_config.variables_device()):
        global_step = slim.create_global_step()

    dataset = dataset_factory.get_dataset('cifar10', 'train', DATA_DIR)

    network_fn = vgg_cifar10_vgg16.inference

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        'cifarnet', is_training=True)

    with tf.device(deploy_config.inputs_device()):
        with tf.name_scope('inputs'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                  dataset,
                  num_readers=6,
                  common_queue_capacity=20 * BATCH_SIZE,
                  common_queue_min=10 * BATCH_SIZE)
            [image, label] = provider.get(['image', 'label'])

            image = image_preprocessing_fn(image, 32, 32)
            images, labels = tf.train.batch(
                  [image, label],
                  batch_size=BATCH_SIZE,
                  num_threads=10,
                  capacity=5 * BATCH_SIZE)
            labels = slim.one_hot_encoding(labels, 10)

            batch_queue = slim.prefetch_queue.prefetch_queue(
                  [images, labels], capacity=2 * deploy_config.num_clones)


    def clone_fn(batch_queue):
        images, labels = batch_queue.dequeue()
        logits, end_points = network_fn(images)
        slim.losses.softmax_cross_entropy(logits, labels)
        predictions = tf.argmax(logits, 1)
        labels = tf.argmax(labels, 1)
        accuracy, update_op = slim.metrics.streaming_accuracy(
           predictions,
           labels,
           metrics_collections=['accuracy'],
           updates_collections=tf.GraphKeys.UPDATE_OPS)
        return end_points

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    with tf.name_scope('synchronization'):
        with tf.device(deploy_config.optimizer_device()):
            decay_steps = int(dataset.num_samples / BATCH_SIZE *
                              NUM_EPOCHS_PER_DECAY)
            learning_rate = tf.train.exponential_decay(
                INIT_LEARNING_RATE,
                global_step,
                decay_steps,
                LR_DECAY,
                staircase=True,
                name='exponential_decay_learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate)
        variables_to_train = tf.trainable_variables()
        total_loss, clones_gradients = model_deploy.optimize_clones(
                clones,
                optimizer,
                var_list=variables_to_train)
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                          name='train_op')
    with tf.name_scope('summaries'):
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.histogram_summary('activations/' + end_point, x))
            summaries.add(tf.scalar_summary('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))
        for variable in slim.get_model_variables():
            summaries.add(tf.histogram_summary(variable.op.name, variable))
        summaries.add(tf.scalar_summary('learning_rate', learning_rate,
                                        name='learning_rate'))
        summaries.add(tf.scalar_summary('eval/total_loss', total_loss,
                                        name='total_loss_summary'))
        accuracy = tf.get_collection('accuracy', first_clone_scope)[0]
        summaries.add(tf.scalar_summary('eval/accuracy', accuracy))
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))
        summary_op = tf.merge_summary(list(summaries), name='summary_op')


    def init_fn():
        if CHECKPOINT_PATH is None:
            return None
        if tf.train.latest_checkpoint(TRAIN_DIR):
            tf.logging.info(
             'Ignoring --checkpoint_path because a checkpoint already exists in %s'
             % TRAIN_DIR)
            return None
        if not EXCLUDED_SCOPES:
            return None
        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in EXCLUDED_SCOPES:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
        tf.logging.info('Fine-tuning from %s' % CHECKPOINT_PATH)
        return slim.assign_from_checkpoint_fn(CHECKPOINT_PATH,
                                              variables_to_restore)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    slim.learning.train(
        train_tensor,
        TRAIN_DIR,
        summary_op=summary_op,
        number_of_steps=MAX_STEPS,
        log_every_n_steps=50,
        save_summaries_secs=60,
        save_interval_secs=60*3,
        init_fn=init_fn(),
        session_config=sess_config)
