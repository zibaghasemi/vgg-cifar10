import tensorflow as tf
import math
from models.slim.datasets import dataset_factory
from models.slim.preprocessing import preprocessing_factory

#import vgg_cifar10
#import vgg_cifar10_vgg9
#import vgg_cifar10_vgg13
import vgg_cifar10_vgg16


slim = tf.contrib.slim

EXPERIMENT_NAME = 'VGG11B_pretrain'
BATCH_SIZE = 256
CHECKPOINT_DIR = '/home/ziba/WVU/tfResults/vgg_cifar10_results/Structures/VGG16/' + EXPERIMENT_NAME + '/train'
EVAL_DIR = CHECKPOINT_DIR[:-5] + 'test'
DATA_DIR = '/home/ziba/DataSets/cifar10'
EVAL_DEVICE = '/cpu:0'

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default() as g:
    with g.device(EVAL_DEVICE):
        dataset = dataset_factory.get_dataset('cifar10', 'test', DATA_DIR)

        tf_global_step = slim.get_or_create_global_step()

        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * BATCH_SIZE,
            common_queue_min=BATCH_SIZE)

        [image, label] = provider.get(['image', 'label'])

        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            'cifarnet',
            is_training=False)

        image = image_preprocessing_fn(image, 32, 32)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=BATCH_SIZE,
            num_threads=2,
            capacity=5 * BATCH_SIZE)

        logits, end_points = vgg_cifar10_vgg16.inference(images)

        predictions = tf.argmax(logits, 1)

        accuracy, update_op = slim.metrics.streaming_accuracy(predictions,
                                                              labels)
        tf.scalar_summary('eval/accuracy', accuracy)
        summary_op = tf.merge_all_summaries()

        num_batches = math.ceil(dataset.num_samples / float(BATCH_SIZE))

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        slim.evaluation.evaluation_loop(
            master='',
            checkpoint_dir=CHECKPOINT_DIR,
            logdir=EVAL_DIR,
            num_evals=num_batches,
            eval_op=update_op,
            eval_interval_secs=160,
            session_config=sess_config,
            variables_to_restore=slim.get_variables_to_restore())
