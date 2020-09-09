

import tensorflow as tf
from config import Config 
from argparse import ArgumentParser
from tensorflow import keras
from pathlib import Path
import numpy as np

cfg = Config()


def extract_test_example_fn(data_record):

    features = {
        # Extract the features using keys set during creation
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64)
        }
            
    sample = tf.io.parse_single_example(data_record, features)
    #image = tf.image.decode_image(sample['image/encoded'])        
    #img_shape = tf.stack([sample['image/height'], sample['image/width'], 3])
    #label = sample['image/object/class/label']
    #filename = sample['image/filename']
    return sample

def extract_example_fn(data_record):

    features = {
        # Extract the features using keys set during creation
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.float32),
        'image/object/class/text': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label':  tf.io.FixedLenFeature([], tf.int64)
    }
            
    sample = tf.io.parse_single_example(data_record, features)
    #image = tf.image.decode_image(sample['image/encoded'])        
    #img_shape = tf.stack([sample['image/height'], sample['image/width'], 3])
    #label = sample['image/object/class/label']
    #filename = sample['image/filename']
    return sample       

def create_input_pipeline(ds, mode, batch_size):

    tune = tf.data.experimental.AUTOTUNE
    # shuffle first to make sure batches contain different samples each epoch
    ds = ds.shuffle(32*batch_size)
    # apply extract function
    if mode=='train':
        ds = ds.map(extract_example_fn, num_parallel_calls=tune)
    else:
        ds = ds.map(extract_test_example_fn, num_parallel_calls=tune)
    # batch with padding
    ds = ds.padded_batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tune)
    return ds

  

def parse_arguments():
    parser = ArgumentParser(
        description=('Train an interleaved LSTM Mobilenet object detection model on the '
                    'KITTI dataset in TFRecord format.'))

    group = parser.add_argument_group(
        title='Training & Validation',
        description='Settings for controlling the training and validation.')
    group.add_argument('--num-epochs', type=int, default=50,
                        help='number of epochs to train')
    group.add_argument('--learning-rate', type=float, default=1e-01,
                        help='base learning rate')
    group.add_argument('--batch-size', type=int, default=2,
                        help='size of batches for training and validation')
    group.add_argument('--training-ratio', type=float, default=0.7,
                        help=('ratio of data that is going to train the '
                            'model, the rest is used for validation, '
                            'if set --num-validation-shards must not '
                            'be set, if neither is set use '
                            '--training-ratio=0.6667'))
    return parser.parse_args()

  
def train():

    args = parse_arguments()

    # load and split dataset
    ds = tf.data.TFRecordDataset(str(Path(cfg.PATH_TO_TFRECORDS,'kitti_train.record')))
    ds_tr = ds.take(int(args.training_ratio*7480))
    ds_va = ds.take(int((1-args.training_ratio)*7480))

    # create input pipelines
    ds_tr = create_input_pipeline(ds_tr, 'train', args.batch_size)
    ds_tr = ds_tr.repeat(args.num_epochs)
    ds_va = create_input_pipeline(ds_va, 'train', args.batch_size)
    ds_va = ds_va.repeat(args.num_epochs)


    # load model backbone
    mobilenet_v2_backbone = tf.keras.applications.MobileNetV2(include_top=False, classes=9 ,input_shape=[None, None, 3])
    out = mobilenet_v2_backbone.output
    # add more layers
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(1024,activation='relu')(out)
    out = tf.keras.layers.Dense(1024,activation='relu')(out)
    out = tf.keras.layers.Dense(512,activation='relu')(out)
    preds = tf.keras.layers.Dense(9, activation='softmax')(out)
    # specify the inputs and outputs
    model = tf.keras.Model(inputs=mobilenet_v2_backbone.input, outputs=preds)
    # specify model
    optimizer = tf.optimizers.SGD(
        learning_rate=args.learning_rate, momentum=0.9)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=[['accuracy'], ['accuracy', 'mse']])
    
    num_epoch_samples_va = int(ds_va.reduce(np.int64(0), lambda x, _: x + 1))
    num_epoch_samples_tr = int(ds_tr.reduce(np.int64(0), lambda x, _: x + 1))

    model.fit(ds_tr, validation_data=ds_va, epochs=ars.num_epochs,
                steps_per_epoch=num_epoch_samples_tr//args.batch_size,
                validation_steps=num_epoch_samples_va//args.batch_size,
                verbose=1)

def main():
    try: 
        train()
    except KeyboardInterrupt:
        print('Training interrupted by user')

if __name__=='__main__':
    main()
  