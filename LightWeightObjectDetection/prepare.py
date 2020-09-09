from scripts.utils.image import convert_png_to_tfRecords
from config import Config
from argparse import ArgumentParser
from pathlib import Path



def create_tf_record(mode):
    cfg = Config()
    if mode == 'train':
        convert_png_to_tfRecords(mode, cfg.PATH_TO_IMAGES, cfg.PATH_TO_LABEL_MAP, cfg.PATH_TO_LABELS, Path(cfg.PATH_TO_TFRECORDS, 'kitti_train.record'))

    if mode == 'test':
        convert_png_to_tfRecords(mode, cfg.PATH_TO_IMAGES_TEST, cfg.PATH_TO_LABEL_MAP, None, Path(cfg.PATH_TO_TFRECORDS, 'kitti_test.record'))
    print("Prepare data: Done.")

def prepare_data(mode='train'):
    if not mode in ['train', 'test']:
        print("Please choose a valid mode: train or test. Exiting.")
        
    else:
        print(f"Prepare data: creating kitti_{mode}.record.")
        create_tf_record(mode)


if __name__ == '__main__':
    
    parser = ArgumentParser(description=('Prepare KITTI dataset in TFRecord format.'))

    parser.add_argument('--mode', type=str, default='train', help='create train or test tf record file')

    args = parser.parse_args()

    prepare_data(args.mode)
