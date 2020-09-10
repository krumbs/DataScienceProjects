from scripts.utils.image import convert_jpg_to_tfRecords, convert_png_to_jpeg
from config import Config
from argparse import ArgumentParser
from pathlib import Path


cfg = Config()
def create_tf_record(use_jpegs, mode, num_samples):
    if mode == 'train':
        convert_jpg_to_tfRecords(use_jpegs, mode, num_samples, cfg.PATH_TO_IMAGES, cfg.PATH_TO_LABEL_MAP, cfg.PATH_TO_LABELS, Path(cfg.PATH_TO_TFRECORDS, 'kitti_train.record'))

    if mode == 'test':
        convert_jpg_to_tfRecords(use_jpegs, mode, num_samples, cfg.PATH_TO_IMAGES_TEST, cfg.PATH_TO_LABEL_MAP, None, Path(cfg.PATH_TO_TFRECORDS, 'kitti_test.record'))
    print("Prepare data: Done.")

def prepare_data(args):

    mode = args.mode 
    num_samples = args.num_samples
    
    if not mode in ['train', 'test']:
        print("Please choose a valid mode: train or test. Exiting.")
        
    else:
        print(f"Prepare data: creating kitti_{mode}.record.")
        create_tf_record(args.use_jpeg, mode, args.num_samples)


if __name__ == '__main__':
    
    parser = ArgumentParser(description=('Prepare KITTI dataset in TFRecord format.'))

    parser.add_argument('--mode', type=str, default='train', help='create train or test tf record file')
    parser.add_argument('--num-samples', type=int, default=0, help='number of images to record')
    parser.add_argument('--use-jpeg', action='store_true', help='True: delete pngs')
    parser.add_argument('--png-to-jpeg', action='store_true', help='True: convert pngs to jpegs')
    parser.add_argument('--remove-pngs', action='store_true', help='True: delete pngs')
    args = parser.parse_args()

    if args.mode=='train':
        if args.png_to_jpeg:
            for img in cfg.PATH_TO_IMAGES.glob("**/*.png"): 
                convert_png_to_jpeg(img, remove_pngs=args.remove_pngs)
    else:
        if args.png_to_jpeg:
            for img in cfg.PATH_TO_IMAGES_TEST.glob("**/*.png"): 
                convert_png_to_jpeg(img, remove_pngs=args.remove_pngs)

    prepare_data(args)

    
