import os
import io

import tensorflow as tf
from google.protobuf import text_format
from pathlib import Path
from PIL import Image


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_list_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_png_to_jpeg(path_to_image, remove_pngs: bool):
	format_new_name = str(path_to_image)[0:-4] +'.jpg'

	print(f"FORMATED NAME {format_new_name}")
	image = Image.open(path_to_image)
	image_rgb = image.convert('RGB')
	image_rgb.save(format_new_name)

	if remove_pngs:
		os.remove(path_to_image)


def convert_jpg_to_tfRecords(use_jpegs, mode, num_samples, path_to_images, path_to_label_map, path_to_labels, path_to_tf_records):
	
	# get list of image names
	image_path = []
	if use_jpegs:
		for img in path_to_images.glob("**/*.jpg"):
			image_path.append(str(img))
	else:
		for img in path_to_images.glob("**/*.png"):
			image_path.append(str(img))
	
	# start tf writer
	writer = tf.io.TFRecordWriter(path_to_tf_records.as_posix())

	for item in range(num_samples):
		print(f'Prepare data: {image_path[item]}')
		# Read image data in terms of bytes
		with tf.io.gfile.GFile(image_path[item], 'rb') as fid:
			img_bytes = fid.read()
		
		
		img_io = io.BytesIO(img_bytes)
		image = Image.open(img_io)
		width, height = image.size
			
		if mode=='train':
			# get list of label names
			label_path = []
			for txt in path_to_labels.glob("**/*.txt"):
				label_path.append(str(txt))

			label_dict = {
				'Car': 1,
				'Van': 2,
				'Pedestrian': 3,
				'Cyclist': 4,
				'Tram': 5,
				'Person_sitting': 6,
				'Truck': 7,
				'Misc': 8,
				'DontCare': 0
			}

			"""
			#Values    Name         Description
			----------------------------------------------------------------------------
			1           type        Describes the type of object: 'Car', 'Van', 'Truck',
									'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
									'Misc' or 'DontCare'
			1           truncated   Float from 0 (non-truncated) to 1 (truncated), where
									truncated refers to the object leaving image boundaries
			1           occluded    Integer (0,1,2,3) indicating occlusion state:
									0 = fully visible, 1 = partly occluded
									2 = largely occluded, 3 = unknown
			1           alpha       Observation angle of object, ranging [-pi..pi]
			4           bbox        2D bounding box of object in the image (0-based index):
									contains left, top, right, bottom pixel coordinates
			3           dimensions  3D object dimensions: height, width, length (in meters)
			3           location    3D object location x,y,z in camera coordinates (in meters)
			1           rotation_y  Rotation ry around Y-axis in camera coordinates [-pi..pi]
			1           score       Only for results: Float, indicating confidence in
									detection, needed for p/r curves, higher is better.
			
			example
			Pedestrian    0.00      0          -0.20   712.40 143.00 810.73 307.92   1.89 0.48 1.20      1.84 1.47 8.41      0.01
			| type     | truncated | occluded | alpha | bbox                        | dimensions        | location          | location_y
			"""
			# Note: ignoring some of the data
			class_types = []
			class_labels = []
			truncateds = []
			occludeds = []
			bboxes = []
			xmins = []
			xmaxs = []
			ymins = []
			ymaxs = []
			l_file = open(label_path[item], 'r')

			# fetch the relevant part of each line and append it to respecitive list
			for line in l_file.readlines():
				parts = line.split()
				class_types.append(parts[0].encode('utf-8'))
				#truncateds.append(parts[1])
				#occludeds.append(parts[2])

				xmins.append(float(parts[4]))
				ymaxs.append(float(parts[5]))
				xmaxs.append(float(parts[6]))
				ymins.append(float(parts[7]))
				class_labels.append(label_dict[parts[0]])
				
			# create the tf example for image and associated labels
			tf_example = tf.train.Example(features=tf.train.Features(
				feature={
					'image/encoded': _bytes_feature(img_bytes),
					'image/filename': _bytes_feature(image_path[item].encode('utf-8')),
					'image/height': _int64_feature(height),
					'image/width': _int64_feature(width),
					'image/object/bbox/xmin': _float_list_feature(xmins),
					'image/object/bbox/xmax': _float_list_feature(xmaxs),
					'image/object/bbox/ymin': _float_list_feature(ymins),
					'image/object/bbox/ymax': _float_list_feature(ymaxs),
					'image/object/class/text': _bytes_list_feature(class_types),
					'image/object/class/label': _int64_list_feature(class_labels)
					}))
			writer.write(tf_example.SerializeToString())
			
		else:
			# create the tf example for image and associated labels
			tf_example = tf.train.Example(features=tf.train.Features(
							feature={
								'image/encoded': _bytes_feature(png_bytes),
								'image/filename': _bytes_feature(image_path[item].encode('utf-8')),
								'image/height': _int64_feature(height),
								}))

			writer.write(tf_example.SerializeToString())
	writer.close()

