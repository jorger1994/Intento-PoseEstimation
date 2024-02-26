import json
import os
import tensorflow as tf

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  if type(value) == list:
    value = value
  else:
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def parse_features(image_str, labels):

    allkeys = ['cx0_player1','cx1_player1','cx2_player1','cx3_player1','cx4_player1','cx5_player1','cx6_player1', 'cx7_player1', 'cx8_player1', 'cx9_player1', 'cx10_player1', 'cx11_player1',
               'cx12_player1', 'cx13_player1', 'cx14_player1', 'cx15_player1', 'cx16_player1',
               'cy0_player1','cy1_player1','cy2_player1','cy3_player1','cy4_player1','cy5_player1','cy6_player1', 'cy7_player1', 'cy8_player1', 'cy9_player1', 'cy10_player1', 'cy11_player1',
               'cy12_player1', 'cy13_player1', 'cy14_player1', 'cy15_player1', 'cy16_player1',
               'visibility0_player1', 'visibility1_player1', 'visibility2_player1', 'visibility3_player1', 'visibility4_player1', 'visibility5_player1', 'visibility6_player1', 
               'visibility7_player1', 'visibility8_player1', 'visibility9_player1', 'visibility10_player1', 'visibility11_player1', 'visibility12_player1', 
               'visibility13_player1', 'visibility14_player1', 'visibility15_player1', 'visibility16_player1',
               'cx0_player2','cx1_player2','cx2_player2','cx3_player2','cx4_player2','cx5_player2','cx6_player2', 'cx7_player2', 'cx8_player2', 'cx9_player2', 'cx10_player2', 'cx11_player2',
               'cx12_player2', 'cx13_player2', 'cx14_player2', 'cx15_player2', 'cx16_player2',
               'cy0_player2','cy1_player2','cy2_player2','cy3_player2','cy4_player2','cy5_player2','cy6_player2', 'cy7_player2', 'cy8_player2', 'cy9_player2', 'cy10_player2', 'cy11_player2',
               'cy12_player2', 'cy13_player2', 'cy14_player2', 'cy15_player2', 'cy16_player2',
               'visibility0_player2', 'visibility1_player2', 'visibility2_player2', 'visibility3_player2', 'visibility4_player2', 'visibility5_player2', 'visibility6_player2', 
               'visibility7_player2', 'visibility8_player2', 'visibility9_player2', 'visibility10_player2', 'visibility11_player2', 'visibility12_player2', 
               'visibility13_player2', 'visibility14_player2', 'visibility15_player2', 'visibility16_player2']
    available_labels = list(labels.keys())

    for x in allkeys:
        if x in available_labels:
            pass
        else:
            labels[x] = 0

    img_feature = _bytes_feature(image_str)
    
    img_shape = tf.io.decode_png(image_str, 4).shape
    img_shape_feature = _float_feature(img_shape)

    cx0f_player1 = _int64_feature(labels['cx0_player1'])
    cx1f_player1 = _int64_feature(labels['cx1_player1'])
    cx2f_player1 = _int64_feature(labels['cx2_player1'])
    cx3f_player1 = _int64_feature(labels['cx3_player1'])
    cx4f_player1 = _int64_feature(labels['cx4_player1'])
    cx5f_player1 = _int64_feature(labels['cx5_player1'])
    cx6f_player1 = _int64_feature(labels['cx6_player1'])
    cx7f_player1 = _int64_feature(labels['cx7_player1'])
    cx8f_player1 = _int64_feature(labels['cx8_player1'])
    cx9f_player1 = _int64_feature(labels['cx9_player1'])
    cx10f_player1 = _int64_feature(labels['cx10_player1'])
    cx11f_player1 = _int64_feature(labels['cx11_player1'])
    cx12f_player1 = _int64_feature(labels['cx12_player1'])
    cx13f_player1 = _int64_feature(labels['cx13_player1'])
    cx14f_player1 = _int64_feature(labels['cx14_player1'])
    cx15f_player1 = _int64_feature(labels['cx15_player1'])
    cx16f_player1 = _int64_feature(labels['cx16_player1'])

    cy0f_player1 = _int64_feature(labels['cy0_player1'])
    cy1f_player1 = _int64_feature(labels['cy1_player1'])
    cy2f_player1 = _int64_feature(labels['cy2_player1'])
    cy3f_player1 = _int64_feature(labels['cy3_player1'])
    cy4f_player1 = _int64_feature(labels['cy4_player1'])
    cy5f_player1 = _int64_feature(labels['cy5_player1'])
    cy6f_player1 = _int64_feature(labels['cy6_player1'])
    cy7f_player1 = _int64_feature(labels['cy7_player1'])
    cy8f_player1 = _int64_feature(labels['cy8_player1'])
    cy9f_player1 = _int64_feature(labels['cy9_player1'])
    cy10f_player1 = _int64_feature(labels['cy10_player1'])
    cy11f_player1 = _int64_feature(labels['cy11_player1'])
    cy12f_player1 = _int64_feature(labels['cy12_player1'])
    cy13f_player1 = _int64_feature(labels['cy13_player1'])
    cy14f_player1 = _int64_feature(labels['cy14_player1'])
    cy15f_player1 = _int64_feature(labels['cy15_player1'])
    cy16f_player1 = _int64_feature(labels['cy16_player1'])
    
    v0f_player1 = _int64_feature(labels['visibility0_player1'])
    v1f_player1 = _int64_feature(labels['visibility1_player1'])
    v2f_player1 = _int64_feature(labels['visibility2_player1'])
    v3f_player1 = _int64_feature(labels['visibility3_player1'])
    v4f_player1 = _int64_feature(labels['visibility4_player1'])
    v5f_player1 = _int64_feature(labels['visibility5_player1'])
    v6f_player1 = _int64_feature(labels['visibility6_player1'])
    v7f_player1 = _int64_feature(labels['visibility7_player1'])
    v8f_player1 = _int64_feature(labels['visibility8_player1'])
    v9f_player1 = _int64_feature(labels['visibility9_player1'])
    v10f_player1 = _int64_feature(labels['visibility10_player1'])
    v11f_player1 = _int64_feature(labels['visibility11_player1'])
    v12f_player1 = _int64_feature(labels['visibility12_player1'])
    v13f_player1 = _int64_feature(labels['visibility13_player1'])
    v14f_player1 = _int64_feature(labels['visibility14_player1'])
    v15f_player1 = _int64_feature(labels['visibility15_player1'])
    v16f_player1 = _int64_feature(labels['visibility16_player1'])

    cx0f_player2 = _int64_feature(labels['cx0_player2'])
    cx1f_player2 = _int64_feature(labels['cx1_player2'])
    cx2f_player2 = _int64_feature(labels['cx2_player2'])
    cx3f_player2 = _int64_feature(labels['cx3_player2'])
    cx4f_player2 = _int64_feature(labels['cx4_player2'])
    cx5f_player2 = _int64_feature(labels['cx5_player2'])
    cx6f_player2 = _int64_feature(labels['cx6_player2'])
    cx7f_player2 = _int64_feature(labels['cx7_player2'])
    cx8f_player2 = _int64_feature(labels['cx8_player2'])
    cx9f_player2 = _int64_feature(labels['cx9_player2'])
    cx10f_player2 = _int64_feature(labels['cx10_player2'])
    cx11f_player2 = _int64_feature(labels['cx11_player2'])
    cx12f_player2 = _int64_feature(labels['cx12_player2'])
    cx13f_player2 = _int64_feature(labels['cx13_player2'])
    cx14f_player2 = _int64_feature(labels['cx14_player2'])
    cx15f_player2 = _int64_feature(labels['cx15_player2'])
    cx16f_player2 = _int64_feature(labels['cx16_player2'])

    cy0f_player2 = _int64_feature(labels['cy0_player2'])
    cy1f_player2 = _int64_feature(labels['cy1_player2'])
    cy2f_player2 = _int64_feature(labels['cy2_player2'])
    cy3f_player2 = _int64_feature(labels['cy3_player2'])
    cy4f_player2 = _int64_feature(labels['cy4_player2'])
    cy5f_player2 = _int64_feature(labels['cy5_player2'])
    cy6f_player2 = _int64_feature(labels['cy6_player2'])
    cy7f_player2 = _int64_feature(labels['cy7_player2'])
    cy8f_player2 = _int64_feature(labels['cy8_player2'])
    cy9f_player2 = _int64_feature(labels['cy9_player2'])
    cy10f_player2 = _int64_feature(labels['cy10_player2'])
    cy11f_player2 = _int64_feature(labels['cy11_player2'])
    cy12f_player2 = _int64_feature(labels['cy12_player2'])
    cy13f_player2 = _int64_feature(labels['cy13_player2'])
    cy14f_player2 = _int64_feature(labels['cy14_player2'])
    cy15f_player2 = _int64_feature(labels['cy15_player2'])
    cy16f_player2 = _int64_feature(labels['cy16_player2'])
    
    v0f_player2 = _int64_feature(labels['visibility0_player2'])
    v1f_player2 = _int64_feature(labels['visibility1_player2'])
    v2f_player2 = _int64_feature(labels['visibility2_player2'])
    v3f_player2 = _int64_feature(labels['visibility3_player2'])
    v4f_player2 = _int64_feature(labels['visibility4_player2'])
    v5f_player2 = _int64_feature(labels['visibility5_player2'])
    v6f_player2 = _int64_feature(labels['visibility6_player2'])
    v7f_player2 = _int64_feature(labels['visibility7_player2'])
    v8f_player2 = _int64_feature(labels['visibility8_player2'])
    v9f_player2 = _int64_feature(labels['visibility9_player2'])
    v10f_player2 = _int64_feature(labels['visibility10_player2'])
    v11f_player2 = _int64_feature(labels['visibility11_player2'])
    v12f_player2 = _int64_feature(labels['visibility12_player2'])
    v13f_player2 = _int64_feature(labels['visibility13_player2'])
    v14f_player2 = _int64_feature(labels['visibility14_player2'])
    v15f_player2 = _int64_feature(labels['visibility15_player2'])
    v16f_player2 = _int64_feature(labels['visibility16_player2'])


    data = {
        'img_feature': img_feature,
        'img_shape_feature' : img_shape_feature,
        'cx0f_player1': cx0f_player1, 
        'cx1f_player1': cx1f_player1, 
        'cx2f_player1': cx2f_player1, 
        'cx3f_player1': cx3f_player1, 
        'cx4f_player1': cx4f_player1, 
        'cx5f_player1': cx5f_player1, 
        'cx6f_player1': cx6f_player1, 
        'cx7f_player1': cx7f_player1, 
        'cx8f_player1': cx8f_player1, 
        'cx9f_player1': cx9f_player1, 
        'cx10f_player1': cx10f_player1, 
        'cx11f_player1': cx11f_player1, 
        'cx12f_player1': cx12f_player1,
        'cx13f_player1': cx13f_player1, 
        'cx14f_player1': cx14f_player1, 
        'cx15f_player1': cx15f_player1, 
        'cx16f_player1': cx16f_player1,
        'cy0f_player1': cy0f_player1, 
        'cy1f_player1': cy1f_player1, 
        'cy2f_player1': cy2f_player1, 
        'cy3f_player1': cy3f_player1, 
        'cy4f_player1': cy4f_player1, 
        'cy5f_player1': cy5f_player1, 
        'cy6f_player1': cy6f_player1, 
        'cy7f_player1': cy7f_player1, 
        'cy8f_player1': cy8f_player1, 
        'cy9f_player1': cy9f_player1, 
        'cy10f_player1': cy10f_player1, 
        'cy11f_player1': cy11f_player1, 
        'cy12f_player1': cy12f_player1,
        'cy13f_player1': cy13f_player1, 
        'cy14f_player1': cy14f_player1, 
        'cy15f_player1': cy15f_player1, 
        'cy16f_player1': cy16f_player1, 
        'v0f_player1': v0f_player1,
        'v1f_player1': v1f_player1, 
        'v2f_player1': v2f_player1, 
        'v3f_player1': v3f_player1, 
        'v4f_player1': v4f_player1, 
        'v5f_player1': v5f_player1, 
        'v6f_player1': v6f_player1, 
        'v7f_player1': v7f_player1, 
        'v8f_player1': v8f_player1, 
        'v9f_player1': v9f_player1, 
        'v10f_player1': v10f_player1, 
        'v11f_player1': v11f_player1, 
        'v12f_player1': v12f_player1, 
        'v13f_player1': v13f_player1, 
        'v14f_player1': v14f_player1,
        'v15f_player1': v15f_player1,
        'v16f_player1': v16f_player1,

        'cx0f_player2': cx0f_player2, 
        'cx1f_player2': cx1f_player2, 
        'cx2f_player2': cx2f_player2, 
        'cx3f_player2': cx3f_player2, 
        'cx4f_player2': cx4f_player2, 
        'cx5f_player2': cx5f_player2, 
        'cx6f_player2': cx6f_player2, 
        'cx7f_player2': cx7f_player2, 
        'cx8f_player2': cx8f_player2, 
        'cx9f_player2': cx9f_player2, 
        'cx10f_player2': cx10f_player2, 
        'cx11f_player2': cx11f_player2, 
        'cx12f_player2': cx12f_player2,
        'cx13f_player2': cx13f_player2, 
        'cx14f_player2': cx14f_player2, 
        'cx15f_player2': cx15f_player2, 
        'cx16f_player2': cx16f_player2,
        'cy0f_player2': cy0f_player2, 
        'cy1f_player2': cy1f_player2, 
        'cy2f_player2': cy2f_player2, 
        'cy3f_player2': cy3f_player2, 
        'cy4f_player2': cy4f_player2, 
        'cy5f_player2': cy5f_player2, 
        'cy6f_player2': cy6f_player2, 
        'cy7f_player2': cy7f_player2, 
        'cy8f_player2': cy8f_player2, 
        'cy9f_player2': cy9f_player2, 
        'cy10f_player2': cy10f_player2, 
        'cy11f_player2': cy11f_player2, 
        'cy12f_player2': cy12f_player2,
        'cy13f_player2': cy13f_player2, 
        'cy14f_player2': cy14f_player2, 
        'cy15f_player2': cy15f_player2, 
        'cy16f_player2': cy16f_player2, 
        'v0f_player2': v0f_player2,
        'v1f_player2': v1f_player2, 
        'v2f_player2': v2f_player2, 
        'v3f_player2': v3f_player2, 
        'v4f_player2': v4f_player2, 
        'v5f_player2': v5f_player2, 
        'v6f_player2': v6f_player2, 
        'v7f_player2': v7f_player2, 
        'v8f_player2': v8f_player2, 
        'v9f_player2': v9f_player2, 
        'v10f_player2': v10f_player2, 
        'v11f_player2': v11f_player2, 
        'v12f_player2': v12f_player2, 
        'v13f_player2': v13f_player2, 
        'v14f_player2': v14f_player2,
        'v15f_player2': v15f_player2,
        'v16f_player2': v16f_player2,

    }

    result = tf.train.Example(features = tf.train.Features(feature = data))

    return result

def create_tf_record(examples, name):
    
    filename = name+'.tfrecords'
    writer = tf.io.TFRecordWriter(filename)
    
    for example in examples:
        writer.write(example.SerializeToString())
        
    writer.close()
    print('Job Done')

def create_records(label_file, image_folder, record_name, destination_dir):
    
    """ 
    The label_file is the json file with the annotations
    the image_folder is the folder where the input images are located - it has to finish with a '/'
    the record_name is the name you want the tfrecord file to have
    """

    # Get the name and the extentions of the images
    img_names = os.listdir(image_folder)
    img_files = [image_folder + '/' + name for name in img_names]
    img_extentions = []
    for file in img_files:
        _, ext = os.path.splitext(file)
        if ext in img_extentions:
            pass
        else:
            img_extentions.append(ext)

    # Read the json file and create the labels dictionary
    with open(label_file, 'r') as f:
        data = json.load(f)
        label_names = list(data.keys())
        examples = []
        
        for name in label_names:
            # This allows us to get the corresponding image shape
            for ext in img_extentions:
                try:
                    short_label_name = name.split(ext)[0] + ext
                except:
                    pass
            
            if short_label_name in img_names:
                idx = img_names.index(short_label_name)
                img = img_files[idx]
                img_string = open(img, 'rb').read()
            else:
                print(f'The label name {short_label_name} was not found on the folder')

            # We gather the X_coordinate, Y_coordinate, the type of joint and the visibility from the json file.
            regions = data[name]
            regions = regions['regions']
            labels = {}
            joints_count = []
            n = str(1)
            for region in regions:
                joint_type = int(region['region_attributes']['Joints'])
                if joint_type in joints_count:
                    n = str(2)
                if joint_type == str(17):
                    n = str(2)
                    pass
                joints_count.append(joint_type)
                cx_str = 'cx'+str(joint_type)+'_player'+n
                cy_str = 'cy'+str(joint_type)+'_player'+n
                visibility_str = 'visibility'+str(joint_type)+'_player'+n

                labels[cx_str] = int(region['shape_attributes']['cx'])
                labels[cy_str] = int(region['shape_attributes']['cy'])
                labels[visibility_str] = int(region['region_attributes']['Visibility'])
            
            example = parse_features(image_str=img_string, labels=labels)
            examples.append(example)
        
        example = create_tf_record(examples = examples, name = record_name)
        f.close()

        # Move the file to the destination folder
        current_dir = os.getcwd()
        current_tfrecord = os.path.join(current_dir, f'{record_name}.tfrecords')
        destination_tfrecord = os.path.join(destination_dir, f'{record_name}.tfrecords')
        os.rename(current_tfrecord, destination_tfrecord)

    print(f'The record file has been created with the following name {record_name}.tfrecords')
    print(f'The file has been stored in {destination_tfrecord}')