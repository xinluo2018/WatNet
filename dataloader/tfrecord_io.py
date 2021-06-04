import tensorflow as tf

feaBand = ['blue','green','red','nir','swir1','swir2']
truBand = ['truth']
mergeBand = feaBand+truBand


'''-------------write out-------------'''
def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# Create a dictionary with features that may be relevant.
def image_example(image,truth):
    feature = {
        'bandShape': int64_feature(image[:,:,0].shape),
        'blue': float_feature(image[:,:,0].flatten()),
        'green': float_feature(image[:,:,1].flatten()),
        'red': float_feature(image[:,:,2].flatten()),
        'nir': float_feature(image[:,:,3].flatten()),
        'swir1': float_feature(image[:,:,4].flatten()),
        'swir2': float_feature(image[:,:,5].flatten()),
        'truth': float_feature(truth.flatten()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

'''-------------read in-------------'''
featuresDict = {
    'bandShape': tf.io.FixedLenFeature([2,],dtype=tf.int64),
    'blue': tf.io.VarLenFeature(dtype=tf.float32),
    'green': tf.io.VarLenFeature(dtype=tf.float32),
    'red': tf.io.VarLenFeature(dtype=tf.float32),
    'nir': tf.io.VarLenFeature(dtype=tf.float32),
    'swir1': tf.io.VarLenFeature(dtype=tf.float32),
    'swir2': tf.io.VarLenFeature(dtype=tf.float32),
    'truth': tf.io.VarLenFeature(dtype=tf.float32),
}

def parse_image(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, featuresDict)

def parse_shape(example_parsed):
    for fea in mergeBand:
        example_parsed[fea] = tf.sparse.to_dense(example_parsed[fea])
        example_parsed[fea] = tf.reshape(example_parsed[fea], example_parsed['bandShape'])
    return example_parsed

def toPatchPair(inputs):
    inputsList = [inputs.get(key) for key in mergeBand]
    stacked = tf.stack(inputsList, axis=2)
    cropped_stacked = tf.image.random_crop(
                    stacked, size=[512, 512, len(mergeBand)])
    return cropped_stacked[:,:,:len(feaBand)], cropped_stacked[:,:,len(feaBand):]

