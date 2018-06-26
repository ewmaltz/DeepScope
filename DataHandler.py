import tensorflow as tf
import numpy as np
import sys
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import filters, exposure

"""
Some Notes About Creating Datasets
* Dataset does not have to load all of the images, just the paths
    * need a helper function to then load the images from the disk

TFRecords converts the data into a highly efficient, binary file format. Split the files to prevent the data from 
getting too large. Only reads a small amount of the data from the file into RAM.
"""


# Helper Functions
def get_stack_paths(stack_root_path):
    stack_paths = os.listdir(stack_root_path)
    return [pos_name for pos_name in stack_paths if pos_name[:3] == 'Pos']  # removes flt, metadata, etc.


def load_stacks_as_images(stack_root_path, stack_ixs, channel, pretty=False):
    """
    Converting stacks of 2D images to individual 3D images, depth is encoded as colors in a color channel.
    :param stack_root_path: root folder for stacks
    :param stack_ixs: [start, end]
    :param pretty: histogram equalization to make the images easier to see
    :param channel: string format; DeepBlue, Red, Yellow, Green, Brightfield, etc.
    :return: numpy array of the images (n_stacks, img_h, img_w, n_zpositions), true focus is middle position
    """
    # Input Error Handling
    if '/acq' not in stack_root_path:
        raise ImportError('stack_root_path must include the name of the acquisition you want.')
    if type(stack_ixs) != list or stack_ixs[0] >= stack_ixs[1]:
        raise ImportError('stack_ixs should be a list: [start_ix:end_ix]')

    # Get paths for later, make sure that the number of stacks you pull is <= total stacks
    stack_paths = get_stack_paths(stack_root_path)
    max_stacks = len(stack_paths)
    n_stacks = min(stack_ixs[1]-stack_ixs[0], max_stacks)

    # load the specified number of stacks into list and return as array
    stacks = []
    for j, pos_name in enumerate(stack_paths[stack_ixs[0]:stack_ixs[1]]):  # start/end --> arbitrary stack calls
        print('Loading %d out of %d. %.2f%% complete.' % (j+1, n_stacks, 100*(j+1)/n_stacks))
        stack = []
        for img_name in os.listdir(stack_root_path+'/'+pos_name):
            if channel in img_name:
                stack.append(np.array(plt.imread(stack_root_path+'/'+pos_name+'/'+img_name)))
        stacks.append(np.array(stack))
    print('Done!')

    # now reshape stacks: (n_stacks, img_h, img_w, n_zpositions), n_zpositions = n_channels
    stacks = [stack.T for stack in stacks]
    stacks = np.array(stacks, np.uint16)
    print(stacks.shape)

    if pretty:
        # remove some noise, not working well and takes forever (perhaps use the real flt path?)
        stacks = exposure.equalize_hist(stacks)

    return stacks


def show_stack_animation(stack, speed=10, repeat_delay=200):
    """
    Matplotlib pop-out animation to show zstack
    :param stack: np array of z stack
    :param speed: number of milliseconds between frames
    :param repeat_delay: time in milliseconds between loops
    :return: None
    """
    fig = plt.figure()
    ims = []
    for i in range(stack.shape[-1]):
        im = plt.imshow(stack[:, :, i], animated=True)  # z-stack treated like color channels
        t = plt.annotate(str(i), (20, 75), bbox=dict(facecolor='white', alpha=0.5))
        ims.append([im, t])
    ani = animation.ArtistAnimation(fig, ims, interval=speed, blit=True,
                                    repeat_delay=repeat_delay)
    plt.show(ani)


def tensor2chunks(tensor, new_shape):
    """
    N dimensional array split helper function.
    :param tensor: N dimensional numpy array
    :param new_shape: dimensions of chunks cut from array, old shape must be divisible by new shape
    :return: stack of chunks
    """
    new_shape = np.array(new_shape)
    old_shape = np.array(tensor.shape)
    if all(old_shape % new_shape):  # should evaluate to True if all 0s, false otherwise
        raise ValueError('Old shape not evenly divisible by new shape.')

    repeats = (old_shape / new_shape).astype(int)
    tmpshape = np.column_stack([repeats, new_shape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])

    return tensor.reshape(tmpshape).transpose(order).reshape(-1, *new_shape)


def preprocess(stack_root_path, batch_size, chunk_size, channel):
    """x
    1st- takes stacks and splits each image into smaller images with size=sub_img_size
    2nd- general preprocessing
    3rd- saves numpy files for loading by tf dataset api
    :param stack_root_path: exactly what it sounds like
    :param batch_size:
    :param chunk_size: size of sub images
    :param channel: string of channel you want to use
    :return: training labels
    """
    n_stacks = len(get_stack_paths(stack_root_path))
    n_batches = int(n_stacks/batch_size)

    # main loop for loading stacks and processing
    labels = []
    for i in range(n_batches):
        print('Working on stacks: ', i*batch_size, 'through', (i+1)*batch_size)
        stacks = load_stacks_as_images(stack_root_path, stack_ixs=[i*batch_size, (i+1)*batch_size], channel=channel)
        for j in range(stacks.shape[0]):  # goes through each stack and finds the in-focus img, must do on whole imgs
            foci = [np.mean(stacks[j, :, :, k] > np.mean(stacks[j, :, :, k])) for k in range(stacks.shape[-1])]
            labels.append(foci.index(min(foci)))  # label is the index of the minimum
        new_shape = stacks.shape[0], chunk_size, chunk_size, stacks.shape[-1]
        chunks = tensor2chunks(stacks, new_shape)
        chunks = [chunks[:, i, :, :, :] for i in range(chunks.shape[1])]  # reshape to switch 0th and 1st indices
        chunks = np.array(chunks)

        np.save('np_stack_saves/'+str(i), chunks)
    return np.array(labels, dtype=np.int)


def load_stacks_from_stacks(ixs, root_path='np_stack_saves/'):
    """
    For loading np_stack_saves.
    :param root_path: np_stack_saves
    :param ixs: the indexes of the stacks you want to load (e.g. [0,2] loads 0.npy, 1.npy)
    :return: returns the stacks
    """
    return np.array([np.load(root_path+str(i)+'.npy') for i in range(ixs[0], ixs[1])])


def wrap_bytes(value):
    """
    Wraps raw bytes so they can be saved to the TFRecords file.
    :param value:
    :return:
    """
    return tf.train.Feature(int64_list=tf.train.BytesList(value=[value]))


def wrap_int64(value):
    """
    Wraps ints so they can be saved to the TFRecords file.
    :param value: integer
    :return:


    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def print_progress(count, total):
    """
    Helper function for printing conversion progress
    :param count:
    :param total:
    :return:
    """
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def convert(image_paths, labels, out_path):
    # Args:
    # image_paths   List of file-paths for the images.
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.

    print("Converting: " + out_path)

    # Number of images. Used when printing the progress.
    num_images = len(image_paths)

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        # Iterate over all the image-paths and class-labels.
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            # Print the percentage-progress.
            print_progress(count=i, total=num_images - 1)

            # Load the image-file using matplotlib's imread function.
            img = plt.imread(path)

            # Convert the image to raw bytes.
            img_bytes = img.tostring()

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'image': wrap_bytes(img_bytes),
                    'label': wrap_int64(label)
                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)


def parse(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = \
        {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['image']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.uint8)

    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float32)

    # Get the label associated with the image.
    label = parsed_example['label']

    # The image and label are now correct TensorFlow types.
    return image, label


