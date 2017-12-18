from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout
from keras import optimizers
import tensorflow as tf
from keras.utils import multi_gpu_model
from sklearn.metrics import confusion_matrix
import urllib3
import cv2
import smart_open


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    # rows,cols,ch = img.shape
    rows, cols, ch = 224, 224, 3
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness


    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img


WEIGHTS_PATH_NO_TOP = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG16(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG16 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = WEIGHTS_PATH_NO_TOP
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


df = pd.read_excel('sample2.xlsx')
df = df[pd.notnull(df['HAZARDMAPRATING'])]
df = df[df['INSPECTIONTYPE'].str.contains('EL')]


def x_y_train_data_gen(batch_size):
    while 1:
        batch_features = np.zeros((batch_size, 224, 224, 3))
        batch_labels = np.zeros((batch_size, 1))
        b = 0
        for i, j in zip(df['HAZARDMAPRATING'].values, df['PHOTOLINK'].values):
            ls = []
            file = smart_open.smart_open('s3://datasetforimagetraining/{}'.format(j))
            for line in file:
                ls.append(float(line.decode("utf-8").split('\r\n')[0]))
            cv2.imwrite("image.png", np.array(ls).reshape((224, 224, 3), order='F'))
            img = cv2.imread('image.png')
            for _ in range(16):
                X = []
                Y = []
                img_transformed = transform_image(img, 20, 10, 5, brightness=1)
                # with open("transformed_image.png", "wb") as code:
                #     code.write(img_transformed)
                # img = image.load_img('transformed_image.png', target_size=(224, 224))
                x = image.img_to_array(img_transformed)
                X.append(x)
                X = np.array(X)
                X = X.astype('float32')
                X /= 255
                if i in [0.0, 1.0, 2.0]:
                    y = 0.0
                    Y.append(y)
                else:
                    y = 1.0
                    Y.append(y)
                if b < batch_size:
                    batch_features[b] = X
                    batch_labels[b] = Y
                    b += 1
                else:
                    b = 0
                    batch_features = np.zeros((batch_size, 224, 224, 3))
                    batch_labels = np.zeros((batch_size, 1))
                    # yield X, np.array(Y)
                    yield (batch_features, batch_labels)




num_classes = 2  # Hard coded now will change

# Get back the convolutional part of a VGG network trained on ImageNet
img_rows, img_cols, img_channel = 224, 224, 3
model_vgg16_conv = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))

print(len(model_vgg16_conv.layers))
ls_trainable_layers = ['False']*19 + ['True']*0
# Make vgg16 model layers as non trainable
for i, layer in enumerate(model_vgg16_conv.layers):
    layer.trainable = ls_trainable_layers[i]

add_model = Sequential()
add_model.add(Flatten(input_shape=model_vgg16_conv.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(1, activation='sigmoid'))

model = Model(inputs=model_vgg16_conv.input, outputs=add_model(model_vgg16_conv.output))
# model = multi_gpu_model(model, gpus=2)
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
##model.fit(X_train, Y[0:6], epochs=10, batch_size=1)
model.fit_generator(generator=x_y_train_data_gen(batch_size=16), samples_per_epoch=5000, nb_epoch=10, verbose=1)


def x_y_test_data_gen():
    while 1:
        for i, j in zip(df['HAZARDMAPRATING'].values[15:], df['PHOTOLINK'].values[15:]):
            X = []
            Y = []
            ls = []
            file = smart_open.smart_open('s3://datasetforimagetraining/{}'.format(j))
            for line in file:
                ls.append(float(line.decode("utf-8").split('\r\n')[0]))
            cv2.imwrite("image.png", np.array(ls).reshape((224, 224, 3), order='F'))
            new_img = image.load_img('image.png', target_size=(224, 224))
            x = image.img_to_array(new_img)
            X.append(x)
            X = np.array(X)
            X = X.astype('float32')
            X /= 255
            if i in [0.0, 1.0, 2.0]:
                y = 0.0
                Y.append(y)
            else:
                y = 1.0
                Y.append(y)
            yield X, np.array(Y)


def decision(a):
    if a[0] > 0.5:
        return 1.0
    else:
        return 0.0


test_data_generator = x_y_test_data_gen()
recall = list()
precision = list()
f1_score = list()
c = 0
y_true = []
y_pred = []
for i in test_data_generator:
    if c < 20:
        x_te, y_te = i
        # print(x_te)
        # print(y_te)
        prediction = model.predict(x_te)
        print(prediction[0][0])
        print(y_te[0])
        y_true.append(y_te[0])
        print(decision(prediction[0]))
        y_pred.append(decision(prediction[0]))
        c += 1
    else:
        break

print(y_pred)
print(y_true)
print(confusion_matrix(y_true, y_pred))
