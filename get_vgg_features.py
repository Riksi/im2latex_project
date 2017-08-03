from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import os
import matplotlib.pyplot as plt

train_data_dir = 'eq_images/cropped/images'

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(img_width, img_height, 3)
)

model = Model(inputs=base_model.input, 
    outputs=base_model.get_layer('block4_conv1').output)

filenames = set(os.listdir(train_data_dir))
for fn in filenames:
    im = np.expand_dims(plt.imread(os.path.join(train_data_dir,fn))[:,:,:3], axis=0)
    imbed = model.predict(im)[0]
    name = fn.split('.')[0]
    np.save(arr = imbed, file='imbeddings3/'+name)


