def dense_block(x, blocks, growth_rate):
    for _ in range(blocks):
        x = conv_block(x, growth_rate)
    return x

def transition_block(x, reduction):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(tf.keras.backend.int_shape(x)[3] * reduction), (1, 1), padding='same')(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

def conv_block(x, growth_rate):
    x1 = BatchNormalization()(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(4 * growth_rate, (1, 1), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(growth_rate, (3, 3), padding='same')(x1)
    x = Concatenate(axis=-1)([x, x1])
    return x

def densenetmini(input_shape=None, classes=38):
    input_shape = (150, 150, 3)
    img_input = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = dense_block(x, 6, 32)
    x = transition_block(x, 0.5)
    x = dense_block(x, 12, 32)
    x = transition_block(x, 0.5)
    x = dense_block(x, 16, 32)
    x = transition_block(x, 0.5)
    x = dense_block(x, 8, 32)

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(img_input, x, name='densenet121')
    return model

