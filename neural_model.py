
import tensorflow as tf

def train_model(epochs):
    batch_size = 32
    img_height = 180
    img_width = 180
    # Directory points to training images.
    directory = "C:\\Users\\Foste\\Desktop\\CV_Final\\imageSet"

    print(f"Directory path is: {directory}")

    # Sets up dataset for training.
    train_ds = tf.keras.utils.image_dataset_from_directory(  # generates a dataset from a directory and its sub directories.
        directory,
        labels='inferred',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True)

    # Creates validation dataset.
    print(type(train_ds))
    validation_ds = tf.keras.utils.image_dataset_from_directory(  # validation images
        directory,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True).repeat()

    # Verify files have the correct properties.
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break


    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Trains and validates model.
    # This network is technically a sequential model with convolutional layers.
    # Declares how many subdirectories live within the main directory,and thus how many classes will be declared.
    
    num_classes = 5 

    model_seq = tf.keras.Sequential([tf.keras.layers.Rescaling(1. / 255),
                                     tf.keras.layers.Conv2D(32, 3, activation='relu'),
                                     tf.keras.layers.MaxPooling2D(),
                                     tf.keras.layers.Conv2D(32, 3, activation='relu'),
                                     tf.keras.layers.MaxPooling2D(),
                                     tf.keras.layers.Conv2D(32, 3, activation='relu'),
                                     tf.keras.layers.MaxPooling2D(),
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(128, activation='relu'),
                                     tf.keras.layers.Dense(num_classes)
                                     ])


    #Compiles the model with every run and saves it to a checkpoint location.
    model_seq.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    checkpoint_path = "C:\\Users\\Foste\\Desktop\\CV_Final\\check\\cp.cpkt" 
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # Declares the training steps per epoch and the number of validation steps after each epoch is completed.
    model_seq.fit(
        train_ds,
        steps_per_epoch= 64,
        validation_data=validation_ds,
        validation_steps=30,
        epochs=epochs,
        callbacks=[cp_callback]  # pass callback to training
    )

    
    # model saved as an h5 for use in catagorizer.
    final_model = "C:\\Users\\Foste\\Desktop\\CV_Final\\final_model.h5" 
    model_seq.save(final_model)