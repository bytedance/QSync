
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice,
    AutoConfig
)

# Number of parameters
def count_para(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params

def create_transformer_config(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)

'''
    Create models
'''
roberta_name = "roberta-base"
bert_name = "bert-base-uncased"
config_roberta = create_transformer_config(roberta_name)
config_bert = create_transformer_config(bert_name)

roberta = AutoModelForMultipleChoice.from_pretrained(
            roberta_name,
            from_tf=bool(".ckpt" in roberta_name),
            config=config_roberta,
)



bert = AutoModelForQuestionAnswering.from_pretrained(
            bert_name,
            from_tf=bool(".ckpt" in bert_name),
            config=config_bert,
)

from torch_ort import DebugOptions
import torch_ort 
model = torch_ort.ORTModule(bert, DebugOptions(save_onnx=True, onnx_prefix='roberta'))
# print(model)


# import tensorflow as tf 

# model = tf.keras.applications.resnet50.ResNet50(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
# )

# import pdb; pdb.set_trace()
# model.GraphDef


# import os

# import tensorflow as tf
# from tensorflow import keras

# print(tf.version.VERSION)

# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# train_labels = train_labels[:1000]
# test_labels = test_labels[:1000]

# train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
# test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# # Define a simple sequential model
# def create_model():
#   model = tf.keras.Sequential([
#     keras.layers.Dense(512, activation='relu', input_shape=(784,)),
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(10)
#   ])

#   model.compile(optimizer='adam',
#                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

#   return model

# # Create a basic model instance
# model = create_model()

# # Display the model's architecture
# model.summary()

# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

# # Train the model with the new callback
# model.fit(train_images, 
#           train_labels,  
#           epochs=10,
#           validation_data=(test_images, test_labels),
#           callbacks=[cp_callback])  # Pass callback to training

# # This may generate warnings related to saving the state of the optimizer.
# # These warnings (and similar warnings throughout this notebook)
# # are in place to discourage outdated usage, and can be ignored.

# # Create a basic model instance
# model = create_model()

# # Evaluate the model
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# # Loads the weights
# model.load_weights(checkpoint_path)

# # Re-evaluate the model
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# # Include the epoch in the file name (uses `str.format`)
# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# batch_size = 32

# # Create a callback that saves the model's weights every 5 epochs
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path, 
#     verbose=1, 
#     save_weights_only=True,
#     save_freq=5*batch_size)

# # Create a new model instance
# model = create_model()

# # Save the weights using the `checkpoint_path` format
# model.save_weights(checkpoint_path.format(epoch=0))

# # Train the model with the new callback
# model.fit(train_images, 
#           train_labels,
#           epochs=50, 
#           batch_size=batch_size, 
#           callbacks=[cp_callback],
#           validation_data=(test_images, test_labels),
#           verbose=0)

# latest = tf.train.latest_checkpoint(checkpoint_dir)
# latest