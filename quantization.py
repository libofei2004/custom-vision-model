import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("/saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_quant_model = converter.convert()

print(len(tflite_quant_model))
with open("tflite_quant_model.tflite", "wb") as f:
    f.write(tflite_quant_model)
