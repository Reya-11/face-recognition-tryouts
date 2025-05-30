import tensorflow as tf
import numpy as np

facenet_model = tf.Graph()
with facenet_model.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile("assets/20180402-114759.pb", "rb") as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

sess = tf.compat.v1.Session(graph=facenet_model)

def get_embedding(image):
    image = image.astype(np.float32)
    image = (image - 127.5) / 128.0
    input_tensor = facenet_model.get_tensor_by_name("input:0")
    embedding_tensor = facenet_model.get_tensor_by_name("embeddings:0")
    phase_tensor = facenet_model.get_tensor_by_name("phase_train:0")
    return sess.run(embedding_tensor, feed_dict={
        input_tensor: [image],
        phase_tensor: False
    })[0]
