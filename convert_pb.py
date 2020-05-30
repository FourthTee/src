import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

"""
Convert the frozen graph in the .pb file to savedmodel that is usable by docker to run tensorflow serving
"""

def read_pb_return_tensors(graph, pb_file, return_elements):
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements


return_elements = ['input/input_data:0',
                       'pred_sbbox/concat_2:0',
                       'pred_mbbox/concat_2:0',
                       'pred_lbbox/concat_2:0']
pb_file = './detector_frozen.pb'
graph = tf.Graph()
return_tensors = read_pb_return_tensors(graph, pb_file, return_elements)
sigs = {}
export_dir = './saved'

#Create builder to save the model at export_dir
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)


#Open tf session to get tensors
with tf.Session(graph=graph) as sess:
  inp = return_tensors[0]
  out = [return_tensors[1], return_tensors[2], return_tensors[3]]

  #Applied the signatures
  sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
      tf.saved_model.signature_def_utils.predict_signature_def(
          {"in": inp}, {"sbbox": out[0], "mbbox": out[1], "lbbox": out[2]})

  builder.add_meta_graph_and_variables(sess,
                                       [tag_constants.SERVING],
                                       signature_def_map=sigs)

builder.save()