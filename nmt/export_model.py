import tensorflow as tf
import argparse
import yaml
import os
import shutil
from tensorflow.python.saved_model import signature_constants
from nmt.nmtservice.export_transformer import ExportTransformer, MODEL_FOLDER_NAME


"""
Usage:
python export_model.py \
    --input-checkpoint ${CHECK_POINT_PATH} \
    --version-name ${VERSION_NAME}
"""


parser = argparse.ArgumentParser()
parser.add_argument("--input-checkpoint", "-i", required=True)
parser.add_argument("--version-name", "-v", required=True)
args = parser.parse_args()

check_point_dir = os.path.dirname(args.input_checkpoint)

assert os.path.exists(check_point_dir)
assert os.path.exists(f"{check_point_dir}/tokenizer"), f"'tokenizer' folder must be in {args.input_checkpoint} path"

if not os.path.exists(args.version_name):
    os.mkdir(args.version_name)

shutil.copytree(f"{check_point_dir}/tokenizer", f"{args.version_name}/tokenizer")

hyp_args = yaml.load(open("train_config.yaml"))
## Build model
model = ExportTransformer(hyp_args, args.input_checkpoint)

inputs_candidate = {
    "inputs": model.input_placeholder
}
outputs_candidate = {
    "outputs": model.decoded_idx
}

var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

print("START Exporting model")

with model.sess as sess:
    # SavedModel builder
    builder = tf.saved_model.builder.SavedModelBuilder(f"{args.version_name}/{MODEL_FOLDER_NAME}")
    
    builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    tf.saved_model.signature_def_utils.predict_signature_def(
                        inputs=inputs_candidate,
                        outputs=outputs_candidate
                    )
            }
        )
    
    builder.save()

print("END Exporting model")