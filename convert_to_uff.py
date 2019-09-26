import uff

filename="./converted/cmu/cmu_epoch477_map44.pb"
uff.from_tensorflow_frozen_model(frozen_file=filename, output_filename="converted_text.uff", text=True)
