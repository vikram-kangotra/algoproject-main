from keras import backend as K
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import sequence_extractionn
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Server is Listening"

model = keras.models.load_model("model1-05-0.7873.hdf5")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = request.get_json()
    protein_sequence = data.get('protein_sequence')
    threshold = data.get('threshold')
    threshold = float(threshold) if threshold else 0
    enable_threshold = data.get('enableThreshold')
    if not enable_threshold:
        threshold=0
    lines = protein_sequence.split('\n')
    sequence_id = lines[0][0:]
    sequence_data = lines[1:]  # Rest of the lines are sequence data
    prediction_results = []

    for line in sequence_data:
        da = line.strip()  # Remove any leading/trailing whitespace

        if "R" not in da:
            continue  # Skip lines without "R"

        r_pos = [i for i, char in enumerate(da) if char == "R"]

        for pos in r_pos:
            temp = dict()
            temp['SeqId'] = sequence_id
            extr_sequence_list = []  # Reset the list for each new iteration
            temp["R site"] = pos+1
            extr_sequence = sequence_extractionn.seq_extr(da, pos, len(da))
            temp["Peptide"] = extr_sequence
            # Convert the extracted sequence to one-hot encoding and make predictions
            encoder = pickle.load(open("one_hot_encoder.pkl", 'rb'))
            extr_sequence_list = list(extr_sequence)
            desired_length = 51
            if len(extr_sequence_list) < desired_length:
                extr_sequence_list = extr_sequence_list + \
                    ['X'] * (desired_length - len(extr_sequence_list))
            elif len(extr_sequence_list) > desired_length:
                extr_sequence_list = extr_sequence_list[:desired_length]
            reshaped_sequence = [[char] for char in extr_sequence_list]
            embedded_data = encoder.transform(reshaped_sequence).toarray()
            embedded_data = np.reshape(embedded_data, (-1, 51, 21, 1))
            result = model.predict([embedded_data, embedded_data]).tolist()
            temp["Prediction Score"] = result
            print(temp)
            # Check if the enable_threshold flag is set and the prediction score is greater than or equal to the threshold
            if result[0][0] >= threshold:
                prediction_results.append(temp)

            # Clear the Keras session to release resources
            K.clear_session()
    return jsonify(prediction_results)


if __name__ == '__main__':
    app.run(debug=True)
