from flask import Flask, render_template, request
import helper
import joblib
import os

# load model
model = joblib.load(open("l1_l2_classification.pkl", "rb"))
model_l1_l2 = joblib.load(open("model_l1_l2.pkl", "rb"))
model_l3 = joblib.load(open("model_l3.pkl", "rb"))

enc = {
       'L1/L2': 0,
       'L3': 1
       }

# list out keys and values separately 
key = list(enc.keys()) 
val = list(enc.values()) 

encoding_l3 = {
 'grp_10': 0,
 'grp_12': 1,
 'grp_13': 2,
 'grp_14': 3,
 'grp_16': 4,
 'grp_18': 5,
 'grp_19': 6,
 'grp_2': 7,
 'grp_25': 8,
 'grp_29': 9,
 'grp_3': 10,
 'grp_4': 11,
 'grp_6': 12,
 'grp_9': 13
 }


# list out keys and values separately 
key_list = list(encoding_l3.keys()) 
val_list = list(encoding_l3.values()) 

encoding_l2 = {'grp_0': 0, 'grp_8': 1}
key_list2 = list(encoding_l2.keys()) 
val_list2 = list(encoding_l2.values()) 

# app
app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')

# routes


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        message = request.form['message']  # Clean and preprocess
        message = helper.fn_translate(message)
        message = helper.clean_text(message)
        message = helper.pre_process(message)
        
        data = [message]
        
        
# predictions
        result = model.predict(data)
        
        # send back to browser
        output = key[val.index(result)]
        
        if output == 'L1/L2':
            group = model_l1_l2.predict(data)

        # send back to browser
            output_final = key_list2[val_list2.index(group)]
        else:
            group = model_l3.predict(data)

        # send back to browser
            output_final = key_list[val_list.index(group)]

    # return data
        return render_template('result.html', prediction=output_final, team=output)

if __name__ == "__main__":
    app.run(debug=True)
