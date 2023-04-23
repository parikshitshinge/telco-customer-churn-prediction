import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from scipy.sparse import hstack

app = Flask(__name__)
file = open('rf.pkl', 'rb')
vectorizers = []
for i in range(16):
    vectorizers.append(pickle.load(file))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    query_point = [x for x in request.form.values()]
    # ['area_code_415', 'yes', 'yes', '40', '242', '113', '54', '59', '69', '7', '241', '127', '7', '6', '8', '4', '6']
    query_point_vectorized = []
    intl_plan = query_point[1]
    vmail_plan = query_point[2]
    del query_point[1:3]
    for i, data in enumerate(query_point):
        vectorizer = vectorizers[i]
        if i in [0]:
            temp = []
            temp.insert(0, data) 
            transformed_data = vectorizer.transform(temp).toarray().ravel()
            query_point_vectorized.extend(transformed_data)
        else:    
            data = int(data)
            transformed_data = vectorizer.transform(np.array(data).reshape(-1,1)).ravel()
            query_point_vectorized.extend(transformed_data)

    query_point_vectorized.insert(3, int(intl_plan))
    query_point_vectorized.insert(4, int(vmail_plan))
    
    gbdt = vectorizers[15]
    y_hat = gbdt.predict(np.array(query_point_vectorized).reshape(1,-1))[0]
    if y_hat:
        result = "Likely to churn"
    else:
        result = "Not likely to churn"

    return render_template('index.html', prediction_result = result )

if __name__ == "__main__":
    app.run(debug=True)