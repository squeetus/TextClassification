from flask import Flask
from flask import request
import pandas as pd
import joblib
import json
from flask import jsonify

app = Flask(__name__)

categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

@app.route('/predict', methods=['POST'])
def predict():
     text = request.json['text']

     classifier = joblib.load('newsGroupClassifier.pkl')
     # prediction = classifier.predict(query)
     # return jsonify({'prediction': list(prediction)})

     probabilityByCategory = classifier.predict_proba([text])[0].tolist()
     print(probabilityByCategory)
     # return json.dumps({"probabilities", probabilityByCategory})
     return jsonify({
         'probabilities': probabilityByCategory,
         'probabilityForCategory': "{0:.0%}".format(max(probabilityByCategory)),
         'category': categories[probabilityByCategory.index(max(probabilityByCategory))]
     })

if __name__ == '__main__':
     app.run(port=8080)
#
#   Custom test
#
#
# myTestCase = twenty_test.data[1020]
# print(myTestCase)
# computedCategory = gs_clf.predict([myTestCase])[0]
# print(twenty_train.target_names[computedCategory])
# probabilityByCategory = gs_clf.predict_proba([myTestCase])
# print(probabilityByCategory[0][computedCategory])
# print("{0:.0%}".format(probabilityByCategory[0][computedCategory]))
