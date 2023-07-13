from flask import render_template, Blueprint, session, request
import requests

from classifierFiles.classifier import TextClassifier

classification_blueprint = Blueprint('classification', __name__)

@classification_blueprint.route('/classification', methods=["POST"])
def classification_route_post():

    input_text = request.form.get('userInputText')

    classifier = TextClassifier(r'C:/Users/ccase/Desktop/CSC7057-40109596/src/ModelTraining/Models/LinearSVC_Classifier_EntireDataset_detokenized_NoTransformationOfPPL_Burst.pickle', r'C:/Users/ccase/Desktop/CSC7057-40109596/src/ModelTraining/Vectorizers/tfidfvectorizer.pickle')
    classification_result = classifier.classify_text(input_text=input_text)

    processed_input_text = classification_result['processed_input_text']
    text_perplexity = classification_result['text_perplexity']
    text_burstiness = classification_result['text_burstiness']
    classifier_output = classification_result['classifier_output'].item()
    human_probability = classification_result['human_probability']
    ai_probability = classification_result['ai_probability']

    if 'authen' in session:
        user_id = session['user_id']
    else:
        user_id = -1

    endPoint = 'http://127.0.0.1:8080/classificationRoute'

    requests.post(endPoint, json={
        "raw_input_text": input_text,
        "preprocessed_input_text": processed_input_text,
        "text_perplexity": text_perplexity,
        "text_burstiness": text_burstiness,
        "classifier_output": classifier_output,
        "human_probability": human_probability,
        "ai_probability": ai_probability,
        "user_id": user_id
    })

    # Maybe do another query here to return all users information and pass it to the template
    # Need a new API route developed 
    # endpoint1 = ''

    # try:
    #     user_info = requests.get(endpoint1).json

    # except Exception as e:
    #     return f"Error occurred: {str(e)}"
    
    return render_template('classificationResult.html', classification_result=classification_result)