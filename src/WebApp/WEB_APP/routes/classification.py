import datetime
from flask import render_template, Blueprint, session, request
import requests
from classifierFiles.classifier import TextClassifier

classification_blueprint = Blueprint('classification', __name__)

@classification_blueprint.route('/classification', methods=["POST"])
def classification_route_post():

    loginStatus = 'authen' in session
    user_id = session.get('user_id', -1)
    loginMessage = f"Logged In as '{session.get('user', ['', ''])[1]}'" if loginStatus else ""

    input_text = request.form.get('userInputText')

    classifier = TextClassifier(r'src\ModelTraining\Models\LinearSVC_Classifier_main.pickle', r'src\ModelTraining\Vectorizers\tfidfvectorizer.pickle')
    classification_result = classifier.classify_text(input_text=input_text)

    if 'error_message' in classification_result:
        return render_template('error.html', error_message=classification_result['error_message'], loginMessage=loginMessage, loginStatus=loginStatus)

    date_of_classification = datetime.date.today().strftime("%d/%m/%Y")
    processed_input_text = classification_result['processed_input_text']
    text_perplexity = classification_result['text_perplexity']
    text_burstiness = classification_result['text_burstiness']
    classifier_output = classification_result['classifier_output'].item()
    human_probability = classification_result['human_probability']
    ai_probability = classification_result['ai_probability']

    # Join two lists for ease of processing
    sentences_perplexities = list(zip(classification_result['sentences'], classification_result['sentence_perplexities']))

    endPoint = 'http://127.0.0.1:8080/classificationRoute'

    payload = {
        'date_of_classification': date_of_classification,
        'raw_input_text': input_text,
        'preprocessed_input_text': processed_input_text,
        'text_perplexity': text_perplexity,
        'text_burstiness': text_burstiness,
        'classifier_output': classifier_output,
        'human_probability': human_probability,
        'ai_probability': ai_probability,
        'user_id': user_id
    }

    try:
        classifier_output_post = requests.post(endPoint, json=payload)
        classifier_output_id = classifier_output_post.json().get('classifier_output_id')

        endpoint2 = 'http://127.0.0.1:8080/sentencePerplexityRoute'

        for index, (sentence, perplexity) in enumerate(sentences_perplexities):
            requests.post(endpoint2, json={
                "sentence_number": index + 1,
                "sentence": sentence,
                "perplexity": perplexity,
                "classifier_output_id": classifier_output_id
            })

        endpoint1 = 'http://127.0.0.1:8080/specificUser'
        api_response = requests.get(endpoint1, json={'user_id': user_id})
        user_info = api_response.json()
        
        return render_template('classificationResult.html', classification_result=classification_result, user_info=user_info, sentences_perplexities=sentences_perplexities, loginMessage=loginMessage, loginStatus=loginStatus)

    except requests.exceptions.RequestException as e:
        return f"Error occurred: {e}"
