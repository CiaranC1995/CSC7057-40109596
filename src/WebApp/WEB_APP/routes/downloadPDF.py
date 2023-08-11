from flask import render_template, Blueprint, session, request, redirect, url_for, Response
import requests, pdfkit

downloadPDF_blueprint = Blueprint('downloadPDF', __name__)

@downloadPDF_blueprint.route('/downloadPDF', methods=['POST'])
def downloadPDF_route():

    loginStatus = 'authen' in session
    loginMessage = f"Logged In as '{session.get('user', ['', ''])[1]}'" if loginStatus else ""

    if loginStatus:
    
        classifier_output_id = request.form.get('classifier_output_id')
        endpoint = f'http://127.0.0.1:8080/specificHistoryRoute'
        config = pdfkit.configuration(wkhtmltopdf=r'C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')
            
        try:
            api_response = requests.get(endpoint, params={'classifier_output_id': classifier_output_id})
            specificHistoryInfo = api_response.json() 
        except requests.exceptions.RequestException as e:
            return f"Error occurred: {e}"
        
        pdf_options = {
            'page-size': 'A4',  # Change this to the desired page size (e.g., A4)
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'header-right': 'Header Text',
            'footer-center': 'Page [page] of [topage]'
        }


        if request.method == 'POST':
            pdfkit.from_string(render_template('specificHistory.html', specificHistoryInfo=specificHistoryInfo), 'classifierOutput.pdf', configuration=config, options={"enable-local-file-access": ""})
            with open('classifierOutput.pdf', 'rb') as pdf_file:
                pdf_data = pdf_file.read()
            response = Response(pdf_data, content_type='application/pdf')
            response.headers['Content-Disposition'] = 'attachment; filename=classifierOutput.pdf'

            return response
        
        return render_template('downloadMessage.html', loginStatus=loginStatus, loginMessage=loginMessage, specificHistoryInfo=specificHistoryInfo)
    
    else:
        return render_template('landing.html', loginStatus=loginStatus, loginMessage=loginMessage)