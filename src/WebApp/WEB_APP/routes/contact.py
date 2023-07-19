from flask import render_template, Blueprint, session, request
import requests

contact_blueprint = Blueprint('contact', __name__)

@contact_blueprint.route('/contact', methods=['GET'])
def contact_route():
    if 'authen' in session:
        loginMessage = f"Logged In as '{session['user'][1]}'"
        loginStatus = True
        return render_template('contact.html', loginStatus=loginStatus, loginMessage=loginMessage)
    else:
        loginStatus = False
        return render_template('contact.html', loginStatus=loginStatus)
    
@contact_blueprint.route('/contact', methods=['POST'])
def contact_route_post():
    if 'authen' in session:
        loginMessage = f"Logged In as '{session['user'][1]}'"
        loginStatus = True 
    else:
        loginStatus = False
    
    isContactMessageCreated = True

    name = request.form.get('name')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message = request.form.get('message')

    endPoint = 'http://127.0.0.1:8080/contactRoute'

    requests.post(endPoint, json={
        'name': name,
        'email': email,
        'subject': subject,
        'message': message
    })

    return render_template('contactMessage.html', loginStatus=loginStatus, isContactMessageCreated=isContactMessageCreated, loginMessage=loginMessage)