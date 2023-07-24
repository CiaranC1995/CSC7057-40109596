from flask import render_template, Blueprint, session, request, redirect, url_for
import requests

contact_blueprint = Blueprint('contact', __name__)

@contact_blueprint.route('/contact', methods=['GET', 'POST'])
def contact_route():
    loginStatus = 'authen' in session
    loginMessage = f"Logged In as '{session.get('user', ['', ''])[1]}'" if loginStatus else ""

    if request.method == 'POST':
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

        return redirect(url_for('contact.contact_message'))

    return render_template('contact.html', loginStatus=loginStatus, loginMessage=loginMessage)

@contact_blueprint.route('/contactMessage', methods=['GET'])
def contact_message():
    isContactMessageCreated = True
    loginStatus = 'authen' in session
    loginMessage = f"Logged In as '{session['user'][1]}'" if loginStatus else ""

    return render_template('contactMessage.html', loginStatus=loginStatus, isContactMessageCreated=isContactMessageCreated, loginMessage=loginMessage)
