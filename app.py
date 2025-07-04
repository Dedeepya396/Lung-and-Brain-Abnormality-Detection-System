from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os
from werkzeug.utils import secure_filename
import runmodel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = '88dfd5ee5d65810d28dc2468b11de5737aefae22db792ec5' 

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

USER_CREDENTIALS = {'username': 'admin', 'password': 'password123'}

@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == USER_CREDENTIALS['username'] and password == USER_CREDENTIALS['password']:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            session['username'] = username
            return redirect(url_for('index'))
        

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'username' not in session:
        return redirect(url_for('login'))

    ct_scan_file = request.files.get('ct_scan')
    xray_file = request.files.get('xray')

    if not ct_scan_file and not xray_file:
        return redirect(request.url)

    if ct_scan_file:
        if ct_scan_file.filename == '':
            return redirect(request.url)
        filename = secure_filename(ct_scan_file.filename)
        ct_scan_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        ct_scan_file.save(ct_scan_filepath)
        model_path = './model/model.pth'
        result = runmodel.GenerateOutput(model_path, ct_scan_filepath)
        return render_template('result.html', result=result, filename=filename, file_type="CT Scan")

    if xray_file:
        if xray_file.filename == '':
            return redirect(request.url)

        filename = secure_filename(xray_file.filename)
        xray_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        xray_file.save(xray_filepath)
        model_path = './model/model.pth'
        result = runmodel.GenerateOutput(model_path, xray_filepath)
        return render_template('result.html', result=result, filename=filename, file_type="X-Ray")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/retrain')
@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    if request.method == 'POST':
        retrain_info = request.form['retrain_info']
        retrain_result = "Model retrained with input: " + retrain_info
        return render_template('retrain.html', result=retrain_result)

    return render_template('retrain.html', result=None)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
