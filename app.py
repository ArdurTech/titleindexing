import datetime
import os
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
import shutil
from werkzeug.utils import secure_filename
from flask import Flask, current_app, flash, render_template, request, jsonify, redirect, session, url_for, send_file
import cv2
import numpy as np
import pymysql
import pytesseract
from pdf2image import convert_from_path
import base64
import tempfile
import re
from functools import wraps
import bcrypt
from database import create_database_and_table, get_db_connection, insert_data, get_user_by_username, create_user
import database

app = Flask(__name__)
app.secret_key = 'secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static/uploads/processed_files')
# app.permanent_session_lifetime = datetime.timedelta(minutes=10)
# Define a folder for saving processed files
UPLOAD_FOLDER = 'static/uploads/processed_files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# MySQL configurations
app.config['MYSQL_HOST'] = 'ardurtech.mysql.database.azure.com'
app.config['MYSQL_USER'] = 'aditya'
app.config['MYSQL_PASSWORD'] = 'Admin3110'
app.config['MYSQL_DB'] = 'ocr_database'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.config['MYSQL_SSL_CA'] = 'cert/DigiCertGlobalRootCA.crt.pem'

# Initialize the database and table
with app.app_context():
    create_database_and_table()
    
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))  # Redirect to login if not logged in
        return f(*args, **kwargs)
    return decorated_function

def role_required(allowed_roles):
    def wrapper(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'logged_in' not in session:
                return redirect(url_for('login'))  # Redirect to login if not logged in
            user_role = session.get('role')  # Get user role from session
            if user_role not in allowed_roles:
                flash('You do not have access to this page.')
                return redirect(url_for('dashboard'))  # Redirect to dashboard if unauthorized
            return f(*args, **kwargs)
        return decorated_function
    return wrapper

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    inverted = cv2.bitwise_not(denoised)
    white_background = np.full_like(image, 255)
    result = cv2.bitwise_and(white_background, white_background, mask=inverted)
    return result

def extract_text_from_image(image):
    preprocessed_image = preprocess_image(image)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
    return clean_text(text)

def clean_text(text):
    # Remove specific unwanted patterns or words
    text = re.sub(r'Vv', 'V', text)
    text = re.sub(r'Bewember', 'December', text)
    text = re.sub(r'DAMPABAS', 'Lampasas', text)
    text = re.sub(r'Sohn', 'John', text)
    text = re.sub(r'YThat', 'That', text)
    text = re.sub(r'|', '', text)
    text = re.sub(r'__', '', text)
    text = re.sub(r'_', '', text)# Remove double underscores
    text = re.sub(r'eeeny', '', text)
    text = re.sub(r'r2csccr', '', text)
    text = re.sub(r'ooo', '', text)
    text = re.sub(r'acco.', '', text)
    text = re.sub(r'ccccccccscsceessseets', '', text)
    text = re.sub(r'wee', '', text)
    text = re.sub(r'eee', '', text)
    text = re.sub(r'Btock', 'Block', text) 
    text = re.sub(r'[^\w\s.,?!()_~]', '', text)
    lines = text.splitlines()
    spaced_text = '\n\n'.join(line.strip() for line in lines if line.strip())
    spaced_text = re.sub(r'(\d{1,2}/\d{1,2}/\d{4})', r'\1\n', spaced_text)
    paragraphs = re.split(r'\n\s*\n', spaced_text)
    formatted_text = '\n\n'.join(paragraph.strip() for paragraph in paragraphs if paragraph.strip())
    return formatted_text

def handle_file_upload(file):
    ext = file.filename.split('.')[-1].lower()
    if ext in ['png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp']:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image, extract_text_from_image(image)
    elif ext == 'pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            file.save(temp_pdf.name)
            images = convert_from_path(temp_pdf.name)
            image = np.array(images[0])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV compatibility
            return image, extract_text_from_image(image)
    return None, "Unsupported file format"

def create_user(username, password, role):
    # Check if username already exists
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT username FROM user WHERE username = %s", (username,))
    existing_user = cursor.fetchone()
    cursor.close()
    connection.close()

    if existing_user:
        return False  # User already exists

    # Create hashed password and insert into the database
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("INSERT INTO user (username, password_hash, role) VALUES (%s, %s, %s)", (username, hashed, role))
    connection.commit()
    cursor.close()
    connection.close()
    return True

@app.route('/register', methods=['GET', 'POST'])
def register():
    username_exists = False
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        role = request.form.get('role')  # Capture role from the form

        # Ensure all fields are filled
        if not username or not password or not confirm_password or not role:
            flash('All fields, including role, are required.')
            return render_template('register.html', username_exists=username_exists, username=username)

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match.')
            return render_template('register.html', username_exists=username_exists, username=username)

        # Check if the username already exists
        existing_user = get_user_by_username(username)  # Assuming this function checks for an existing user
        if existing_user:
            username_exists = True
            flash('Username already exists. Please choose a different username.')
            return render_template('register.html', username_exists=username_exists, username=username)

        # If the username is new, create the user with the selected role
        create_user(username, password, role)

        flash('Registration successful! You can now log in.')
        return redirect(url_for('login'))

    return render_template('register.html', username_exists=username_exists, username=request.form.get('username'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = get_user_by_username(username)
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            session['logged_in'] = True
            session['username'] = username  # Store the username in the session
            session['role'] = user['role']  # Store the user's role in the session
            return redirect(url_for('dashboard'))  # Redirect to the main app after successful login
        else:
            return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Pass username to the template if needed
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/check_role_access/<role>', methods=['GET'])
def check_role_access(role):
    # Get the user's role from session
    user_role = session.get('role')
    
    # Logic to check if the user has access based on role hierarchy
    if role == 'dataentry' and user_role not in ['dataentry', 'qc', 'lead']:
        return jsonify({'access': False, 'message': 'Access Denied: You do not have permission to access Data Entry.'})
    elif role == 'qc' and user_role not in ['qc', 'lead']:
        return jsonify({'access': False, 'message': 'Access Denied: You do not have permission to access QC.'})
    elif role == 'lead' and user_role != 'lead':
        return jsonify({'access': False, 'message': 'Access Denied: You do not have permission to access Lead.'})

    # If access is granted
    return jsonify({'access': True})

@app.route('/qc')
@login_required
@role_required(['qc', 'lead'])  # Only QC and Lead roles can access the QC page
def qc():
    username = session.get('username')
    return render_template('qc.html', username=username)

@app.route('/lead')
@login_required
@role_required(['lead'])  # Only Lead role can access the Lead page
def lead():
    username = session.get('username')
    return render_template('lead.html', username=username)

@app.route('/finalreports')
def final_reports():
    if 'username' in session:
        username = session['username']
        connection = database.get_db_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)
        
        # Fetch Final Reports data
        cursor.execute("""
            SELECT username, filename, input1, input2, input3, input4, input5, qc_check, qc_done_by, created_time, qc_submission_time
            FROM ocrdata
            WHERE qc_check = 'Yes'
            ORDER BY created_time DESC
        """)
        reports = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return render_template('finalreports.html', username=username, reports=reports)
    else:
        return redirect(url_for('login'))

@app.route('/')
@role_required(['dataentry', 'qc', 'lead'])  # All roles can access the index page
@login_required
def index():
    username = session.get('username')
    if username is None:
        flash('Please log in first.')
        return redirect(url_for('login'))  # Ensure redirection if username is None
    return render_template('index.html', username=username)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'})
    image, text = handle_file_upload(file)
    if image is None:
        return jsonify({'error': text})
    _, image_encoded = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(image_encoded).decode('utf-8')
    return jsonify({
        'image': image_base64,
        'text': text,
        'filepath': file.filename
    })

@app.route('/submit', methods=['POST'])
@login_required  # Ensure the user is logged in
def submit_data():
    username = session.get('username')  # Retrieve username from session
    inputs = [request.form.get(f'input{i+1}') for i in range(5)]
    extracted_text = request.form.get('extractedText')
    filename = request.form.get('filepath')

    if any(not field for field in inputs + [extracted_text, filename]):
        return jsonify({'success': False, 'error': 'All fields must be filled.'})

    # Create a secure filename
    safe_filename = secure_filename(filename)
    
    # Save the file to the upload folder
    image_data = request.form.get('imageData')  # Assuming the image is sent as base64
    if image_data:
        image_data = image_data.split(',')[1]  # Extract base64 data
        image_bytes = base64.b64decode(image_data)

        # Generate the final save path
        save_path = os.path.join(UPLOAD_FOLDER, safe_filename)

        # Save the file to the designated folder
        with open(save_path, 'wb') as image_file:
            image_file.write(image_bytes)

    # Insert data into the database, including the file path and username
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO ocrdata (username, filename, file_path, input1, input2, input3, input4, input5, extracted_text)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (username, filename, save_path, *inputs, extracted_text))
    connection.commit()
    cursor.close()
    connection.close()

    return jsonify({'success': True})

@app.route('/review')
def review():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('review.html', username=session['username'])

@app.route('/submit_qc', methods=['POST'])
@login_required
@role_required(['qc', 'lead'])
def submit_qc():
    filename = request.form.get('filename')
    username = session.get('username')

    if filename and username:
        try:
            # Log received data
            print(f'Received filename: {filename}, username: {username}')

            # Update qc_check to 'Yes' and add QC submission time
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute("""
                UPDATE ocrdata 
                SET qc_check = 'Yes', qc_done_by = %s, qc_submission_time = CURRENT_TIMESTAMP
                WHERE filename = %s
            """, (username, filename))
            connection.commit()

            # Move the file from 'uploaded' folder to 'qc' folder
            uploaded_folder = os.path.join(app.root_path, 'static', 'uploads/processed_files')
            qc_folder = os.path.join(app.root_path, 'static', 'qc')

            source_file = os.path.join(uploaded_folder, filename)
            target_file = os.path.join(qc_folder, filename)

            if os.path.exists(source_file):
                shutil.move(source_file, target_file)
            else:
                return jsonify({'success': False, 'message': 'File not found in uploaded folder.'})

            # Fetch the next file for the same user with qc_check = 'No'
            next_file = find_next_file_for_user(username)  # Placeholder function to find the next file

            # Log successful operation
            print(f'File processed successfully for user: {username}')

            # Close cursor and connection
            cursor.close()
            connection.close()

            return jsonify({'success': True, 'next_file': next_file})
        except Exception as e:
            print(f'Error during QC submission: {e}')
            cursor.close()
            connection.close()
            return jsonify({'success': False, 'message': str(e)})
    else:
        print('Missing filename or username')
        return jsonify({'success': False, 'message': 'Filename or username is missing.'})

def find_next_file_for_user(username):
    """Find the next file for the given user."""
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT filename FROM ocrdata WHERE username = %s AND qc_check = 'No' LIMIT 1", (username,))
    next_file_row = cursor.fetchone()  # This returns a tuple
    cursor.close()
    connection.close()
    
    # Access the filename from the tuple
    return next_file_row[0] if next_file_row else None

@app.route('/qcreports')
def qc_reports():
    if 'username' in session:
        username = session['username']
        connection = database.get_db_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)
        
        # Fetch QC reports data for the logged-in user
        cursor.execute("""
            SELECT username, filename, qc_check, qc_done_by, qc_submission_time 
            FROM ocrdata
            WHERE qc_done_by = %s
            ORDER BY created_time DESC
        """, (username,))
        reports = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return render_template('qcreports.html', username=username, reports=reports)
    else:
        return redirect(url_for('login'))  # Redirect to login if user is not authenticated

@app.route('/get_submissions')
def get_submissions():
    if 'username' not in session:
        return jsonify({'error': 'User not logged in'}), 401

    username = session['username']
    date = request.args.get('date')

    if date:
        try:
            # Ensure the date format is correct
            formatted_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid date format'}), 400
    else:
        # Default to today's date if no date is provided
        formatted_date = datetime.date.today()

    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    query = '''
        SELECT filename, input1, input2, input3, input4, input5, created_time
        FROM ocrdata
        WHERE username = %s AND DATE(created_time) = %s
    '''
    
    cursor.execute(query, (username, formatted_date))
    submissions = cursor.fetchall()
    conn.close()

    return jsonify({'submissions': submissions})

@app.route('/get_ocr_users', methods=['GET'])
@login_required
@role_required(['qc', 'lead'])
def get_ocr_users():
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT DISTINCT username FROM ocrdata")
    users = cursor.fetchall()
    cursor.close()
    connection.close()
    return jsonify({'users': [user['username'] for user in users]})

@app.route('/get_uploaded_files')
@login_required
def get_uploaded_files():
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT DISTINCT filename FROM ocrdata")
    files = cursor.fetchall()
    cursor.close()
    connection.close()
    return jsonify({'files': files})

@app.route('/get_files_by_user', methods=['GET'])
@login_required
@role_required(['qc', 'lead'])
def get_files_by_user():
    username = request.args.get('username')
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    
    # Select files where qc_check is 'No'
    cursor.execute("SELECT filename FROM ocrdata WHERE username = %s AND qc_check = 'No'", (username,))
    files = cursor.fetchall()
    
    cursor.close()
    connection.close()
    
    return jsonify({'files': [file['filename'] for file in files]})

@app.route('/get_file_details')
@login_required
def get_file_details():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'success': False, 'message': 'Filename not provided'})

    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    cursor.execute("""
        SELECT filename, input1, input2, input3, input4, input5
        FROM ocrdata
        WHERE filename = %s
    """, (filename,))
    file_details = cursor.fetchone()
    cursor.close()
    connection.close()

    if not file_details:
        return jsonify({'success': False, 'message': 'File not found'})

    # Path to the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Read and convert the image
    try:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'success': False, 'message': 'Failed to read image'}), 500
        
        # Convert the image to PNG format
        _, buffer = cv2.imencode('.png', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

    # Return the image, input fields, and username
    return jsonify({
        'success': True,
        'image': image_base64,
        'input1': file_details['input1'],
        'input2': file_details['input2'],
        'input3': file_details['input3'],
        'input4': file_details['input4'],
        'input5': file_details['input5']
    })

@app.route('/download_csv')
def download_csv():
    connection = database.get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    cursor.execute("""
        SELECT username, filename, input1, input2, input3, input4, input5, qc_check, qc_done_by, created_time, qc_submission_time
        FROM ocrdata
        WHERE qc_check = 'Yes'
        ORDER BY created_time DESC
    """)
    reports = cursor.fetchall()
    cursor.close()
    connection.close()
    
    df = pd.DataFrame(reports)
    csv = df.to_csv(index=False)
    return send_file(BytesIO(csv.encode()), download_name='final_reports.csv', as_attachment=True, mimetype='text/csv')


@app.route('/download_excel')
def download_excel():
    connection = database.get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    cursor.execute("""
        SELECT username, filename, input1, input2, input3, input4, input5, qc_check, qc_done_by, created_time, qc_submission_time
        FROM ocrdata
        WHERE qc_check = 'Yes'
        ORDER BY created_time DESC
    """)
    reports = cursor.fetchall()
    cursor.close()
    connection.close()
    
    df = pd.DataFrame(reports)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Final Reports')
    output.seek(0)
    return send_file(output, download_name='final_reports.xlsx', as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from io import BytesIO
from flask import send_file
import pymysql

@app.route('/download_pdf')
def download_pdf():
    # Fetch data from the database
    connection = database.get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    cursor.execute("""
        SELECT username, filename, input1, input2, input3, input4, input5, qc_check, qc_done_by, created_time, qc_submission_time
        FROM ocrdata
        WHERE qc_check = 'Yes'
        ORDER BY created_time DESC
    """)
    reports = cursor.fetchall()
    cursor.close()
    connection.close()

    # Buffer to hold the PDF
    pdf_buffer = BytesIO()

    # PDF document setup (use landscape for more space)
    doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(letter))
    elements = []

    # Title
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_style.alignment = TA_CENTER
    title = Paragraph("Final Reports", title_style)
    elements.append(title)

    # Create table headers
    table_data = [['Username', 'Filename', 'Input1', 'Input2', 'Input3', 'Input4', 'Input5', 'QC Check', 'QC Done By', 'Created Time', 'QC Submission Time']]

    # Add report data to table
    for report in reports:
        row = [
            report['username'], 
            report['filename'], 
            report['input1'], 
            report['input2'], 
            report['input3'], 
            report['input4'], 
            report['input5'], 
            report['qc_check'], 
            report['qc_done_by'], 
            str(report['created_time']), 
            str(report['qc_submission_time'])
        ]
        table_data.append(row)

    # Adjust column widths dynamically to fit the content
    col_widths = [
        1.5 * inch,  # Username
        2.0 * inch,  # Filename
        1.5 * inch,  # Input1
        1.5 * inch,  # Input2
        1.5 * inch,  # Input3
        1.5 * inch,  # Input4
        1.5 * inch,  # Input5
        1.0 * inch,  # QC Check
        1.5 * inch,  # QC Done By
        2.0 * inch,  # Created Time
        2.0 * inch   # QC Submission Time
    ]

    # Define table style
    table = Table(table_data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),  # Header background
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center all text
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Bold headers
        ('FONTSIZE', (0, 0), (-1, -1), 9),  # Font size
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # Padding for header
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),  # Row background color
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),  # Gridlines
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),  # Alternate row colors
    ]))

    elements.append(table)

    # Build the PDF
    doc.build(elements)

    # Return the PDF as a downloadable file
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, download_name='final_reports.pdf', as_attachment=True, mimetype='application/pdf')


@app.route('/logout')
def logout():
    session.pop('logged_in')
    session.pop('username')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
