import bcrypt
import pymysql
from flask import current_app

def get_db_connection():
    return pymysql.connect(
        host=current_app.config['MYSQL_HOST'],
        user=current_app.config['MYSQL_USER'],
        password=current_app.config['MYSQL_PASSWORD'],
        database=current_app.config['MYSQL_DB'],
        ssl={'ca': current_app.config['MYSQL_SSL_CA']}  # Use current_app for SSL configuration
    )

def create_database_and_table():
    connection = pymysql.connect(
        host=current_app.config['MYSQL_HOST'],
        user=current_app.config['MYSQL_USER'],
        password=current_app.config['MYSQL_PASSWORD'],
        ssl={'ca': current_app.config['MYSQL_SSL_CA']}  # Use current_app for SSL configuration
    )
    cursor = connection.cursor()

    # Create database if it doesn't exist
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {current_app.config['MYSQL_DB']}")

    # Connect to the database
    connection.select_db(current_app.config['MYSQL_DB'])

   # Create ocrdata table with all required columns, including qc_submission_time
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ocrdata (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255),
            filename VARCHAR(255),
            file_path VARCHAR(255),  # Add file_path column
            input1 TEXT,
            input2 TEXT,
            input3 TEXT,
            input4 TEXT,
            input5 TEXT,
            extracted_text TEXT,
            qc_check ENUM('Yes', 'No') DEFAULT 'No',  # Add qc_check column with default value 'No'
            qc_done_by VARCHAR(255),  # Add qc_done_by column to store the username who submitted QC
            created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            qc_submission_time TIMESTAMP NULL DEFAULT NULL  # Add qc_submission_time column
        )
    """)

    # Create user table with role column
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) UNIQUE,
            password_hash VARCHAR(255),
            role ENUM('dataentry', 'qc', 'lead') DEFAULT 'dataentry'
        )
    """)

    connection.commit()
    cursor.close()
    connection.close()

def insert_data(username, filename, file_path, inputs, extracted_text):
    connection = get_db_connection()
    cursor = connection.cursor()

    qc_check = 'No'  # Default qc_check value for data entry submission

    cursor.execute("""
        INSERT INTO ocrdata (username, filename, file_path, input1, input2, input3, input4, input5, extracted_text, qc_check)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (username, filename, file_path, *inputs, extracted_text, qc_check))

    connection.commit()
    cursor.close()
    connection.close()
    
def get_user_by_username(username):
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    cursor.execute("""
        SELECT username, password_hash, role FROM user WHERE username = %s
    """, (username,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    return user

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
    cursor.execute("""
        INSERT INTO user (username, password_hash, role)
        VALUES (%s, %s, %s)
    """, (username, hashed.decode('utf-8'), role))
    connection.commit()
    cursor.close()
    connection.close()
    return True

def get_submissions_by_username_and_date(username, date):
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    
    query = """
        SELECT filename, input1, input2, input3, input4, input5, created_time
        FROM ocrdata
        WHERE username = %s AND DATE(created_time) = %s
    """
    cursor.execute(query, (username, date))
    submissions = cursor.fetchall()
    
    cursor.close()
    connection.close()
    return submissions
