# blindness.py

from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import pymongo
import bcrypt
from model import inference  # Import inference function from model.py
from send_sms import Send  # Import Send class from send_sms.py

print('GUI SYSTEM STARTED...')

# ---------------------------------------------------------------------------------
# **Database Connection - MongoDB**
try:
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["batch_db_new"]
    users_collection = db["users"]  # Collection to store user data
    print("Connected to MongoDB successfully!")
except Exception as e:
    messagebox.showerror("Database Error", f"Failed to connect to MongoDB!\n{e}")
    exit()

y = False  # Global login status variable

def hash_password(password):
    """Hashes the password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(stored_password, entered_password):
    """Verifies hashed password."""
    return bcrypt.checkpw(entered_password.encode(), stored_password)

def LogIn():
    """Handles user login using MongoDB."""
    global y
    username = box1.get().strip()
    password = box2.get().strip()

    if not username or not password:
        messagebox.showinfo("Error", "Username and Password cannot be empty!")
        return

    user = users_collection.find_one({"_id": username})

    if user and check_password(user["password"], password):
        messagebox.showinfo('Success', f'Welcome {username}!')
        y = True  # Set login status
    else:
        messagebox.showinfo('Error', 'Invalid Username or Password')

def OpenFile():
    """Handles file upload, prediction, and sends SMS."""
    if not y:
        messagebox.showinfo("Error", "You need to Login first!")
        return

    file_path = askopenfilename()
    if not file_path:
        messagebox.showerror("Error", "No file selected!")
        return

    try:
        value, classes = inference(file_path)  # Perform prediction
        messagebox.showinfo("Prediction", f"Predicted Label: {value}\nPredicted Class: {classes}")

        # Update MongoDB with prediction
        username = box1.get().strip()
        users_collection.update_one({"_id": username}, {"$set": {"predict": value}})

        # Send SMS using Twilio
        sms_sender = Send()
        sms_sender.send_msg(value, classes)

        # Display image with prediction
        image = Image.open(file_path).convert('RGB')
        plt.imshow(np.array(image))
        plt.title(f'Label: {value}, Class: {classes}')
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong!\n{e}")

def Signup():
    """Handles user registration using MongoDB."""
    username = box1.get().strip()
    password = box2.get().strip()

    if not username or not password:
        messagebox.showinfo("Error", "Username and Password cannot be empty!")
        return

    # Check if user already exists
    if users_collection.find_one({"_id": username}):
        messagebox.showinfo("Error", "Username already exists. Try another one.")
        return

    # Store hashed password
    hashed_password = hash_password(password)

    # Insert new user into MongoDB
    users_collection.insert_one({"_id": username, "password": hashed_password, "predict": None})
    messagebox.showinfo("Success", f"Hi {username}, you can now log in!")

# -----------------------------------------------------------------------------------------

root = Tk()
root.geometry('1000x500')
root.title("Diabetic Retinopathy Detection")

# Background image handling
try:
    filename = PhotoImage(file="im1.jpg")
    background_label = Label(root, image=filename)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
except:
    print("Warning: Background image 'im1.png' not found!")

# Labels and Inputs
Label(root, text="Diabetic Retinopathy Detection", font=('Arial', 30)).grid(padx=10, pady=30, row=0, column=0, sticky='W')
Label(root, text="Enter your username: ", font=('Arial', 20)).grid(padx=10, pady=10, row=1, column=0, sticky='W')
Label(root, text="Enter your password: ", font=('Arial', 20)).grid(padx=10, pady=10, row=2, column=0, sticky='W')

box1 = Entry(root)
box1.grid(row=1, column=1)

box2 = Entry(root, show='*')
box2.grid(row=2, column=1)

Button(root, text="Signup", command=Signup).grid(padx=20, pady=30, row=3, column=1)
Button(root, text="LogIn", command=LogIn).grid(padx=20, pady=30, row=3, column=2)
Button(root, text="Upload Image", command=OpenFile).grid(padx=20, pady=30, row=2, column=3)

# Start GUI
root.mainloop()
