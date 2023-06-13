from flask import Flask, render_template, request,session,redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import sys
from utils import solver,solvearray
from utils.exception import CustomException
from utils.logger import logging
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)

app.secret_key = '***'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # receiving image file from web page
        uploaded_img = request.files['uploaded-file']

        # Extracting uploaded image data file name
        img_filename = secure_filename(uploaded_img.filename)
        
        # Upload file (defined uploaded folder in static folder)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))

        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        img_file_path = session.get('uploaded_img_file_path', None)
        imgwidth=450
        imgheight=450
        model=solver.initializePredictionModel()
        img = cv2.imread(img_file_path)
        img = cv2.resize(img, (imgwidth, imgheight))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
        imgBlank = np.zeros((imgwidth, imgheight, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
        imgThreshold = solver.preProcess(img)
                
        #FIND CONTOURS
        imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS
        

        # FIND THE BIGGEST CONTOUR
        biggest, maxArea = solver.biggestContour(contours)
        if biggest.size != 0:
            biggest =solver.reorder(biggest)
            cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
            pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0],[imgwidth, 0], [0, imgheight],[imgwidth, imgheight]]) # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
            imgWarpColored = cv2.warpPerspective(img, matrix, (imgwidth, imgheight))
            imgDetectedDigits = imgBlank.copy()
            imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

            #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
            imgSolvedDigits = imgBlank.copy()
            boxes = solver.splitBoxes(imgWarpColored)
            numbers = solver.getPrediction(boxes, model)
            imgDetectedDigits = solver.displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
            numbers = np.asarray(numbers)
            posArray = np.where(numbers > 0, 0, 1)


            #### 5. FIND SOLUTION OF THE BOARD
            board = np.array_split(numbers,9)
            try:
                solvearray.solve(board)
            except Exception as e:
                raise CustomException(e,sys)
        
            flatList = []
            for sublist in board:
                for item in sublist:
                    flatList.append(item)
            solvedNumbers =flatList*posArray
            imgSolvedDigits= solver.displayNumbers(imgSolvedDigits,solvedNumbers)
            cv2.imwrite('static/solved_digit.png', imgSolvedDigits)


            return render_template('error.html', errormsg="image uploaded succesfully",imgSolvedDigits=imgSolvedDigits)
    
        else:
            return render_template('error.html',errormsg="image not uploaded")

if __name__ == '__main__':
    app.run(debug=True,port=5000)