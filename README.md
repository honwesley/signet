# Project SIGNET
# Wesley Hon -- UCLA Electrical Engineering 

# SIGNET
SIGNET is a real time American Sign Language fingerspelling recognition prototype. It uses a webcam, Mediapipe hand landmarks, and machine-learning classifiers to convert ASL letters into on-screen text.

## Features
1. Recognizes 24 static ASL letters  
2. Recognizes motion-based J and Z  
3. Converts recognized letters into text
4. Displays prediction confidence
5. Supports spaces, backspace, and a clearing text
6. Runs locally using a webcam

## Important Limitation
SIGNET currently recognizes ASL fingerspelling letters. It is not a complete ASL translation system. ASL also uses movement, facial expressions, body position, and grammar. 

## Technologies
Python 3.12  
OpenCV  
MediaPipe  
scikit-learn  
pandas  
Numpy  
joblib  

## Installation
### Clone Repo
in powershell:   
git clone https://github.com/honwesley/signet.git   
cd signet   
### Create and activate a virtual environment
in powershell:    
py -3.12 -m venv .venv   
.\.venv\Scripts\Activate.ps1   
### Install dependencies
in powershell:    
python -m pip install -r requirements.txt   

## Download Mediapipe Model
New-Item -ItemType Directory -Force models   
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/float16/latest/ hand_landmarker.task" -Outfile "models\hand_landmarker.task"  

## Test webcam
python tools/camera_test.py  

Press 'Q' to close the camera.  

## Collect static-letter data
python training/collect_data.py A  

Press 'Space' to begin recording. Slowly vary the wrist angle and camera distance whlie maintaining the correct handshape.  
J and Z are excluded because they require motion.  

## Train static classifier
python training/train_model.py  

Model saved as: models/asl_classifier.joblib  

## Collect J and Z motion data
Collect examples for all three motion classes  

in powershell:  
python training/collect_motion_data.py J  
python training/collect_motion_data.py Z   
python training/collect_motion_data.py OTHER   

## Train the motion classifier
python training/train_motion_model.py  

Model saved as: models/motion_classifier.joblib  

## Run complete application
in powershell:    
python app/full_asl_app.py (v 1.1)  
python app/gui_app.py (v 1.2)  

### Controls
Key -> Action:    
'M' -> Record a J or Z motion  
'Space' -> Add a space  
'Backspace' -> Remove the last character 
'C' -> Clear the text   
'Q' -> Quit   

Static letters are recognized automatically. For J or Z, press 'M' and immediately perform the complete motion.  
J and Z are detected automatically from their motion now. The manual J/Z button remains available as a fallback.  

### Project Structure
signet/  
    app/  
        full_asl_app.py  
        fingerspell_app.py  
        live_predict.py   
        live_motion_test.py  
        gui_app.py  
        recognition_engine.py  
    training/  
        collect_data.py  
        collect_motion_data.py  
        train_model.py  
        train_motion_model.py  
        verify_data.py  
        reset_letters.py  
    tools/  
        camera_test.py  
        hand_tracker.py  
    data/  
        landmarks.csv  
        motion/  
    models/  
        hand_landmarker.task  
        asl_classifier.joblib  
        motion_classifier.joblib  
    .gitignore   
    README.md  
    requirements.txt  

app/ contains the applications and live recoginition programs.   
training/ contrains data collection, verification, and model-training scripts  
tools/ contains webcam and hand tracking diagnostics  
data/ contains locally collected training samples  
models/ contains downlaoded and trained model files  
 
## Current Limitations
Training data is currently limited  
Accuracy may vary between different users.  
J and Z require a manual motion-recording trigger.   
Similar handshapes may occasionally be confused  
The current model recognizes individual letters, not complete ASL signs or sentences   

## Future Improvements
Train with multiple signers    
Add an unknown-handshape class  
Improve support for left-handed signers  
Add text-to-speech  
Create a graphical user interface  
Evaluate using signer independent testing  
