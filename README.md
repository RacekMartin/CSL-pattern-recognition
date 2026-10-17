# CSL-pattern-recognition
Project for my bachelors thesis - sign language pattern recognition and generating subtitles.

I have already contacted multiple people and organisations within Czech republic to get access to
video data. I'm currently working on pattern recognising script for both camera and video input,
so I can respond to theirs possitive feedback on my emails.

People / organisations  I tried contacting
- Marek Hruz
- Jan Fikejs
- UNB
- CUN
- Alfons
- VOS
- SNNCR
- trojzmer
- Sweet potatoes on the way
- Markét Lundáková
- doc. Ing. Jiří Krejsa, Ph.D
- divadlo Barka

People to try making connection with:
- Snajder
- Krejsa's connections
- Alfons
- Terezia Vasilovčík Šustová
- Tichá Linka

Here are some usefeull links I came across so far whilst finalising my scripts:
https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision
https://developers.google.com/edge/mediapipe/solutions/tasks
https://www.npmjs.com/package/@mediapipe/tasks-vision
https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python
https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
https://developers.google.com/edge/mediapipe/solutions/setup_python

Code examples:
https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb

Comunity:
https://groups.google.com/g/mediapipe
https://stackoverflow.com/questions/tagged/mediapipe

Literature:
https://ubn.ff.cuni.cz/ustav/lide/zamestnanci/mgr-andrea-hudakova-ph-d/

Tutorials I got my inspiration from:
https://www.youtube.com/watch?v=doDUihpj6ro&t=152s
https://www.youtube.com/watch?v=wa2ARoUUdU8

Proof of exissting fingerspelling applicatsaion:
https://www.webbyawards.com/crafted-with-code/fingerspelling-with-machine-learning/

Google's Kaggle project with most refined dataset on ASL:
https://www.kaggle.com/competitions/asl-fingerspelling
https://www.kaggle.com/competitions/asl-fingerspelling/writeups/darragh-dieter-1st-place-solution-improved-squeeze

MUNI's dictio web app:
https://www.dictio.info/cs/translate/czj/text/q?slovni_druh=finger

CNN course by 3blue1brown:
https://www.3blue1brown.com/?topic=neural-networks&lesson=neural-networks

Interesting websites
https://www.pomuckyproneslysici.cz/online-kurz-znakoveho-jazyka-1-lekce-prstova-abeceda-na-dve-ruce-a/

Theatre Barka:
https://divadlobarka.cz

Ticha linka:
https://www.tichalinka.cz/kontakt

The data will be stored as .npz format instead of plain .txt. or common .npy. 
.npz is a ZIP archive which contains multiple NumPt arrays with different names.
These will contain data like: timestamp_ms, landmarks coords, handedness_score,
width, height, fps.

For each participant I will create a CSV record to store necessary metadata such as
gender, age, whether this person is hearing impaired/expert/beginner, handedness,
answers on my questionaire.

For each video I will create a CSV record to store information such as participnat_ID,
resolution, handedness, whether the video is mirrored, resolution, fps etc.

I'm not totally sure how this works, but based on NaN landmark coordinates I can
segment the video to individual character signings.

*** Poznamky z korespendencie s ludmi:


*** Porovnania, ktore zahrnut do prace:
- normalizacia podla vzdialenosti oci vs ramien
- model natrenovany s landmarkmi tvare vs bez
- dosiahnuta presnost obojrucnej abecedy vs jednorucnej u Snajdera
- Transformer vs LSTM
- predtrenovanie na amateroch + finetuning na profikoch vs bez predtrenovania
- predtrenovanie na ASL vs finetuning na profikoch vs bez predtrenovania
- zistenie ci auumentovane data priniesli presnost alebo ju praveze znizili