# Heart Failure Prediction

![title](https://cpb-eu-w2.wpmucdn.com/blogs.brighton.ac.uk/dist/f/6375/files/2019/12/website-pic-2.gif)

Cardiovascular disease (CVD) is the leading cause of deaths globally. An estimated 18 million people died from CVD in 2019, accounting for about 32% of all deaths. Of those, 85% were due to heart attacks and strokes and about 30% occured in people under the age of 70.


### Risk factors for CVD?

The most important behavioral risk factors for heart disease and stroke are unhealthy diet, physical inactivity, tobacco use and excessive consumption of alcohol. The effects of these risk factors may manifest in individuals as high blood pressure, elevated blood glucose and lipids, and obesity. These “intermediate risks factors” can be measured in primary care facilities and indicate an increased risk of heart attack, stroke, heart failure and other complications.

There are also a number of underlying determinants of CVD. These are a reflection of the major forces driving social, economic and cultural change --globalization, urbanization and population aging. Other determinants of CVD include poverty, stress and hereditary factors.

### Predicting Heart Failure?

Heart failure is a common event caused by CVD and this dataset contains 11 features that can be used to predict possible heart disease.

People with CVD or those who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or other comorbidities) need early detection and management wherein a machine learning model could be of great help.

#### Data dictionary:

- Age: age of the patient (years)
- Sex: sex of the patient (M: Male, F: Female)
- ChestPainType: chest pain type (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)
- RestingBP: resting blood pressure (mm Hg)
- Cholesterol: serum cholesterol (mm/dl)
- FastingBS: fasting blood sugar (1: if FastingBS > 120 mg/dl, 0: otherwise)
- RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- MaxHR: maximum heart rate achieved (Numeric value between 60 and 202)
- ExerciseAngina: exercise-induced angina (Y: Yes, N: No)
- Oldpeak: oldpeak = ST (Numeric value measured in depression)
- ST_Slope: the slope of the peak exercise ST segment (Up: upsloping, Flat: flat, Down: downsloping)
- HeartDisease: output class [1: heart disease, 0: Normal]

Reference: https://www.kaggle.com/fedesoriano/heart-failure-prediction
