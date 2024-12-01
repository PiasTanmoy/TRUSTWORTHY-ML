GCS_motor =  {
"1 No Response": 1,
"No response": 1,
"2 Abnorm extensn": 2,
"Abnormal extension": 2,
"3 Abnorm flexion": 3,
"Abnormal Flexion": 3,
"4 Flex-withdraws": 4,
"Flex-withdraws": 4,
"5 Localizes Pain": 5,
"Localizes Pain": 5,
"6 Obeys Commands": 6,
"Obeys Commands": 6
}


GCS_motor_reverse =  {
1: "1 No Response",
2: "2 Abnorm extensn",
3: "3 Abnorm flexion",
4: "4 Flex-withdraws",
5: "5 Localizes Pain",
6: "6 Obeys Commands",
}



GCS_eye = {
"None": 0,
"1 No Response": 1,
"2 To pain": 2,
"To Pain": 2,
"3 To speech": 3,
"To Speech": 3,
"4 Spontaneously": 4,
"Spontaneously": 4
}


GCS_eye_reverse = {
0: "None",
1: "1 No Response",
2: "2 To pain",
3: "3 To speech",
4: "4 Spontaneously",
}


GCS_speech = {
"No Response-ETT": 1,
"No Response": 1,
"1 No Response": 1,
"1.0 ET/Trach": 1,
"2 Incomp sounds": 2,
"Incomprehensible sounds": 2,
"3 Inapprop words": 3,
"Inappropriate Words": 3,
"4 Confused": 4,
"Confused": 4,
"5 Oriented": 5,
"Oriented": 5
}


GCS_speech_reverse = {
1: "No Response-ETT",
2: "2 Incomp sounds",
3: "3 Inapprop words",
4: "4 Confused",
5: "5 Oriented",
}





# # Load the data from the CSV file

# df['Glascow coma scale eye opening'] = df['Glascow coma scale eye opening'].map(GCS_eye)
# df['Glascow coma scale motor response'] = df['Glascow coma scale motor response'].map(GCS_motor)
# df['Glascow coma scale verbal response'] = df['Glascow coma scale verbal response'].map(GCS_speech)
# df['Glascow coma scale total'] = df['Glascow coma scale total'].astype('Int64')

def GCS_int(df):
    df['Glascow coma scale eye opening'] = df['Glascow coma scale eye opening'].map(GCS_eye)
    df['Glascow coma scale motor response'] = df['Glascow coma scale motor response'].map(GCS_motor)
    df['Glascow coma scale verbal response'] = df['Glascow coma scale verbal response'].map(GCS_speech)
    df['Glascow coma scale total'] = df['Glascow coma scale total'].astype('Int64')

    return df
    