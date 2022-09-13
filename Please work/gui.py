import PySimpleGUI as sg
from sales import *

sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
layout = [  
            [sg.Text('Predictor')],
            [sg.Text('Gender', size=(15, 1)), sg.Radio('M', 'group1', key='-Male-'), sg.Radio('F', 'group1', key='-Female-')],
            [sg.Text('Age', size=(15, 1)), sg.InputText(key='-Age-')],
            [sg.Text('Annual Salary', size=(15, 1)), sg.InputText(key='-Salary-')],
            [sg.Text('Credit Dept', size=(15, 1)), sg.InputText(key='-Dept-')],
            [sg.Text('Net Worth', size=(15, 1)), sg.InputText(key='-Worth-')],
            [sg.Button('Ok'), sg.Button('Cancel')]
        ]
window = sg.Window('Window Title', layout)
while True:
    event, values = window.read()
    
    if event == "Ok":
        if values["-Male-"]==True:
            gender=1
        elif values["-Female-"]==True:
            gender=0
            
        input_test_sample=np.array([[gender, values["-Age-"], values["-Salary-"], values["-Dept-"], values["-Worth-"]]])
        input_test_sample_scaled=scaler_in.transform(input_test_sample)
        output_predict_sample_scaled=model.predict(input_test_sample_scaled)
        print('Predicted Output (Scaled) =', output_predict_sample_scaled)
        output_predict_sample=scaler_out.inverse_transform(output_predict_sample_scaled)
        print('Predict Output / Purchase Amount ', output_predict_sample)
        
        sg.popup('Predicted Output / Purchase Amount ', output_predict_sample)
        
    elif event == sg.WIN_CLOSED:
        break
    
window.close

