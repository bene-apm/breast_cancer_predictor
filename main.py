import sklearn
import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    data = pd.read_csv('data/data.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis = 1)
    data['diagnosis'] = data['diagnosis'].map({ 'M' : 1, 'B': 0 })

    return data

def add_sidebar():
    st.sidebar.header('ค่าจากห้องปฏิบัติการเซลล์วิทยา')

    data = get_clean_data()

    slider_labels = [
        ("เส้นรัศมี (ค่าเฉลี่ย)", "radius_mean"),
        ("เนื้อสัมผัส (ค่าเฉลี่ย)", "texture_mean"),
        ("เส้นรอบวง (ค่าเฉลี่ย)", "perimeter_mean"),
        ("พื้นที่ (ค่าเฉลี่ย)", "area_mean"),
        ("ความเรียบเนียน (ค่าเฉลี่ย)", "smoothness_mean"),
        ("ความแน่น (ค่าเฉลี่ย)", "compactness_mean"),
        ("ความโค้งเว้า (ค่าเฉลี่ย)", "concavity_mean"),
        ("จุดเว้า (ค่าเฉลี่ย)", "concave points_mean"),
        ("ความสมมาตร (ค่าเฉลี่ย)", "symmetry_mean"),
        ("รูปแบบย่อยที่ซ้ำซ้อน (ค่าเฉลี่ย)", "fractal_dimension_mean"),
        ("เส้นรัศมี (ค่าความคลาดเคลื่อนมาตรฐาน)", "radius_se"),
        ("เนื้อสัมผัส (ค่าความคลาดเคลื่อนมาตรฐาน)", "texture_se"),
        ("เส้นรอบวง (ค่าความคลาดเคลื่อนมาตรฐาน)", "perimeter_se"),
        ("พื้นที่ (ค่าความคลาดเคลื่อนมาตรฐาน)", "area_se"),
        ("ความเรียบเนียน (ค่าความคลาดเคลื่อนมาตรฐาน)", "smoothness_se"),
        ("ความแน่น (ค่าความคลาดเคลื่อนมาตรฐาน)", "compactness_se"),
        ("ความโค้งเว้า (ค่าความคลาดเคลื่อนมาตรฐาน)", "concavity_se"),
        ("จุดเว้า (ค่าความคลาดเคลื่อนมาตรฐาน)", "concave points_se"),
        ("ความสมมาตร (ค่าความคลาดเคลื่อนมาตรฐาน)", "symmetry_se"),
        ("รูปแบบย่อยที่ซ้ำซ้อน (ค่าความคลาดเคลื่อนมาตรฐาน)", "fractal_dimension_se"),
        ("เส้นรัศมี (ค่าวิกฤติ)", "radius_worst"),
        ("เนื้อสัมผัส (ค่าวิกฤติ)", "texture_worst"),
        ("เส้นรอบวง (ค่าวิกฤติ)", "perimeter_worst"),
        ("พื้นที่ (ค่าวิกฤติ)", "area_worst"),
        ("ความเรียบเนียน (ค่าวิกฤติ)", "smoothness_worst"),
        ("ความแน่น (ค่าวิกฤติ)", "compactness_worst"),
        ("ความโค้งเว้า (ค่าวิกฤติ)", "concavity_worst"),
        ("จุดเว้า (ค่าวิกฤติ)", "concave points_worst"),
        ("ความสมมาตร (ค่าวิกฤติ)", "symmetry_worst"),
        ("รูปแบบย่อยที่ซ้ำซ้อน (ค่าวิกฤติ)", "fractal_dimension_worst")
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(label, 
                          min_value = float(0), 
                          max_value = float(data[key].max()),
                          value = float(data[key].mean()))

    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
  
    X = data.drop(['diagnosis'], axis=1)
  
    scaled_dict = {}
  
    for key, value in input_dict.items():
        max_value = X[key].max()
        min_value = X[key].min()
        scaled_value = (value - min_value) / (max_value - min_value)
        scaled_dict[key] = scaled_value
  
    return scaled_dict


def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['เส้นรัศมี','เนื้อสัมผัส','เส้นรอบวง', 'พื้นที่','ความเรียบเนียน', 'ความแน่น', 
                  'ความโค้งเว้า', 'จุดเว้า', 'ความสมมาตร','รูปแบบย่อยที่ซ้ำซ้อน']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
                                     input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
                                     input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
                                     input_data['fractal_dimension_mean']], theta=categories, fill='toself', name='ค่าเฉลี่ย'))
    
    fig.add_trace(go.Scatterpolar(r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
                                     input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
                                     input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
                                     input_data['fractal_dimension_se']], theta=categories, fill='toself', name='ค่าความคลาดเคลื่อนมาตรฐาน'))

    fig.add_trace(go.Scatterpolar(r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
                                     input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
                                     input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
                                     input_data['fractal_dimension_worst']], theta=categories, fill='toself', name='ค่าวิกฤติ'))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)

    return fig

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
  
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    scaled_input_array = scaler.transform(input_array)
    prediction = model.predict(scaled_input_array)

    st.subheader('การทำนายโอกาสการเป็นมะเร็งเต้านม')
    st.write('กลุ่มตัวอย่างเซลล์ :')
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>ไม่อันตราย</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>อันตราย</span>", unsafe_allow_html=True)
    st.write('ความน่าจะเป็นที่จะกลายเป็นเซลล์มะเร็ง :', model.predict_proba(scaled_input_array)[0][1])
    st.write('ความน่าจะเป็นที่ไม่กลายเป็นเซลล์มะเร็ง :', model.predict_proba(scaled_input_array)[0][0])
    st.write('แอพพลิเคชั่นนี้สามารถช่วยเหลือผู้เชี่ยวชาญทางการแพทย์ในการวินิจฉัยโรคได้ แต่ไม่ควรใช้แทนการวินิจฉัยโดยผู้เชี่ยวชาญ')

def main():
    st.set_page_config(page_title='Breast Cancer Predictor(BCP)', layout='wide', initial_sidebar_state='expanded')

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
    with st.container():
        st.title('Breast Cancer Predictor (BCP)')
        st.write('''โปรดเชื่อมต่อแอพพลิเคชั่นนี้กับผลตรวจจากห้องปฏิบัติการเซลล์วิทยา เพื่อช่วยวินิจฉัยมะเร็งเต้านมจากตัวอย่างเนื้อเยื่อ 
                 ว่ามวลเต้านมนั้นจะกลายเป็นเนื้อร้ายหรือไม่ แอพพลิเคชั่นนี้คาดการณ์โดยใช้โมเดลการเรียนรู้ของเครื่อง คุณยังสามารถปรับ
                 แต่งการวัดด้วยมือได้ที่แถบเลื่อนด้านซ้าย''')

    input_data = add_sidebar()

    col1, col2 = st.columns([4,1])
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
        
    with col2:
        add_predictions(input_data)

if __name__ == '__main__':
    main()