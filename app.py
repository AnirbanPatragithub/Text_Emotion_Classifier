
from base64 import encode
from secrets import choice
from turtle import color
import streamlit as st
import altair as alt

import pandas as pd
import numpy as np

import joblib

pipe_lr = joblib.load('models/Text_Emotion_Detection_15-09-2022.pkl')

def predict_emotion(docx):
    result = pipe_lr.predict([docx])
    return result[0]

def get_prediction_proba(docx):
    result = pipe_lr.predict_proba([docx])
    return result

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


def main():
    st.title('Emotion-Classifier App')
    menu = ['Home','Monitor','About']
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == 'Home':
        
        st.subheader('Text-Emotion')
        
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area('Type Here')
            submit_text= st.form_submit_button('Submit')

        if submit_text:
            col1,col2 = st.columns(2)

        #Apply Function Here
            prediction = predict_emotion(raw_text)
            probability = get_prediction_proba(raw_text)    

            with col1:
                st.success('Original Text')
                st.write(raw_text)

                st.success('Prediction')
                emoji_icon = emotions_emoji_dict[prediction]
                st.write('{}:{} '.format(prediction,emoji_icon))
                st.write('Confidence:{}'.format(np.max(probability)))
                
    
            with col2:
                st.success('Prediction Probability')
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ['Emotions','Probability']
                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Emotions',y='Probability',color='Emotions')
                st.altair_chart(fig,use_container_width=True)

    elif choice == 'Monitor':
        st.subheader('Monitor App')
        st.markdown('Work In Progress,please head to Home or About Section.Thank You :smirk:')

    else:
        st.subheader('About')
        
         
        
         

        st.write('''Text-Emotion-Detection App can classify texts based on seven different emotions.
        It can be used in real-world scenarios to ascertain the response from reviews and comments.
        To use this app enter the text and press submit in the Home Section.
        It will also display a chart with different probabilities and the most prominent emotion will be the output.
        Designed and Developed : ''')
        
        st.markdown('[Anirban Patra](https://anirbanpatragithub.github.io/)') 


if __name__=='__main__':
    main()
