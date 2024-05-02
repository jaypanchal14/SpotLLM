import streamlit as st
import json
import requests

# To run the file
# streamlit run --server.headless true streamlit-ui.py

st.title('SpotLLM')

# Input box for user input
user_input = st.text_area('Enter text here', height=200)

def call_fastapi(text):
    # URL of your FastAPI endpoint
    url = 'http://localhost:8000/predict'
    # Data to be sent in the request
    data = {'text': text}
    # Send POST request to FastAPI
    response = requests.post(url, data=json.dumps(data))
    return response.json()


if st.button('Predict'):
    if user_input:
        # Call FastAPI with user input
        result = call_fastapi(user_input)

        lab1 = result['labels'][0]
        lab2 = result['labels'][1]

        # res1 = result['result'][0] * 100
        # res2 = result['result'][1] * 100
        res1 = result['result'][0]
        res2 = result['result'][1]

        st.slider(lab1, 0, 100, int(res1 *100))
        st.slider(lab2, 0, 100, int(res2 *100))

        st.write(lab1, "( /100)", res1)
        st.write(lab2, "( /100)",res2)
        # st.write("Visualization:")
        # st.bar_chart({lab1: res1, lab2:res2})
        # st.write('Result:', result)
    else:
        st.write('Please enter some text.')