import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

log_model = pickle.load(open('log_model.pkl', 'rb'))
nb_model = pickle.load(open('nb_model.pkl', 'rb'))
rg_model = pickle.load(open('rg_model.pkl', 'rb'))

svm_model = pickle.load(open('svm_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))


tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def classify(num):
    if num == 0:
        return 'Leftist Ideology'
    elif num == 1:
        return 'Neutral Ideology'
    else:
        return 'Rightist Ideology'

def main():
    st.title("BTECH PROJECT -II")
    st.text("")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Prediction of Political Biasness     </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities = ['Logistic Regression', 'Ridge Classifier', 'Naive Bayes', 'Support Vector Machine', 'Random Forest']
    option = st.sidebar.selectbox('Which model would you like to use?', activities)
    st.text("")
    
    title = st.text_input('Write the statement',)
    a_ser = pd.Series(title)
    f = tfidf.transform(a_ser)

    st.subheader(option)

    if st.button('Classify'):
        
        if option == 'Ridge Classifier':
            st.success(classify(rg_model.predict(f)))

        elif option == 'Logistic Regression':
            st.success(classify(log_model.predict(f)))

        elif option == 'Support Vector Machine':
            st.success(classify(svm_model.predict(f)))

        elif option == 'Random Forest':
            st.success(classify(rf_model.predict(f)))

        else:
            st.success(classify(nb_model.predict(f)))

if __name__ == '__main__':
    main()
