import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import streamlit as st


def get_processed_data():
    data = get_clean_data()
    x = data.drop('DEATH_EVENT', axis=1) 
    y = data[['DEATH_EVENT']]

    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=100,test_size=0.2)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test= sc.transform(x_test)

    return x_train,x_test,y_train,y_test


def get_clean_data():
    data = pd.read_csv("data\heart_failure_clinical_records_dataset.csv")
    # print(data.head())
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    data = data.reset_index(drop = True)
    return data

def select_paras(classifier_name):
    paras = dict()
    if(classifier_name == "KNN"):
        k = st.slider("k",1,15)
        paras["k"] = k
    elif(classifier_name == "SVM"):
        c = st.slider("C",0.01,10.0)
        paras["C"] = c
    elif (classifier_name == "Random Forest"):
        n_estimators = st.slider("n_estimators",10,100)
        max_depth = st.slider("max_depth",2,15)
        criterion = st.selectbox("criterion",("gini","entropy","log_loss"))
        paras['max_depth'] = max_depth
        paras['n_estimators'] = n_estimators
        paras['criterion'] = criterion
    elif(classifier_name == 'k Means clustering'):
        n_clusters = st.slider("n_clusters",2,15)
        random_state = st.slider("random_state",2,15)
        paras['n_clusters'] = n_clusters
        paras['random_state'] = random_state
    return paras

def get_model(classifier_name,paras):
    x_train,x_test,y_train,y_test = get_processed_data()

    if(classifier_name == "KNN"):
        model = KNeighborsClassifier(paras["k"])
        # st.write("using KNN")
    elif(classifier_name == "SVM"):
        model = SVC(C=paras["C"])
        # st.write("using SVM")
    elif(classifier_name == "Random Forest"):
        model = RandomForestClassifier(n_estimators=paras['n_estimators'],max_depth=paras['max_depth'],criterion=paras['criterion'])
        # st.write("using random forest")
    elif(classifier_name == 'Logistic Regression'):
        model = LogisticRegression()
        # st.write("using Logistic Regression")
    elif(classifier_name=='Naive Bayes'):
        model = GaussianNB()
    elif(classifier_name == 'k Means clustering'):
        model = KMeans(n_clusters= paras['n_clusters'],random_state=paras['random_state'])
    return model


def main():
    # st.set_page_config(
    #     page_title = "Heart Failure Prediction",
    #     page_icon = ":female-doctor:",
    #     layout='wide',
    #     initial_sidebar_state='expanded',
    # )

    # data = get_clean_data()
    # x_train,x_test,y_train,y_test = get_processed_data(data)


    with st.container():
        st.title("Heart Failure Prediction-App")

    classifier_name = st.sidebar.selectbox("Select Classifier",("Logistic Regression","KNN","SVM","Random Forest","Naive Bayes","k Means clustering"))
    
    if ~(classifier_name == "Logistic Regression" or classifier_name=='Naive Bayes'):
        st.write(f"Select Parameters for {classifier_name}")

    age = st.sidebar.slider("age",1,100)
    anemia = st.sidebar.select_slider("anemia",(0,1))
    creatinine_phosphokinase = st.sidebar.slider("creatinine_phosphokinase",1,2500)
    diabetes = st.sidebar.select_slider("diabetes",(0,1))
    ejection_fraction = st.sidebar.slider("ejection_fraction",1,100)    
    high_blood_pressure = st.sidebar.select_slider("high_blood_pressure",(0,1))
    platelets = st.sidebar.slider("platelets",1,500000)
    serum_creatinine = st.sidebar.slider("serum_creatinine",0.0,10.0)
    serum_sodium = st.sidebar.slider("serum_sodium",100,150)
    sex = st.sidebar.select_slider("sex",(0,1))
    smoking = st.sidebar.select_slider("smoking",(0,1))
    time = st.sidebar.slider("time",0,150)
    paras = select_paras(classifier_name)

    patient_status = {'age':age, 'anaemia':anemia, 'creatinine_phosphokinase':creatinine_phosphokinase, 'diabetes':diabetes,
       'ejection_fraction':ejection_fraction, 'high_blood_pressure':high_blood_pressure, 'platelets':platelets,
       'serum_creatinine':serum_creatinine, 'serum_sodium':serum_sodium, 'sex':sex, 'smoking':smoking, 'time':time}
    

    x_train,x_test,y_train,y_test = get_processed_data()
    model = None
    model = get_model(classifier_name,paras)
    model.fit(x_train,y_train)

    test_preds = model.predict(x_test)
    train_preds = model.predict(x_train)
    # col1.write(accuracy_score(y_test,test_preds))
    # col1.write(accuracy_score(y_train,train_preds))
    st.write( f"testing Accuracy   {round(accuracy_score(y_test,test_preds)*100,2)}%")
    st.write(f"training Accuracy    {round(accuracy_score(y_train,train_preds)*100,2)}%")
    
    repo = confusion_matrix(y_test,test_preds)
    tp = repo[0,0]
    fp = repo[0,1]
    fn = repo[1,0]
    tn = repo[1,1]

    st.write("\n\n")
    col1, col2 ,col3= st.columns(3)
    
    col1.write("CONFUSION MATRIX")
    col2.write("ACTUAl (YES)")
    col3.write("ACTUAL (NO)")
    col1.write("PREDICTIONS (YES)")
    col2.write(tp)
    col3.write(fp)
    col1.write("PREDICTIONS (NO)")
    col2.write(fn)
    col3.write(tn)

    # st.write({confusion_matrix(y_test,test_preds)})
    # repo = classification_report(y_test,test_preds)
    # st.write(repo)
    
    patient_info = [[age, anemia, creatinine_phosphokinase, diabetes,
       ejection_fraction, high_blood_pressure, platelets,
       serum_creatinine, serum_sodium, sex, smoking, time]]
    
    st.write("\n\n")
    st.write("Patient Report")
    st.dataframe(patient_status)
    # st.write(patient_status)
    st.write("Will heart Fail?")
    failure = model.predict(patient_info)
    if(failure == 0):
        st.write("You are SAFE!!! Heart will not fail ")
    else:
        st.write("WARNING!!!! Heart Gonna fail")

if __name__ == '__main__':
    main()

