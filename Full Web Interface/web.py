import streamlit as st
import pandas as pd
import pickle
st.title("Exoplanet Detector 🪐")


#Importing the models from files using pickle


with open('XGB.pkl','rb') as file: 
    XGB=pickle.load(file)

with open('XGB_scaler.pkl','rb') as file: 
    scaler=pickle.load( file)

with open('LGBM.pkl','rb') as file: 
    LGBM=pickle.load( file)

with open('cat.pkl','rb') as file: 
    cat=pickle.load( file)


#creating a form in streamlit
with st.form('user_input'):
    
    #The drop down menu for model selction
    option = st.selectbox(
        "Select a ML model:",
        ("XGB", "CAT", "LGBM")
    )


    #Creating columns to arrange the inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        Period = st.number_input("Orbital Period",step=1.0)
        duration = st.number_input("Transit Duration",step=1.0)
        prad = st.number_input("Planet Radius",step=1.0)
        stellar_logg = st.number_input("Stellar Logg",step=1.0)

    with col2:
        t = st.number_input("Mid-Transit Time",step=1.0)
        dep = st.number_input("Transit Depth",step=1.0)
        mes = st.number_input("MES",step=1.0)
        stellar_radius = st.number_input("Stellar Radius", step=1.0)

    with col3:
        stellar_teff = st.number_input("Stellar TEFF", step=1.0)
        impact_para = st.number_input("Impact Parameter", step=1.0)
        transit_snr = st.number_input("Transit snr", step=1.0)
        mag = st.number_input("MAG", step=1.0)
    
    
    button = st.form_submit_button("🔭 Predict Exoplanet")
    

    Data={
        'orbital_period':[Period],
        'transit_duration':[duration],
        'transit_depth':[dep],
        'planet_radius':[prad],
        'impact_parameter':[impact_para],
        'transit_snr':[transit_snr],
        'mes':[mes],
        'stellar_teff':[stellar_teff],
        'stellar_logg':[stellar_logg],
        'stellar_radius':[stellar_radius],
        'mag':[mag]

        
    }
    cf = pd.DataFrame(Data)

if button:   
    
    X = scaler.transform(cf)
    

    match option:
        case "XGB":
            prediction = XGB.predict(X)
            st.metric(label="Average Accuracy ✅", value="74%")
        case "CAT":
            prediction = cat.predict(X)
            st.metric(label="Average Accuracy ✅", value="73%")
        case "LGBM":
            prediction = LGBM.predict(X)
            st.metric(label="Average Accuracy ✅", value="72%")

    pred = prediction[0]
    if pred == 0:
        st.error("🚫 **False Positive** — Not a planet.")
    else:
        st.success("✅ **Planetary Candidate Detected!**")
   


    