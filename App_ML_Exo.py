import streamlit as st
import pandas as pd
import pickle

st.image("https://www.laformation.ma/images/contenu/855904b5b.jpg")
st.title("MSDE6 : ML Course")
st.header("Iris Flower Prediction App")
st.markdown("This app predicts the iris flower type")
st.selectbox('How would you like to use the prediction model', ['Input parameters directly', 'Load a file o data'])
st.sidebar.image("https://cdn.britannica.com/39/91239-004-44353E32/Diagram-flowering-plant.jpg")
st.sidebar.markdown("User Input Parameters")

# Charger le modèle de machine learning
with open(r'C:\Users\072850\Desktop\MASTER_EHTP\M06_Machine-Learning\02_Labs\10_Labs_deploiement\10_deploiement\Streamlit\modeliris6.pkl', 'rb') as file:
    model = pickle.load(file)
    
# Création des sliders dans la barre latérale
sepal_length = st.sidebar.slider('Sepal Length', 4.00, 8.00)
sepal_width = st.sidebar.slider('Sepal Width', 2.00, 5.00)
petal_length = st.sidebar.slider('Petal Length', 1.00, 7.00)
petal_width = st.sidebar.slider('Petal Width', 0.10, 3.00)

# Organiser les valeurs sélectionnées dans un dictionnaire
data = {
    "Sepal Length": [sepal_length],
    "Sepal Width": [sepal_width],
    "Petal Length": [petal_length],
    "Petal Width": [petal_width]
}

# Convertir le dictionnaire en DataFrame
df_selected = pd.DataFrame(data)

# Afficher les valeurs sélectionnées dans un tableau
st.write("User Input Parameters :")
st.dataframe(df_selected)

# Bouton pour effectuer la prédiction
if st.button("Prédire"):
    # Faire une prédiction en utilisant le modèle
    prediction = model.predict(df_selected)
    
    # Afficher le résultat de la prédiction
    st.write("Résultat de la prédiction :", prediction[0])




