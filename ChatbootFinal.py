import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath('nltk_data'))
nltk.download('punkt')

# 1. Définir les intentions
intents = [

    {
        'tag': 'Salutation',
        'patterns': ['Salut','salut','Slt','slt', 'Bonjour','bonjour','Bjr','bjr','coucou','Hello','hello'],
        'responses': ['Bonjour comment tu te sens?']
    },
    {
        'tag': 'fièvre',
        'patterns': ['J\'ai mal à la gorge', 'Je tousse' 'Je sens chaud', 'J\'ai les frissons', 'J\'ai des douleurs musculaires', 'les articulations me font mal', 'J\'ai pas d\'appetit'],
        'responses': ['Possibilité de fièvre typhoïde ou paludisme. Consultez un médecin pour un diagnostic précis.']
    },
    {
        'tag': 'cholera',
        'patterns': ['J\'ai les gaz','J\'ai la diarhé', 'Je vais trop aux toilettes', 'j\'ai les nausée', 'j\'ai des vomissements'],
        'responses': ['Possibilité de choléra. Consultez un médecin pour un diagnostic précis.']
    },
    {
        'tag': 'fatigue',
        'patterns': ['Je me sens fatigué', 'j\'ai des vomissements', 'j\'ai des Nauséées', 'Je manque d\'énergie', 'J\'ai des convulsions'],
        'responses': ['Peut être lié à diverses conditions. Paludisme, fièvre, ou meme cholera. Consultez un médecin pour un diagnostic précis.']
    },
    {
        'tag': 'autre',
        'patterns': ['J\'ai un problème de santé', 'Je ne me sens pas bien'],
        'responses': ['Préciser exactement de quoi vous souffrez pour un diagnostic plus précis de votre état de santé.']
    }
]

# Créer le vextoriseur et le classificateur
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0,max_iter=1000)

# Prétraiter les données
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Entrainer le modèle
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x,y)

# Fonction python pour discuter avec le Chatbot
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Déploiement du chatbot
counter = 0

def main():
    global counter
    st.title('chatbot')
    st.write('Bienvenue sur le chatbot créer par ABOUEM qui conseille sur l’orientation decisionelle d’un patient pour les eventuelles possibilités de diagnostiques therapeutiques. Veuillez saisir un message et appuyer sur entrée pour démarrer la conversation.')

    counter += 1
    user_input = st.text_input('Vous:', key = f'user_input_ {counter}')

    if user_input:
        response = chatbot(user_input)
        st.text_area('chatbot:', value=response, height=100, max_chars=200, key=f'chatbot_response_{counter}')

        if response.lower() in ['Au revoir', 'bye']:
            st.write('Merci d\'avoir discuté avec moi. Passez une bonne Journée')
            st.stop()

if __name__ == '__main__':
    main()
