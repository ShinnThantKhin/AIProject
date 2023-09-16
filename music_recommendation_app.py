import streamlit as st
import pickle  
import pandas as pd  
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

with open('genre_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

music_features = pd.read_csv('fma_metadata/features.csv')
music_features = music_features.apply(pd.to_numeric, errors='coerce')
music_features = music_features.fillna(music_features.mean())

genre_mapping = {
    0: 'Avant-Garde',
    1: 'International',
    2: 'Blues',
    3: 'Jazz',
    4: 'Classical',
    5: 'Novelty',
    6: 'Comedy',
    7: 'Old-Time / Historic',
    8: 'Country',
    9: 'Pop',
    10: 'Disco',
    11: 'Rock',
    12: 'Easy Listening',
    13: 'Soul-RnB',
    14: 'Electronic',
    15: 'Sound Effects',
    16: 'Folk',
    17: 'Soundtrack',
    18: 'Funk',
    19: 'Spoken',
    20: 'Hip-Hop',
    21: 'Audio Collage',
    22: 'Punk',
    23: 'Post-Rock',
    24: 'Lo-Fi',
    25: 'Field Recordings',
    26: 'Metal',
    27: 'Noise',
    28: 'Psych-Folk',
    29: 'Krautrock',
    30: 'Jazz: Vocal',
    31: 'Experimental',
    32: 'Electroacoustic',
    33: 'Ambient Electronic',
    34: 'Radio Art',
    35: 'Loud-Rock',
    36: 'Latin America',
    37: 'Drone',
    38: 'Free-Folk',
    39: 'Noise-Rock',
    40: 'Psych-Rock',
    41: 'Bluegrass',
    42: 'Electro-Punk',
    43: 'Radio',
    44: 'Indie-Rock',
    45: 'Industrial',
    46: 'No Wave',
    47: 'Free-Jazz',
    48: 'Experimental Pop',
    49: 'French',
    50: 'Reggae - Dub',
    51: 'Afrobeat',
    52: 'Nerdcore',
    53: 'Garage',
    54: 'Indian',
    55: 'New Wave',
    56: 'Post-Punk',
    57: 'Sludge',
    58: 'African',
    59: 'Freak-Folk',
    60: 'Jazz: Out',
    61: 'Progressive',
    62: 'Alternative Hip-Hop',
    63: 'Death-Metal',
    64: 'Middle East',
    65: 'Singer-Songwriter',
    66: 'Ambient',
    67: 'Hardcore',
    68: 'Power-Pop',
    69: 'Space-Rock',
    70: 'Polka',
    71: 'Balkan',
    72: 'Unclassifiable',
    73: 'Europe',
    74: 'Americana',
    75: 'Spoken Weird',
    76: 'Interview',
    77: 'Black-Metal',
    78: 'Rockabilly',
    79: 'Easy Listening: Vocal',
    80: 'Brazilian',
    81: 'Asia-Far East',
    82: 'N. Indian Traditional',
    83: 'South Indian Traditional',
    84: 'Bollywood',
    85: 'Pacific',
    86: 'Celtic',
    87: 'Be-Bop',
    88: 'Big Band/Swing',
    89: 'British Folk',
    90: 'Techno',
    91: 'House',
    92: 'Glitch',
    93: 'Minimal Electronic',
    94: 'Breakcore - Hard',
    95: 'Sound Poetry',
    96: '20th Century Classical',
    97: 'Poetry',
    98: 'Talk Radio',
    99: 'North African',
    100: 'Sound Collage',
    101: 'Flamenco',
    102: 'IDM',
    103: 'Chiptune',
    104: 'Musique Concrete',
    105: 'Improv',
    106: 'New Age',
    107: 'Trip-Hop',
    108: 'Dance',
    109: 'Chip Music',
    110: 'Lounge',
    111: 'Goth',
    112: 'Composed Music',
    113: 'Drum & Bass',
    114: 'Shoegaze',
    115: 'Kid-Friendly',
    116: 'Thrash',
    117: 'Synth Pop',
    118: 'Banter',
    119: 'Deep Funk',
    120: 'Spoken Word',
    121: 'Chill-out',
    122: 'Bigbeat',
    123: 'Surf',
    124: 'Radio Theater',
    125: 'Grindcore',
    126: 'Rock Opera',
    127: 'Opera',
    128: 'Chamber Music',
    129: 'Choral Music',
    130: 'Symphony',
    131: 'Minimalism',
    132: 'Musical Theater',
    133: 'Dubstep',
    134: 'Skweee',
    135: 'Western Swing',
    136: 'Downtempo',
    137: 'Cumbia',
    138: 'Latin',
    139: 'Sound Art',
    140: 'Romany (Gypsy)',
    141: 'Compilation',
    142: 'Rap',
    143: 'Breakbeat',
    144: 'Gospel',
    145: 'Abstract Hip-Hop',
    146: 'Reggae - Dancehall',
    147: 'Spanish',
    148: 'Country & Western',
    149: 'Contemporary Classical',
    150: 'Wonky',
    151: 'Jungle',
    152: 'Klezmer',
    153: 'Holiday',
    154: 'Salsa',
    155: 'Nu-Jazz',
    156: 'Hip-Hop Beats',
    157: 'Modern Jazz',
    158: 'Turkish',
    159: 'Tango',
    160: 'Fado',
    161: 'Christmas',
    162: 'Instrumental'
}


# Building Streamlit app
st.title('Music Recommendation App')

# User inputs
tempo = st.slider('Select Tempo:', 0.0, 250.0, 120.0)
rhythm = st.radio('Select Rhythm:', ['Fast', 'Medium', 'Slow'])
loudness = st.slider('Select Loudness:', -60.0, 0.0, -30.0)
energy = st.slider('Select Energy:', 0.0, 1.0, 0.5)
danceability = st.slider('Select Danceability:', 0.0, 1.0, 0.5)

# Encode rhythm
rhythm_encoder = LabelEncoder()
rhythm_encoded = rhythm_encoder.fit_transform([rhythm])

# Create a DataFrame for user input
user_input = pd.DataFrame({
    'tempo': [tempo],
    'rhythm_encoded': rhythm_encoded,
    'loudness': [loudness],
    'energy': [energy],
    'danceability': [danceability]
})

# Perform PCA on music_features
pca = PCA(n_components=60)
pca_components = pca.fit_transform(music_features)

# Concatenate PCA components with user input
user_input_with_pca = pd.concat([user_input, pd.DataFrame(pca_components, columns=[f'PCA_{n}' for n in range(1, 61)])], axis=1)

# Make predictions based on user inputs
if st.button('Get Recommendations'):
    # Use the trained model for predictions
    prediction = model.predict(user_input_with_pca)[0]
    predicted_genre = genre_mapping.get(prediction, 'Unknown Genre')

    # Display Recommendations
    st.subheader('Recommendations:')
    st.write(f'Predicted Genre: {predicted_genre}')
