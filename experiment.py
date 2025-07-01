import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

data = pd.read_csv('Churn_Modelling.csv')

# Preprocess the data
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical variables
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# One hot encoding geographical column
from sklearn.preprocessing import OneHotEncoder
onehot_encoder_geo = OneHotEncoder()
geo_encoder = onehot_encoder_geo.fit_transform(data[['Geography']])
geo_encoder_df = pd.DataFrame(geo_encoder.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
data = pd.concat([data.drop('Geography', axis=1), geo_encoder_df], axis = 1 )

# Save the encoders and scaler
with open('label_encoder.pkl', 'wb') as file:
  pickle.dump(label_encoder, file)

with open('onehot_encoder_geo.pkl', 'wb') as file:
  pickle.dump(onehot_encoder_geo, file)
  
# Divide the dataset into independent and dependent features
X = data.drop('Exited' , axis=1)
y = data['Exited']

# Split the data in training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Scale these features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open ('scaler.pkl', 'wb') as file:
  pickle.dump(scaler, file)
  
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
import datetime

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'), # Hidden layer 1 connected with input layer
    Dense(32, activation='relu'), # Hidden layer 2
    Dense(1, activation='sigmoid') # output layer
])


# Compile the model
import tensorflow
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
loss = tensorflow.keras.losses.BinaryCrossentropy()
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

# Set up the Tensorboard
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

import tensorflow
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
tensorflow_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Setup Early Stopping
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train,y_train,validation_data = (X_test, y_test), epochs = 100,
    callbacks = [tensorflow_callback, early_stopping_callback]
)

model.save('model.keras')

# Load the trained model, scaler pickle, onehot
model = load_model('model.keras')

# Load the encoder and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
  label_encoder_geo = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
  label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
  scaler = pickle.load(file)


# Example input data
# Example input data
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1 ,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}

input_df = pd.DataFrame([input_data])

# Encode 'Gender'
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

# Onehot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform(input_df[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Concatenate and drop 'Geography'
input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)

# Scale features
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)

if prediction > 0.5:
  print('The customer is likely to churn')
else:
  print('The customer is not likely to churn')
