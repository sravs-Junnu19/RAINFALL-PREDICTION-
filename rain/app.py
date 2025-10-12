from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import pickle
import requests
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
MODEL_PATH = "model.pkl"

# ------------------ Model Setup ------------------
def create_and_save_dummy_model(path=MODEL_PATH):
    np.random.seed(42)
    N = 8000
    lats = np.random.uniform(-60, 60, N)
    lons = np.random.uniform(-180, 180, N)
    months = np.random.randint(1, 13, N)
    days = np.random.randint(1, 29, N)

    X, y = [], []
    for lat, lon, m, d in zip(lats, lons, months, days):
        temp = 30 - abs(lat)/3 + 5*np.sin(2*np.pi*(m-1)/12)
        humidity = 60 + 20*np.cos(2*np.pi*(m-1)/12) - (abs(lat)/90)*30
        wind = 5 + 2*np.sin(2*np.pi*d/30) + (abs(lon)/180)*2
        elevation = max(0, 500 - abs(lat)*5 + (lon % 90))
        X.append([temp, humidity, wind, elevation])
        rainfall = max(0, 0.6*humidity - 0.4*temp + 0.02*elevation + np.random.normal(0,8))/4
        y.append(rainfall)

    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Dummy model saved at {path}")

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        create_and_save_dummy_model(path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def make_features(lat, lon, dt):
    month = dt.month
    day = dt.day
    temp = 30 - abs(lat)/3 + 5*np.sin(2*np.pi*(month-1)/12)
    humidity = 60 + 20*np.cos(2*np.pi*(month-1)/12) - (abs(lat)/90)*30
    wind = 5 + 2*np.sin(2*np.pi*day/30) + (abs(lon)/180)*2
    elevation = max(0, 500 - abs(lat)*5 + (lon % 90))
    return [temp, humidity, wind, elevation]

def classify_rainfall(rainfall):
    if rainfall == 0:
        return 'No rain'
    elif rainfall <= 2.5:
        return 'Drizzle'
    elif rainfall <= 7.6:
        return 'Light rain'
    elif rainfall <= 50:
        return 'Moderate rain'
    else:
        return 'Heavy rain'

model = load_model()

# ------------------ Routes ------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        lat = float(data['lat'])
        lng = float(data['lng'])
        start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
    except Exception:
        return jsonify({'error': 'Invalid input'}), 400

    if start_date > end_date:
        return jsonify({'error': 'Start date cannot be after end date'}), 400

    # Reverse geocode
    location = "Unknown"
    try:
        resp = requests.get(f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}', headers={'User-Agent': 'RainfallPredictionApp'}, timeout=5)
        if resp.status_code == 200:
            location = resp.json().get('display_name', 'Unknown')
    except:
        pass

    delta = end_date - start_date
    predictions = []
    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        features = make_features(lat, lng, day)
        pred = float(model.predict([features])[0])
        pred = round(max(0.0, pred), 2)
        rainfall_type = classify_rainfall(pred)
        if pred > 10: weather, icon = 'rainy','🌧️'
        elif pred < 5: weather, icon = 'sunny','☀️'
        else: weather, icon = 'cloudy','☁️'
        predictions.append({'date': day.strftime('%Y-%m-%d'), 'rainfall': pred, 'rainfall_type': rainfall_type, 'weather': weather, 'icon': icon})

    wiki_link = f"https://en.wikipedia.org/wiki/{location.replace(' ', '_')}"
    session['results'] = {'predictions': predictions, 'location': location, 'wiki_link': wiki_link,
                          'start_date': start_date.strftime('%Y-%m-%d'), 'end_date': end_date.strftime('%Y-%m-%d')}
    return jsonify({'redirect': url_for('results')})

@app.route('/results')
def results():
    results = session.get('results')
    if not results:
        return redirect(url_for('home'))
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
