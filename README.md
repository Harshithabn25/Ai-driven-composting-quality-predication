# capstone-project
# Battery Life Prediction

This project is a **Streamlit web application** that predicts battery life using a trained **machine learning model**. The model is built using **RandomForestRegressor** and deployed using **Streamlit Cloud**.

## 📂 Project Structure
```
📁 project-folder/
│-- app.py               # Streamlit application script
│-- model.pkl            # Trained machine learning model
│-- requirements.txt     # List of dependencies
│-- pro10thfeb.ipynb     # Jupyter Notebook (model training & analysis)
│-- README.md            # Project documentation (this file)
```

---
## 🚀 Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### **2️⃣ Create a Virtual Environment (Optional but Recommended)**
```bash
python3 -m venv myenv
source myenv/bin/activate  # macOS/Linux
myenv\Scripts\activate     # Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Run the Streamlit App**
```bash
streamlit run app.py
```
The app will start and open in your browser at **http://localhost:8501**.

---
## 🛠 Model Training (Jupyter Notebook)
The model is trained in the **pro10thfeb.ipynb** notebook using **RandomForestRegressor**. If you need to retrain the model:
```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv('data.csv')
X = df.drop(columns=['battery_life'])
y = df['battery_life']

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
```

---
## 🌍 Deployment
### **1️⃣ Deploy on Streamlit Cloud**
1. Push your code to **GitHub**.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and log in.
3. Click **Deploy an app** and select your GitHub repo.
4. Set `app.py` as the entry point.
5. Click **Deploy** and wait a few minutes.
6. 🎉 Your app is live at **https://your-app.streamlit.app**!

---
## 📜 License
This project is **open-source** under the **MIT License**.
