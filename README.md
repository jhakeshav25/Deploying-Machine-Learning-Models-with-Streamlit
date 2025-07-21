# 🌸 Iris Flower Prediction App

![Banner](https://raw.githubusercontent.com/jhakeshav25/Deploying-Machine-Learning-Models-with-Streamlit/main/assets/iris_banner.png)

A beautiful and intuitive web app built using **Streamlit** to predict the species of Iris flowers using a trained **Random Forest** classifier.

---

## 📊 Features

- 📥 Input: Sepal & Petal dimensions via sliders
- 🧠 Prediction: Iris-setosa, Iris-versicolor, Iris-virginica
- 📈 Visualization: Probability bar chart
- 💡 Lightweight & interactive UI

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/jhakeshav25/Deploying-Machine-Learning-Models-with-Streamlit.git
cd Deploying-Machine-Learning-Models-with-Streamlit
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🌐 Deploy Online

Deploy easily using **[Streamlit Cloud](https://streamlit.io/cloud)**. Just upload your code and `requirements.txt`, and it's live in minutes.

---

## 📂 Project Structure

| File                | Description                                  |
|---------------------|----------------------------------------------|
| `app.py`            | Main Streamlit web app file                  |
| `requirements.txt`  | List of required Python packages             |
| `assets/`           | Contains banner image and other visuals      |
| `README.md`         | Project documentation                        |

---

## 🔍 Model Overview

This app uses the **Iris dataset** and a trained **Random Forest Classifier** to make predictions based on 4 inputs:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The output is a predicted species and the corresponding class probabilities.

---

## 👀 Preview

| 🎯 Prediction | 📊 Visualization |
|---------------|------------------|
| ![UI](https://raw.githubusercontent.com/jhakeshav25/Deploying-Machine-Learning-Models-with-Streamlit/main/assets/iris_ui.png) | ![Chart](https://raw.githubusercontent.com/jhakeshav25/Deploying-Machine-Learning-Models-with-Streamlit/main/assets/iris_chart.png) |

---

## 🙋‍♂️ Author

Made with ❤️ by [**Keshav Kumar Jha**](https://github.com/jhakeshav25)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
