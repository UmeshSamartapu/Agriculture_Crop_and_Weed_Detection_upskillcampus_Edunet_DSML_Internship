# 🚦 Traffic Pattern Forecasting Report for Smart City Initiative

This project focuses on **predicting vehicle traffic volume** at city junctions using a combination of **feature engineering** and **machine learning models  XGBoost Regressor.** 
It includes **data preprocessing, feature selection, model training, evaluation, and model saving.**


# 📂 Project Structure
- **Train.csv** - Training data with traffic volume at different junctions.

- **Test.csv** - Test data where traffic needs to be predicted.

- **vehicle_model.pkl** - Trained model file saved for future use.

- **submission.csv** – Final predictions for the test dataset.

# 🔧 Features & Techniques Used

- **Feature Engineering:**

- Extracted Hour, DayOfWeek, IsWeekend, and Month from DateTime.
  
- Converted DateTime into UNIX time for feature importance analysis.

- **Feature Importance:** Used ExtraTreesClassifier to identify key features impacting traffic volume.

- **Modeling:** Trained an XGBoost Regressor for accurate traffic volume prediction.

- **Evaluation:** Model performance measured using Mean Absolute Error (MAE).

- **Visualization:** Feature importance bar chart. Histogram of vehicle counts.Traffic trends across junctions.

- **Model Deployment:** Saved trained model using Joblib.


# 📊 Data Preprocessing

- **DateTime Parsing:** Converted string DateTime fields into Python datetime objects.

- **Feature Extraction:** Extracted: Hour, DayOfWeek, IsWeekend, Month.

- **Feature Engineering:** IsWeekend derived based on DayOfWeek.DateTime transformed to UNIX time for feature analysis.

# 🧠 Model Training

- **Feature Selection:** Used key features: Junction, Hour, DayOfWeek, IsWeekend, Month.

- **Model:** XGBoost Regressor trained on extracted features.

- **Evaluation Metrics:** Mean Absolute Error (MAE) was computed for validation set.


Example Output:
```bash
Mean Absolute Error (MAE) on validation set: 8.31
``

# 📈 Visualizations

- **Feature Importance Bar Chart** – Top features influencing traffic.

- **Histogram of Vehicle Counts** – Distribution of traffic volume.

- **Traffic Volume Trends by Junctions** – Line plot combining training and prediction data.

# 💾 Saving the Model
After training, the XGBoost model is saved as:
```bash
import joblib
joblib.dump(model, "traffic_forecasting_model.pkl")
```

# 📦 How the Prediction Works

1.Preprocess the test dataset using the same transformations.

2.Load the saved model traffic_forecasting_model.pkl.

3.Predict the traffic volume on test data.

4.Save the final predictions into submission.csv.

5.Visualize the combined traffic volume.

# 🛠 Requirements
- Python 3.8+

- Libraries:
```bash
- pandas

- numpy

- matplotlib

- scikit-learn

- xgboost

- joblib
```

# 📜 Important Notes

- Always apply the same feature extraction steps to both training and test datasets.

- Make sure DateTime parsing is correct (use dayfirst=True).

- Set a fixed random_state for reproducible results.

- Fine-tune XGBoost hyperparameters for further performance improvement.



# ✨ Acknowledgements
Special thanks to:

- Scikit-learn developers for easy-to-use ML APIs.

- Kaggle and open datasets for real-world traffic data inspiration.

## 📫 Let's Connect

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/umeshsamartapu/)
[![Twitter](https://img.shields.io/badge/-Twitter-1DA1F2?style=flat-square&logo=twitter&logoColor=white)](https://x.com/umeshsamartapu)
[![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:umeshsamartapu@gmail.com)
[![Instagram](https://img.shields.io/badge/-Instagram-E4405F?style=flat-square&logo=instagram&logoColor=white)](https://www.instagram.com/umeshsamartapu/)
[![Buy Me a Coffee](https://img.shields.io/badge/-Buy%20Me%20a%20Coffee-FBAD19?style=flat-square&logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/umeshsamartapu)

---

🔥 Always exploring new technologies and solving real-world problems with code!
