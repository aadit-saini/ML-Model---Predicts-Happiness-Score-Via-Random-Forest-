README.md
# City Happiness Score Prediction using Random Forest

A machine learning project that predicts city happiness scores based on various urban lifestyle and environmental factors using Random Forest Regressor.

## Overview

This project analyzes how different city characteristics impact overall happiness scores. The model uses machine learning to understand the relationship between urban features and citizen happiness levels.

## Dataset

The dataset (`city_lifestyle_dataset.csv`) contains the following features:
- **Population Density**: Number of people per square kilometer
- **Average Income**: Mean income of city residents
- **Internet Penetration**: Percentage of population with internet access
- **Average Rent**: Mean monthly rent costs
- **Air Quality Index**: Measure of air pollution levels
- **Green Space Ratio**: Percentage of green/park areas in the city
- **Public Transport Score**: Quality rating of public transportation system
- **Happiness Score**: Target variable (overall happiness rating)

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas
scikit-learn
seaborn
matplotlib
```

## Project Structure
```
happiness-prediction-model/
├── RandomForestToPredictHappiness.py    # Main ML script
├── city_lifestyle_dataset.csv           # Dataset
├── README.md                             # Project documentation
└── requirements.txt                      # Dependencies
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/happiness-prediction-model.git
cd happiness-prediction-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the model:
```bash
python RandomForestToPredictHappiness.py
```

## Model Details

- **Algorithm**: Random Forest Regressor
- **Number of Estimators (Trees)**: 100
- **Train-Test Split**: 80% training, 20% testing
- **Random State**: 42 (for reproducibility)
- **Feature Scaling**: StandardScaler applied to normalize all features
- **Evaluation Metric**: R² Score (Coefficient of Determination)

## Code Workflow

1. **Data Loading**: Load the city lifestyle dataset
2. **Data Preparation**: 
   - Separate target variable (happiness_score) from features
   - Split data into training and testing sets
3. **Feature Scaling**: Standardize features using StandardScaler
4. **Model Training**: Train Random Forest Regressor on training data
5. **Prediction**: Generate predictions on test data
6. **Evaluation**: Calculate R² score to assess model performance

## Results

The model's performance:
- **R² Score**: [Run the code to get your score - typically between 0 and 1]
  - Values closer to 1 indicate better predictive performance
  - Shows how well the model explains variance in happiness scores

## Key Features Used

The model considers 7 urban characteristics:
1. Population density
2. Average income
3. Internet penetration
4. Average rent
5. Air quality index
6. Green space ratio
7. Public transport score

## Future Improvements

- Add Mean Absolute Error (MAE) and Mean Squared Error (MSE) metrics
- Implement cross-validation for more robust evaluation
- Visualize feature importance to identify key happiness drivers
- Add data visualizations (correlation heatmap, feature distributions)
- Hyperparameter tuning using GridSearchCV
- Try other algorithms (XGBoost, Gradient Boosting, Neural Networks)
- Add more features like crime rate, education quality, healthcare access

## Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and preprocessing
- **matplotlib & seaborn**: Data visualization (imported but not yet used)

## Author

Aadit Saini

## License

This project is open source and available under the MIT License.