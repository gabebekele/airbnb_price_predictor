# NYC Airbnb Price Prediction Using AutoGluon

## Project Overview

This project uses machine learning to predict nightly Airbnb listing prices in New York City. Using the NYC Airbnb 2019 dataset, I trained and evaluated multiple regression models through AutoGluon and compared their performance against a traditional Linear Regression baseline.

In addition to predicting prices, the project explores which property characteristics have the greatest influence on pricing, how prices vary across NYC boroughs and room types, and whether advertised amenities are associated with higher nightly rates.

## Dataset

**Source:** [New York City Airbnb Open Data — Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

The original dataset contains 48,895 listings, with information including:

- Property location (borough, neighborhood, latitude, longitude)
- Room type (entire home, private room, shared room)
- Nightly price and minimum stay requirements
- Number of reviews and review frequency
- Availability throughout the year
- Listing names and host information

Data preprocessing included handling missing values, filtering unrealistic or unsuitable observations, scaling numerical features, and preparing categorical variables for modeling.

The cleaned dataset was divided into 80% training and 20% testing data using a fixed random seed.

## Technologies Used

- **Python:** Pandas, NumPy, Scikit-learn
- **Machine Learning:** AutoGluon, Linear Regression, LightGBM, CatBoost, XGBoost, Random Forest, Extra Trees, PyTorch, FastAI
- **Visualization:** Matplotlib, Seaborn
- **Development:** Jupyter Notebook, VS Code, Conda

## Methodology

### 1. Data Preprocessing

The dataset was cleaned to address missing values and remove listings outside the selected price and availability ranges.

Numerical variables were scaled, while categorical variables were handled through AutoGluon's automatic preprocessing or one-hot encoding for the Linear Regression baseline.

An 80/20 training and testing split was used to evaluate model performance on unseen listings.

### 2. AutoGluon Model Training

AutoGluon was used to train multiple regression algorithms and construct weighted ensembles.

Model performance was evaluated using:

- **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual prices.
- **Root Mean Squared Error (RMSE):** Measures prediction error while penalizing larger mistakes.
- **R² Score:** Measures how much variation in nightly prices is explained by the model.

### 3. Controlled Model Comparison

A Linear Regression baseline was developed to compare traditional regression against AutoGluon's automated machine learning approach.

To make the comparison more consistent, a second AutoGluon model was trained using the same original input features as Linear Regression, excluding listing names and identifier-related columns.

All three models were evaluated on the same held-out test observations.

## Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Original AutoGluon Ensemble | **$42.70** | **$84.29** | **0.55** |
| Controlled AutoGluon Ensemble | $46.28 | $89.43 | 0.49 |
| Linear Regression | $57.60 | $99.74 | 0.36 |

**Key result:** The controlled AutoGluon ensemble reduced MAE by approximately **19.7%** compared with Linear Regression when using the same input features.

The original AutoGluon model achieved the lowest prediction error, suggesting that additional listing information, including property names, provides useful predictive signals.

## Feature Importance

Permutation feature importance was used to identify which listing characteristics contributed most to prediction accuracy.

Key findings included:

- **Room type** was the most important feature for both AutoGluon and Linear Regression.
- **Location** was highly influential, particularly neighborhood categories and geographic coordinates.
- **Minimum nights** and other numerical features contributed more strongly to AutoGluon's predictive performance.
- **Listing names** provided additional useful information for the original AutoGluon model.

These findings suggest that Airbnb pricing depends on a combination of property characteristics, geographic location, and relationships between features that may not be fully captured by a simple linear model.

## Exploratory Data Analysis

### Geographic Pricing Differences

Nightly prices varied across NYC boroughs. Manhattan generally had the highest prices, while listings in the Bronx, Queens, and Staten Island tended to be less expensive.

### Room Type Pricing

Entire homes and apartments generally commanded higher nightly rates than private or shared rooms.

The model also produced larger absolute prediction errors for entire homes, partly reflecting their higher prices.

### Advertised Amenities

Because the dataset does not provide a dedicated amenities column, listing names were searched for keywords associated with selected property features.

Several advertised features were associated with higher median nightly prices:

| Advertised Feature | Median Price Difference |
|---|---:|
| Pool | +$116 |
| Rooftop | +$97 |
| Laundry | +$91 |
| Hot Tub | +$66 |
| Garden | +$35 |
| Park | +$25 |

These results represent associations rather than causal effects. Keyword mentions do not guarantee the presence of an amenity, and certain categories contained relatively few listings.

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/gabebekele/airbnb_Predictor
cd airbnb_Predictor
```

### 2. Install Dependencies

If using the included Conda environment:

```bash
conda env create -f environment.yml
conda activate airbnb-prediction
```

You could also install dependencies using:

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download `AB_NYC_2019.csv` from the Kaggle dataset linked above and place it in the location expected by the notebook (inside the data folder).

### 4. Run the Notebook

Open `code.ipynb` in Jupyter Notebook or VS Code and run all cells in order.

AutoGluon will train and save its models locally in the `AutogluonModels/` directory.

**Note:** Model training can be computationally intensive. Results may vary slightly across runs because of randomness in training and differences between software environments. The metrics reported above reflect the saved experiment.

## Limitations

- The dataset represents Airbnb listings from 2019 and may not reflect current NYC rental market conditions.
- Some records were excluded during preprocessing, which may affect how representative the final dataset is.
- Advertised amenities were inferred from listing titles rather than verified property details.
- Several model configurations were evaluated using the same test data, creating some risk of optimistic model selection.
- Although the controlled comparison used matching original input features, AutoGluon and Linear Regression used different preprocessing and modeling approaches.

## Conclusion

This project demonstrates how automated machine learning can improve price prediction compared with traditional Linear Regression. AutoGluon's weighted ensemble achieved stronger predictive performance, while the controlled experiment showed that its advantage was not solely due to access to additional listing features.The analysis also highlighted the importance of room type, location, and property information in determining NYC Airbnb nightly prices.

Potential future improvements include incorporating verified property amenities, exploring additional feature engineering techniques, and evaluating model performance on newer Airbnb datasets.