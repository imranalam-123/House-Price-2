# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_explore_data(self, file_path):
        """Load dataset and perform exploratory data analysis"""
        try:
            # Load dataset
            self.df = pd.read_csv(file_path)
            st.success("📊 Dataset loaded successfully!")
            st.write(f"Dataset shape: {self.df.shape}")
            return self.df
            
        except FileNotFoundError:
            st.warning("❌ File not found. Creating sample data...")
            self._create_sample_data()
            return self.df
    
    def _create_sample_data(self):
        """Create sample housing data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'area': np.random.normal(1500, 500, n_samples).astype(int),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'stories': np.random.randint(1, 3, n_samples),
            'parking': np.random.randint(0, 3, n_samples),
            'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
            'year_built': np.random.randint(1980, 2023, n_samples),
        }
        
        self.df = pd.DataFrame(data)
        # Make price dependent on features
        self.df['price'] = (self.df['area'] * 200 + 
                           self.df['bedrooms'] * 50000 + 
                           self.df['bathrooms'] * 30000 +
                           np.random.normal(0, 50000, n_samples)).astype(int)
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        st.info("🔄 Preprocessing data...")
        
        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        # Handle categorical variables
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            self.df[col] = self.label_encoders[col].fit_transform(self.df[col])
        
        st.success("✅ Data preprocessing completed!")
        return self.df
    
    def train_models(self, test_size=0.2):
        """Train multiple models and compare performance"""
        st.info("🤖 Training machine learning models...")
        
        # Prepare features and target
        X = self.df.drop('price', axis=1)
        y = self.df['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            st.write(f"Training {name}...")
            
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            results[name] = {
                'model': model,
                'predictions': predictions,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        
        self.results = results
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        
        st.success("✅ Model training completed!")
        return results
    
    def predict_new_house(self, house_features):
        """Predict price for a new house"""
        best_model_name = max(self.results.items(), 
                            key=lambda x: x[1]['r2'])[0]
        best_model = self.results[best_model_name]['model']
        
        # Create feature DataFrame
        feature_df = pd.DataFrame([house_features])
        
        # Preprocess the input features
        for col in feature_df.columns:
            if col in self.label_encoders:
                if house_features[col] in self.label_encoders[col].classes_:
                    feature_df[col] = self.label_encoders[col].transform([house_features[col]])[0]
                else:
                    # Default to first category if unknown
                    feature_df[col] = 0
        
        # Ensure all columns are present and in correct order
        for col in self.X_train.columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        feature_df = feature_df[self.X_train.columns]
        
        # Make prediction
        if best_model_name == 'Linear Regression':
            feature_scaled = self.scaler.transform(feature_df)
            prediction = best_model.predict(feature_scaled)[0]
        else:
            prediction = best_model.predict(feature_df)[0]
        
        return prediction, best_model_name

# Streamlit App
def main():
    st.set_page_config(page_title="🏡 House Price Predictor", layout="wide")
    
    st.title("🏡 House Price Prediction using Machine Learning")
    st.markdown("Predict house prices based on various features using multiple ML models")
    
    # Initialize predictor
    @st.cache_resource
    def load_predictor():
        predictor = HousePricePredictor()
        predictor.load_and_explore_data('house_data.csv')
        predictor.preprocess_data()
        predictor.train_models()
        return predictor
    
    # Load predictor
    with st.spinner("Loading and training models..."):
        predictor = load_predictor()
    
    # Sidebar for input
    st.sidebar.header("📝 House Features")
    
    area = st.sidebar.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1500)
    bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 4, 2)
    stories = st.sidebar.slider("Stories", 1, 3, 2)
    parking = st.sidebar.slider("Parking Spaces", 0, 3, 1)
    location = st.sidebar.selectbox("Location", ["Urban", "Suburban", "Rural"])
    year_built = st.sidebar.number_input("Year Built", min_value=1950, max_value=2023, value=2010)
    
    if st.sidebar.button("Predict Price"):
        house_features = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'parking': parking,
            'location': location,
            'year_built': year_built
        }
        
        predicted_price, model_used = predictor.predict_new_house(house_features)
        
        st.success(f"💰 Predicted House Price: **${predicted_price:,.2f}**")
        st.info(f"🎯 Model used: **{model_used}**")
        
        # Show feature impact
        st.subheader("📊 Feature Impact on Price")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Area", f"{area} sq ft")
            st.metric("Bedrooms", bedrooms)
            
        with col2:
            st.metric("Bathrooms", bathrooms)
            st.metric("Stories", stories)
            
        with col3:
            st.metric("Parking", parking)
            st.metric("Location", location)
    
    # Main content - Model Performance
    st.subheader("📈 Model Performance Comparison")
    
    if hasattr(predictor, 'results'):
        performance_data = []
        for name, result in predictor.results.items():
            performance_data.append({
                'Model': name,
                'R² Score': f"{result['r2']:.4f}",
                'RMSE': f"${result['rmse']:,.2f}",
                'MAE': f"${result['mae']:,.2f}"
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
    
    # Dataset preview
    with st.expander("📊 View Dataset Preview"):
        st.dataframe(predictor.df.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dataset Info:**")
            st.write(f"Shape: {predictor.df.shape}")
            st.write(f"Features: {list(predictor.df.columns)}")
        
        with col2:
            st.write("**Price Statistics:**")
            st.write(f"Average Price: ${predictor.df['price'].mean():,.2f}")
            st.write(f"Minimum Price: ${predictor.df['price'].min():,.2f}")
            st.write(f"Maximum Price: ${predictor.df['price'].max():,.2f}")

if __name__ == "__main__":
    main()