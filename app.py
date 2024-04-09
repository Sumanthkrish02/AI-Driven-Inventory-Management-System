from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import matplotlib
matplotlib.use('agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)

model=pickle.load(open('trained_model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Save the file to a desired location
    file.save('/Users/bharath/Downloads/Major_project/data_set/data')
    
    # Perform any necessary processing on the uploaded file
    
    return "File uploaded successfully!"



def recommend_near_expiry_products():
    # Read data from CSV file
    df = pd.read_csv("/Users/bharath/Downloads/Major_project/data_set/data")

    # Convert Expiration_Date to datetime
    df['Expiration_Date'] = pd.to_datetime(df['expiry_date'])

    # Filter out products with past expiration dates
    df = df[df['Expiration_Date'] >= pd.Timestamp.today()]

    # Calculate days until expiration
    df['Days_Until_Expiry'] = (df['Expiration_Date'] - pd.Timestamp.today()).dt.days

    # Identify products nearing expiration (e.g., within 7 days)
    near_expiry_threshold = 7
    near_expiry_products = df[df['Days_Until_Expiry'] <= near_expiry_threshold]

    # Remove duplicate products
    near_expiry_products = near_expiry_products.drop_duplicates(subset=['product_id'])

    # Convert DataFrame to list of dictionaries for easy rendering in HTML
    recommendations = near_expiry_products.to_dict(orient='records')

    return recommendations


def recommend_restock():
    # Read inventory data from CSV file
    inventory_df = pd.read_csv("/Users/bharath/Downloads/Major_project/data_set/data")

    # Check for low stock and suggest restocking
    low_stock_recommendations = []
    for index, row in inventory_df.iterrows():
        if row["quantity_stock"] <= 300:
            stock_difference = 301 - row["quantity_stock"]
            recommendation = {
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "stock_difference": max(stock_difference, 0)  # Ensure non-negative stock difference
            }
            low_stock_recommendations.append(recommendation)

    # Remove duplicate low stock recommendations
    seen_product_names = set()
    unique_low_stock_recommendations = []
    for recommendation in low_stock_recommendations:
        if recommendation["product_name"] not in seen_product_names:
            unique_low_stock_recommendations.append(recommendation)
            seen_product_names.add(recommendation["product_name"])

    # Get near expiry recommendations
    near_expiry_recommendations = recommend_near_expiry_products()

    # Return both low stock and near expiry recommendations
    return unique_low_stock_recommendations, near_expiry_recommendations

@app.route('/inventory')
def inventory():
    # Call both recommendation functions
    low_stock_recommendations, near_expiry_recommendations = recommend_restock()
    return render_template('inventory.html', restock_recommendations=low_stock_recommendations, near_expiry_recommendations=near_expiry_recommendations)



@app.route('/predict', methods=["GET", "POST"]) 
def predict():
    try:
        # Load the dataset
        df = pd.read_csv("/Users/bharath/Downloads/Major_project/data_set/data")

        # Fit the MinMaxScaler with the data
        scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize MinMaxScaler
        scaler.fit(df[['quantity_sold']])  # Fit the scaler with the 'quantity_sold' column data

        # Get the last three terms from the dataset
        last_three_terms = df['quantity_sold'].values[-3:]

        # Normalize the input data using the fitted scaler
        input_data = scaler.transform(last_three_terms.reshape(-1, 1))

        # Reshape the input data to match the model input shape
        input_data = input_data.reshape(1, 3, 1)

        # Make predictions
        predictions = model.predict(input_data)
        print(input_data)

        # Inverse transform the predictions
        predictions = scaler.inverse_transform(predictions)

        # Pass the prediction result to the template
        return render_template('prediction.html', prediction=predictions[0])

    except Exception as e:
        return render_template('error.html', error_message=str(e))


@app.route('/analytics',methods=["GET", "POST"])
def sales_analytics():
        # Load data from CSV file
        data = pd.read_csv("/Users/bharath/Downloads/Major_project/data_set/data")

        # Convert date column to datetime format
        data["date"] = pd.to_datetime(data["date_sale"], format='%d-%m-%Y') 
        # Calculate total sales
        total_sales = data["total_revenue"].sum()

        # Calculate average order value
        average_order_value = data["total_revenue"].mean()

        # Find top 5 selling products
        top_selling_products = data.nlargest(5, "quantity_stock")

        bottom_selling_products = data.nsmallest(5, "quantity_stock")

        return render_template('analytics.html', total_sales=total_sales,
                       average_order_value=average_order_value,
                       top_selling_products=top_selling_products,
                       bottom_selling_products=bottom_selling_products)

def calculate_sales_trend(data):
    monthly_sales = data.resample('M').sum()  # Resample data to monthly frequency and calculate total sales
    return monthly_sales

@app.route('/', methods=["GET", "POST"])
def sales_trend():
    # Load data
    data = pd.read_csv("/Users/bharath/Downloads/Major_project/data_set/data")
    # Calculate sales trend
    monthly_sales = calculate_sales_trend(data)
    print(monthly_sales)
    # Plot sales trend
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_sales.index, monthly_sales['total_revenue'], marker='o', linestyle='-')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/Users/bharath/Downloads/Major_project/static/sales_trend.png')  # Save the plot as a static image
    plt.close()

    # Render HTML template with the sales trend graph
    return render_template('analytics.html')

if __name__ == '__main__':
    app.run(debug=True)
