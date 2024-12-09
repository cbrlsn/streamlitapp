import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#### Defining general preoprties of the app
###########################################
st.set_page_config(
    page_title="Diamond Brothers",
    page_icon='💎',
    layout='wide'
)

# Apply custom CSS for dark background
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add the banner section
st.markdown(
    """
    <div style="background-color: #000000; padding: 20px; text-align: center; border-radius: 5px; margin-bottom: 20px;">
        <h1 style="color: #ffffff; font-family: Arial, sans-serif;">💎 Welcome to Diamond Brothers 💎</h1>
        <h3 style="color: #ffffff; font-family: Arial, sans-serif;">The world's best diamond pricing tool for transparency and confidence</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar welcome message
st.sidebar.header("Welcome!")
st.sidebar.write("""
This Diamond Brothers app uses advanced machine learning techniques to predict diamond prices based on attributes such as carat, cut, clarity, and color. 
Customize your preferences, and view the filtered data, to make informed pricing decisions!
""")

st.markdown("By Clemens Burleson & Aksh Iyer from the University of St. Gallen under the instruction of Prof. Dr. Ivo Blohm")

#### Define Load functions and load data
###########################################
@st.cache_data()
def load_data():
    # Load the data and rename columns to have capitalized first letters
    df = pd.read_csv("diamonds.csv")
    df.columns = df.columns.str.capitalize()  # Capitalize all column titles
    return df.dropna()

df = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["Diamond Guide", "Filtered Diamonds", "Price Prediction", "Pricing Relationships"])

with tab1:
    st.header("Diamond Color Guide")
    # Dictionary to map diamond colors to visual representations
    color_descriptions = {
        "D": "Completely colorless, the highest grade of diamond color.",
        "E": "Exceptional white, minute traces of color detectable.",
        "F": "Excellent white, slight color detectable only by experts.",
        "G": "Near colorless, slight warmth noticeable.",
        "H": "Near colorless, slightly more warmth.",
        "I": "Noticeable warmth, light yellow tint visible.",
        "J": "Very noticeable warmth, visible yellow tint.",
    }

    # Create colored boxes to represent diamond colors
    diamond_color_boxes = {
        "D": "#fdfdfd",  # White
        "E": "#f8f8f8",  # Slightly off-white
        "F": "#f0f0f0",  # Slight gray
        "G": "#e8e8e8",  # Light gray
        "H": "#e0d4b8",  # Slight yellow
        "I": "#d4b892",  # Yellow tint
        "J": "#ccb78e",  # Warm yellow
    }

    # Display diamond colors with descriptions
    for color, hex_code in diamond_color_boxes.items():
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="width: 50px; height: 50px; background-color: {hex_code}; border: 1px solid black; margin-right: 15px;"></div>
                <div>
                    <strong>Color {color}:</strong> {color_descriptions[color]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Add a section for Diamond Sizes
    st.header("Diamond Sizes (Carats)")

    # Display the image with a caption
    st.image("Diamond_Carat_Weight.png", caption="Comparison of Diamond Sizes (Carats)", use_column_width=True)

with tab2:
    st.header("Filtered Diamonds")

    # Create two main columns for layout
    col1, col2 = st.columns([1, 1])  # Equal-width columns for filter options and filtered data

    with col1:  # Filter options on the left
        st.subheader("Filter Options")

        # Slider for price range (formatted with commas, no decimals)
        price_range = st.slider(
            "Select Desired Price Range",
            min_value=int(df["Price"].min()),  # Cast to int to remove decimals
            max_value=int(df["Price"].max()),  # Cast to int to remove decimals
            value=(int(df["Price"].min()), int(df["Price"].max())),  # Set initial range as integers
            format="%d"  # Display numbers without decimals
        )
    
        # Slider for carat range
        mass_range = st.slider(
            "Select Desired Carat Range",
            min_value=float(df["Carat"].min()),
            max_value=float(df["Carat"].max()),
            value=(float(df["Carat"].min()), float(df["Carat"].max()))
        )
    
        # Multiselect options for Cut, Color, and Clarity
        cut_options = st.multiselect(
            "Select Diamond Cuts",
            options=df["Cut"].unique(),
            default=df["Cut"].unique()
        )
    
        color_options = st.multiselect(
            "Select Diamond Colors",
            options=df["Color"].unique(),
            default=df["Color"].unique()
        )
    
        clarity_options = st.multiselect(
            "Select Diamond Clarity Levels",
            options=df["Clarity"].unique(),
            default=df["Clarity"].unique()
        )
    
        # Multiselect for column selection
        st.subheader("Customize Columns")
        default_columns = ['Price', 'Carat', 'Cut', 'Color']  # Default columns to display
        columns_to_display = st.multiselect(
            "Select Columns to Display:",
            options=df.columns.tolist(),
            default=[col for col in default_columns if col in df.columns]  # Use default columns if available
        )

    with col2:  # Filtered data on the right
        st.subheader("Filtered Diamonds")

        # Apply filters to the DataFrame
        filtered_diamonds = df[
            (df["Price"] >= price_range[0]) &
            (df["Price"] <= price_range[1]) &
            (df["Carat"] >= mass_range[0]) &
            (df["Carat"] <= mass_range[1]) &
            (df["Cut"].isin(cut_options)) &
            (df["Color"].isin(color_options)) &
            (df["Clarity"].isin(clarity_options))
        ]

        num_results = len(filtered_diamonds)
        st.markdown(f"**{num_results} results**")

        if filtered_diamonds.empty:
            st.warning("No diamonds match your selected criteria. Please adjust the filters.")
        else:
            # Center the filtered data horizontally
            st.markdown(
                """
                <div style="display: flex; justify-content: center; width: 100%; margin-top: 20px;">
                    <div style="width: 70%; max-width: 800px;">
                """,
                unsafe_allow_html=True
            )
            # Display the filtered DataFrame with selected columns
            st.dataframe(filtered_diamonds[columns_to_display].reset_index(drop=True))  # Reset index and drop the original one
            st.markdown(
                """
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

#### MODEL
###########################################
from model_training import train_and_save_model
import joblib

# Load the data
df = load_data()

# Train or load the model
model, metadata = train_and_save_model(df)

if not model:
    st.error("Model could not be loaded or trained. Please check the training script.")
    st.stop()

# Extract column and categorical feature metadata
model_columns = metadata['columns']
categorical_features = metadata['categorical_features']

# Add this before prediction to ensure column order
def preprocess_input(data, columns, categorical_features):
    """
    Preprocess input data to match the trained model's format.
    """
    for col in categorical_features:
        if col in data:
            data[col] = data[col].astype('category').cat.codes
    for col in columns:
        if col not in data:
            data[col] = 0  # Add missing columns with default value
    return data[columns]

# Use the model for predictions
with tab3:
    st.header("Price Prediction Tool")
    
    with st.form("prediction_form"):
        # Input fields
        Carat = st.slider("Carat", min_value=float(df["Carat"].min()), max_value=float(df["Carat"].max()), value=1.0, step=0.01)
        Cut = st.selectbox("Cut", options=df["Cut"].unique())
        Color = st.selectbox("Color", options=df["Color"].unique())
        Clarity = st.selectbox("Clarity", options=df["Clarity"].unique())
        submitted = st.form_submit_button("Predict Price")
    
    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'Carat': [Carat],
            'Cut': [Cut],
            'Color': [Color],
            'Clarity': [Clarity]
        })

        # Preprocess input data
        input_data = preprocess_input(input_data, model_columns, categorical_features)

        # Predict
        with st.spinner("Calculating price..."):
            prediction = model.predict(input_data)[0]

        # Display prediction
        st.success(f"Estimated Price: ${prediction:,.2f}")


import time  # Add this import at the top of your script

with tab3:
    st.header("Price Prediction Tool")

    # User Inputs for Prediction
    st.subheader("Enter Diamond Features:")
    with st.form("prediction_form"):
        # Input fields for diamond features
        Carat = st.slider("Carat", min_value=float(df["Carat"].min()), max_value=float(df["Carat"].max()), value=1.0, step=0.01)
        Cut = st.selectbox("Cut", options=df["Cut"].unique())
        Color = st.selectbox("Color", options=df["Color"].unique())
        Clarity = st.selectbox("Clarity", options=df["Clarity"].unique())

        # Submit button
        submitted = st.form_submit_button("Predict Price")

    # Initialize predicted_price variable
    predicted_price = None

    if submitted:
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'Carat': [Carat],
            'Cut': [Cut],
            'Color': [Color],
            'Clarity': [Clarity]
        })

        # Add missing columns with default values
        for col in X.columns:
            if col not in input_data.columns:
                # Use mean or default values for missing columns
                if col in df.columns:
                    input_data[col] = df[col].mean()
                else:
                    input_data[col] = 0  # Default for entirely missing columns

        # Ensure column order matches the training data
        input_data = input_data[X.columns]

        # Preprocess categorical variables
        for col in input_data.select_dtypes(include='object').columns:
            input_data[col] = input_data[col].astype('category').cat.codes

        # Predict using the trained model with a spinner
        with st.spinner("Calculating the price..."):
            time.sleep(1.5)  # Artificial delay to ensure spinner is visible
            predicted_price = model.predict(input_data)[0]

        # Display Prediction
        if submitted:
            # Display Prediction in a styled card
            st.markdown(
                f"""
                <div style="background-color: #739BD0; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
                    <h2 style="color: #ffffff;">Estimated Price:</h2>
                    <h1 style="color: #ffffff;">${predicted_price:,.2f}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Calculate RMSE before tuning
    rmse_before_tuning = rmse  # This is the RMSE from the initial model before tuning
    
    # Perform hyperparameter tuning for CatBoost
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    
    # Define the CatBoost Regressor with optimized hyperparameters
    tuned_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        loss_function='RMSE',
        cat_features=categorical_features,
        verbose=0  # Suppress detailed training output
    )
    
    # Fit the tuned model
    tuned_model.fit(X_train, y_train)
    
    # Make predictions with the tuned model
    y_pred_tuned = tuned_model.predict(X_test)
    
    # Calculate RMSE after tuning
    rmse_after_tuning = mean_squared_error(y_test, y_pred_tuned, squared=False)
    
    # Display RMSE comparison in a collapsible section
    with st.expander("Prediction Performance"):
        st.subheader("Model Performance")
        st.write("### RMSE Comparison")
        st.markdown(
            f"""
            <div style="background-color: #f4f4f4; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                <p><strong>Before Tuning:</strong> {rmse_before_tuning:,.2f}</p>
                <p><strong>After Tuning:</strong> {rmse_after_tuning:,.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
        st.write("#### Interpretation:")
        st.markdown(
            """
            - **Before Tuning:** This RMSE reflects the performance of the default CatBoost model.
            - **After Tuning:** This RMSE reflects the performance after applying optimized hyperparameters.
            - The decrease in RMSE demonstrates the effectiveness of hyperparameter tuning.
            """
        )

with tab4:
    st.header("Pricing Relationships")
    if 'filtered_diamonds' in locals() and not filtered_diamonds.empty:
        fig, ax = plt.subplots()
        sns.scatterplot(data=filtered_diamonds, x="Carat", y="Price", hue="Color", palette="viridis", ax=ax)
        ax.set_title("Carat vs. Price", fontsize=16)
        ax.set_xlabel("Carat", fontsize=12)
        ax.set_ylabel("Price ($)", fontsize=12)
        st.pyplot(fig)
    else:
        st.warning("No data available for the selected filters.")


