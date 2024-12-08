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

    # Create sliders and filters for user preferences
    mass_range = st.slider(
        "Select Desired Carat Range",
        min_value=float(df["Carat"].min()),
        max_value=float(df["Carat"].max()),
        value=(float(df["Carat"].min()), float(df["Carat"].max()))
    )

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

    # Apply filters to the DataFrame
    filtered_diamonds = df[
        (df["Carat"] >= mass_range[0]) &
        (df["Carat"] <= mass_range[1]) &
        (df["Cut"].isin(cut_options)) &
        (df["Color"].isin(color_options)) &
        (df["Clarity"].isin(clarity_options))
    ]

    # Add a multiselect for column selection
    st.subheader("Customize Columns")
    default_columns = ['Price', 'Carat', 'Cut', 'Color']  # Default columns to display
    columns_to_display = st.multiselect(
        "Select Columns to Display:",
        options=filtered_diamonds.columns.tolist(),
        default=[col for col in default_columns if col in filtered_diamonds.columns]  # Use default columns if available
    )

    # Display filtered results with selected columns
    st.subheader("Filtered Diamonds")

    num_results = len(filtered_diamonds)
    st.markdown(f"**{num_results} results**")

    if filtered_diamonds.empty:
        st.warning("No diamonds match your selected criteria. Please adjust the filters.")
    else:
        # Display the filtered DataFrame with selected columns
        st.dataframe(filtered_diamonds[columns_to_display].reset_index(drop=True))  # Reset index and drop the original one


#### MODEL
###########################################
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#### Separating target and features
X = df.drop(columns=['Price', 'Table', 'Depth'])  # Exclude unnecessary columns
y = df['Price']  # Target variable

# Encode categorical variables (if needed)
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the CatBoost regressor
model = CatBoostRegressor(
    iterations=1000, 
    learning_rate=0.1,
    depth=6,
    cat_features=categorical_features,  # Specify categorical columns
    verbose=100  # Show progress
)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

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
        st.markdown(
            f"""
            <div style="background-color: #000000; padding: 15px; border-radius: 5px; text-align: center; margin-top: 20px;">
                <h2 style="color: #ffffff;">${predicted_price:,.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
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
