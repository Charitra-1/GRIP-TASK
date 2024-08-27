import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Set the page config
st.set_page_config(page_title="Interactive Prediction App", layout="centered")

# Set background color and text color
st.markdown(
    """
    <style>
    body {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #000000;
        border: 1px solid #FFFFFF;
        padding: 10px;
        border-radius: 5px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def home_page():
    # Set the title with centered text
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1 style='color: white;'>Interactive Prediction App</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Add a description about the web app in a single paragraph
    st.markdown(
        """
        <div style='text-align: justifly; margin: 20px 0;'>
            <p style='font-size: 18px; color: white;'>
                This web application provides interactive prediction capabilities for two key tasks.
                Predict Student Scores - Analyze student performance based on hours of study, helping to understand how study time impacts academic results.
                Predict Iris Species - Classify iris species based on its measurements, including Sepal Length, Sepal Width, Petal Length, and Petal Width, to identify species based on these attributes.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Add buttons for tasks with more styling
    st.markdown(
        """
        <style>
        .button-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 30px;
        }
        .stButton>button {
            background-color: white;
            color: black;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-size: 18px;
            transition: background-color 0.7s ease;
        }
        .stButton>button:hover {
            transform: scale(1.1);
            color: black;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    st.write("#### Choose a task below to make predictions")

    # Use columns to layout the buttons in a single row with better alignment
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        if st.button('Predict Student Scores By Hours'):
            st.session_state.page = 'scores'

    with col2:
        if st.button('Predict Iris Species By its Sizes'):
            st.session_state.page = 'iris'

    # Add the "Created by" section with GitHub icon
    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <p style='font-size: 18px; color: white;'>
                <a href='https://github.com/chandankumarm55' style='text-decoration: none; color: #FF5722;' target='_blank'> 
                Created by 
                    <img src='https://img.icons8.com/?size=100&id=467&format=png&color=FFFFFF' style='vertical-align: middle;' width='20px' height="20px">
                </a>
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )




# Predict Student Scores Page
def scores_page():
    st.title("Predict Student Scores")
    st.write("Enter the number of hours studied to predict the score.")

    # Load dataset
    df = pd.read_csv("web-application-code/score.csv")
    X = df[["Hours"]]
    y = df["Scores"]

    # Polynomial Regression (Degree 2)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict scores for the dataset
    y_pred = model.predict(X_poly)

    # Input for hours
    hours = st.number_input("Hours Studied", min_value=0.0, step=0.5)

    # Predict score for user input
    if st.button("Predict"):
        if hours == 0:
            st.write(
                "<span style='color: red; font-weight:bold; font-size:larger'>Please select the hours of study.</span>",
                unsafe_allow_html=True
            )
        else:
            hours_poly = poly.transform([[hours]])
            prediction = model.predict(hours_poly)[0]
            clipped_prediction = min(max(float(prediction), 0), 100)  # Ensure prediction is between 0 and 100
            st.write(
                f"**Predicted Score:** <span style='color: green; font-weight:bold; font-size:larger'>{clipped_prediction:.2f}</span>",
                unsafe_allow_html=True
            )

            # Calculate and display evaluation metrics
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            st.write("### Model Evaluation Metrics")
            scoreTable = pd.DataFrame(
                {
                    'Metric': ["Mean Squared Error", "Mean Absolute Error", "R-Squared"],
                    'Value': [f"{mse:.2f}", f"{mae:.2f}", f"{r2:.2f}"]
                }
            )

            st.table(scoreTable)

    # Visualization: Scatter Plot with Regression Line
    st.write("# Data-Set Insights :-")
    st.write("### Hours vs Scores")
    fig, ax = plt.subplots(facecolor='black')
    ax.scatter(X, y, color='white', s=20, label='Actual Scores')  # Reduce the size of the scatter plot markers
    ax.plot(X, model.predict(X_poly), color='red', label='Polynomial Regression Line')
    ax.set_xlabel("Hours Studied", color='white')
    ax.set_ylabel("Scores", color='white')
    ax.set_title("Hours Studied vs Scores", color='white')
    ax.legend()
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # Additional Visualization: Residual Plot
    st.write("### Residual Plot")
    residuals = y - y_pred
    fig, ax = plt.subplots(facecolor='black')
    ax.scatter(X, residuals, color='white', s=20)  # Reduce the size of the scatter plot markers
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel("Hours Studied", color='white')
    ax.set_ylabel("Residuals", color='white')
    ax.set_title("Residual Plot", color='white')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # Additional Visualization: Distribution of Scores
    st.write("### Distribution of Scores")
    fig, ax = plt.subplots(facecolor='black')
    sns.histplot(y, bins=10, kde=True, color='white', ax=ax)
    ax.set_xlabel("Scores", color='white')
    ax.set_ylabel("Frequency", color='white')
    ax.set_title("Distribution of Scores", color='white')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    if st.button("Back to Home"):
        st.session_state.page = "home"
    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <p style='font-size: 18px; color: white;'>
                <a href='https://github.com/chandankumarm55' style='text-decoration: none; color: #FF5722;' target='_blank'>
                Created by  
                    <img src='https://img.icons8.com/?size=100&id=467&format=png&color=FFFFFF' style='vertical-align: middle;' width='20px' height="20px">
                </a>
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )


def iris_page():
    st.title("Predict Iris Species")
    st.write("Enter the Iris flower details to predict its species.")

    # Load Iris dataset
    iris_df = pd.read_csv("web-application-code/Iris.csv")  # Load the dataset
    X_iris = iris_df.drop(columns=["Species", "Id"])  # Drop the "Species" and "Id" columns
    y_iris = iris_df["Species"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

    # Train model
    iris_model = RandomForestClassifier(random_state=42)
    iris_model.fit(X_train, y_train)

    # Input for Iris features
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=1.0)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.5)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.8)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.4)

    # Validate input before predicting
    if st.button("Predict"):
        if sepal_length == 0 or sepal_width == 0 or petal_length == 0 or petal_width == 0:
            st.markdown(
                "<span style='color: red; font-weight:bold; font-size:larger'>Please ensure all values are greater than zero for accurate prediction.</span>",
                unsafe_allow_html=True
            )
            if sepal_length == 0:
                st.markdown("<span style='color: red;'>Please select a valid Sepal Length.</span>", unsafe_allow_html=True)
            if sepal_width == 0:
                st.markdown("<span style='color: red;'>Please select a valid Sepal Width.</span>", unsafe_allow_html=True)
            if petal_length == 0:
                st.markdown("<span style='color: red;'>Please select a valid Petal Length.</span>", unsafe_allow_html=True)
            if petal_width == 0:
                st.markdown("<span style='color: red;'>Please select a valid Petal Width.</span>", unsafe_allow_html=True)
        else:
            iris_prediction = iris_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
            st.markdown(
                f"**Predicted Species:** <span style='color: green; font-weight:bold; font-size:larger'>{iris_prediction}</span>",
                unsafe_allow_html=True
            )
            
            # Show model evaluation metrics after prediction
            y_pred = iris_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            table = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],'Value': [f"{accuracy:.2f}", f"{precision:.2f}", f"{recall:.2f}", f"{f1:.2f}"]})

            st.table(table)

            # Display Confusion Matrix with proper background and visibility
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            # Set the color map to only use white for all values
            cmap = sns.color_palette(["white", "white"])
            
            fig, ax = plt.subplots()

            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, linecolor='black', linewidths=1,
                        annot_kws={"color": "black", "size": 16, "weight": "bold"}, ax=ax)

            ax.set_xlabel('Predicted', color='white')
            ax.set_ylabel('Actual', color='white')
            ax.set_title('Confusion Matrix', color='white')
            ax.xaxis.set_ticklabels(iris_model.classes_, color='white')
            ax.yaxis.set_ticklabels(iris_model.classes_, color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')

            st.pyplot(fig)

    # Distribution and Scatter Plots
    st.write("# Data-Set Insights :-")
    st.write("### Feature Distribution")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), facecolor='black')
    iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].hist(edgecolor='white', color='white', bins=15, ax=axs)

    for ax in axs.flat:
        ax.set_facecolor('black')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

    plt.suptitle("Iris Dataset Feature Distribution", color='white')
    st.pyplot(fig)

    st.write('### Petal and Sepal Distribution')
    fig, ax = plt.subplots(1, 2, figsize=(17, 9), facecolor='black')
    iris_df.plot(x="SepalLengthCm", y="SepalWidthCm", kind="scatter", ax=ax[0], label="Sepal", color='white')
    iris_df.plot(x="PetalLengthCm", y="PetalWidthCm", kind="scatter", ax=ax[1], label="Petal", color='white')

    ax[0].set(title='Sepal Comparison', ylabel='Sepal Width', facecolor='black')
    ax[1].set(title='Petal Comparison', ylabel='Petal Width', facecolor='black')
    ax[0].legend()
    ax[1].legend()

    ax[0].title.set_color('white')
    ax[1].title.set_color('white')
    ax[0].xaxis.label.set_color('white')
    ax[0].yaxis.label.set_color('white')
    ax[1].xaxis.label.set_color('white')
    ax[1].yaxis.label.set_color('white')
    ax[0].tick_params(axis='x', colors='white')
    ax[0].tick_params(axis='y', colors='white')
    ax[1].tick_params(axis='x', colors='white')
    ax[1].tick_params(axis='y', colors='white')

    st.pyplot(fig)

    if st.button("Back to Home"):
        st.session_state.page = "home"
    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <p style='font-size: 18px; color: white;'>
                <a href='https://github.com/chandankumarm55' style='text-decoration: none; color: #FF5722;' target='_blank'> 
                Created by 
                    <img src='https://img.icons8.com/?size=100&id=467&format=png&color=FFFFFF' style='vertical-align: middle;' width='20px' height="20px">
                </a>
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )



# Application Logic
if 'page' not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "scores":
    scores_page()
elif st.session_state.page == "iris":
    iris_page()
