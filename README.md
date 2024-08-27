# Interactive Prediction App

## Overview

This Streamlit app provides interactive prediction capabilities for two key tasks:

1. **Predict Student Scores**: Analyze student performance based on hours of study.
2. **Predict Iris Species**: Classify iris species based on measurements including Sepal Length, Sepal Width, Petal Length, and Petal Width.

## Prerequisites

Ensure you have Python installed on your system. You can download Python from [python.org](https://www.python.org/downloads/).

## Installation

1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
   Create a Virtual Environment (optional but recommended):

bash
Copy code
python -m venv venv
Activate the Virtual Environment:

On Windows:
bash
Copy code
venv\Scripts\activate
On macOS/Linux:
bash
Copy code
source venv/bin/activate
Install Dependencies: Ensure you have pip installed, then install the necessary packages:

bash
Copy code
pip install streamlit pandas scikit-learn matplotlib seaborn
Running the Streamlit App
Navigate to the App Directory:

bash
Copy code
cd <directory-containing-app>
Run the Streamlit App:

bash
Copy code
streamlit run main.py
