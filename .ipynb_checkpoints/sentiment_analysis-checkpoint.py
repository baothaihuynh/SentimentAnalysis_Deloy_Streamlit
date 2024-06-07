# Import Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st
from PIL import Image
from wordcloud import WordCloud
import plotly.express as px
from plotly.graph_objects import Figure
from plotly.graph_objs import graph_objs
import plotly.graph_objects as go


# Read data cleaned
data_analysis = pd.read_csv('data_cleaned/data_analysis.csv')
data_model = pd.read_csv('data_cleaned/data_model.csv')

# Create menu
menu = [
    "📝Overview",
    "📊About Project",
    "🔎New Predict",
    "📂Find Customer Information in Dataset",
]
choice = st.sidebar.selectbox("TABLE OF CONTENTS", menu)