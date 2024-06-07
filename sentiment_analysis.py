# Import Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st
from PIL import Image
import pickle
import joblib
from wordcloud import WordCloud
import plotly.express as px
from plotly.graph_objects import Figure
from plotly.graph_objs import graph_objs
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from helpers.find_adj_word import find_negative_words, find_positive_words
from helpers.find_restaurantinfo_function import find_restaurant
import helpers.xuly_tiengviet as xt

pd.set_option("display.max_rows", None)
# pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", None)
service_word_nega = [
    "lâu",
    "không chuyên nghiệp",
    "không phản hồi",
    "không thân thiện",
    "không linh hoạt",
    "không tốt",
    "chậm",
    "khó khăn",
    "phức tạp",
    "khó chịu",
    "gây khó dễ",
    "rườm rà",
    "không thể chấp nhận",
    "không rõ ràng",
    "rối rắm",
    "không tiện lợi",
    "cau có",
    "chậm",
    "chậm chạm",
    "lâu",
    "quá lâu",
    "chờ",
    "bực_mình",
]

food_word_nega = [
    "kém",
    "tệ",
    "dở",
    "kém chất lượng",
    "không thích",
    "không thú vị",
    "không ổn",
    "không hợp",
    "không an toàn",
    "thất bại",
    "tồi tệ",
    "không thể chấp nhận",
    "tồi tệ",
    "chưa đẹp",
    "không đẹp",
    "bad",
    "thất vọng",
    "không ngon",
    "không hợp",
    "hôi",
    "không_ngon",
    "không_thích",
    "không_ổn",
    "không_hợp",
    "quá dở",
    "điểm trừ",
    "thức ăn tệ",
    "đồ ăn tệ",
    "nhạt nhẽo",
    "không đặc sắc",
    "tanh",
    "không chất lượng",
    "nhạt",
    "khủng khiếp",
    "thất_vọng",
    "nhạt",
]

price_word_nega = [
    "không đáng giá",
    "không đáng tiền",
    "trộm cướp",
    "quá mắc",
    "không đáng",
    "chả đáng",
]

service_word_posi = [
    "tốt",
    "xuất sắc",
    "tuyệt vời",
    "hài lòng",
    "ưng ý",
    "nhanh",
    "tốn ít thời gian",
    "thân thiện",
    "tận tâm",
    "đáng tin cậy",
    "đẳng cấp",
    "an tâm",
    "phục vụ tốt",
    "làm hài lòng",
    "gây ấn tượng",
    "nổi trội",
    "hoà nhã",
    "chăm chỉ",
    "cẩn thận",
    "vui vẻ",
    "sáng sủa",
    "hào hứng",
    "nhiệt tình",
    "nhanh",
    "niềm nở",
    "sạch sẽ",
    "phục vụ nhanh",
    "dài_dài",
    "tin_tưởng",
    "ủng_hộ",
    "ủng_hộ quán",
    "thoải_mái",
]

food_word_posi = [
    "thích",
    "tốt",
    "xuất sắc",
    "tuyệt vời",
    "tuyệt hảo",
    "đẹp",
    "ổn",
    "ngon",
    "hoàn hảo",
    "chất lượng",
    "thú vị",
    "hấp dẫn",
    "tươi mới",
    "lạ mắt",
    "cao cấp",
    "độc đáo",
    "hợp khẩu vị",
    "rất tốt",
    "rất thích",
    "hấp dẫn",
    "không thể cưỡng lại",
    "thỏa mãn",
    "best",
    "good",
    "nghiện",
    "ngon nhất",
    "quá ngon",
    "quá tuyệt",
    "đúng vị",
    "thức ăn ngon",
    "khá ngon",
    "rất thích",
    "tươi ngon",
    "thơm",
    "chất lượng",
    "sạch sẽ",
    "món ngon",
    "ăn rất ngon",
    "đồ ăn ngon",
    "đa dạng",
    "thơm",
    "ăn ngon",
    "hấp_dẫn",
    "ấn_tượng",
    "quán ngon",
]

price_word_posi = [
    "đáng tiền",
    "giá rẻ",
    "rẻ",
    "giá hợp",
    "ngon giá",
]
# Read data cleaned
data_analysis = pd.read_csv("data_cleaned/data_analysis.csv")
data_model = pd.read_csv("data_cleaned/data_model.csv")

# with open("model/xgboots_model.pkl", "rb") as f:
# model = pickle.load(f)
# with open("model/tfidf.pkl", "rb") as g:
# tfidf = pickle.load(g)


# Create image
img = Image.open("image/Whats-Sentiment-Analysis.png")
new_width = 800
new_height = 400
img = img.resize((new_width, new_height))
st.image(img)
# Create menu
menu = [
    "📝Overview",
    "📊About Project",
    "📂Find Restaurant Information in Dataset",
    "🔎New Predict",
]
choice = st.sidebar.selectbox("TABLE OF CONTENTS", menu)

# Create Overview table
if choice == "📝Overview":
    st.subheader("📝**Overview**")

    # Project Summary
    st.write("#### 🔍Project Summary")
    st.write(
        """
             - We are always excited to explore new things every day, and cuisine is a field that attracts a lot of attention. When choosing a new restaurant, we tend to consider reviews from those who have enjoyed it to decide whether or not to try it.  
             - This is becoming increasingly important in the foodservice industry. Restaurants need to strive to improve the quality of their food and service attitude in order to maintain their reputation and attract new customers.  
             - This project is based on data from 3,615 restaurants with over 15,000 reviews collected from the online food delivery platform ShopeeFood in order to build a system to support restaurants in classifying customer feedback into 3 groups: positive, negative, and neutral through text data.              
             """
    )

    # Business Goal
    st.write("#### 📌Business Goal")
    st.write(
        """
             - Data normalization of Vietnamese text for model building.  
             - Utilize Machine Learning algorithms to categorize customer feedback into three groups: Positive, Negative, and Neutral.  
             - Construct an end-user application that enables users to generate new sentiment predictions/classifications based on trained Machine Learning data.  
    """
    )

    # Work Scope
    st.write("#### 🗂Work Scope")
    st.write(
        """
             - Data collection from the ShopeeFood platform using the Selenium library through [Project Crawl Data.](https://huynhthaibao.notion.site/Crawl-Data-From-ShopeeFood-using-Selenium-1f0cd452570f44548f9f6796b6bffc14)  
             - Analysis of the dataset overview.  
             - Employing Machine Learning algorithms like Decision Tree, XGBoots, Random Forest, etc., to classify customer sentiment.
             - Developing an end-user application using Streamlit.    
    """
    )

    # Domain Knowledge
    st.write("#### 💡Domain Knowledge")
    st.write(
        """ 
            ##### What is Sentiment Analysis?  
             - Sentiment analysis (also known as opinion mining, emotion analysis, and sentiment mining) is the field of using natural language processing, text analysis, computational linguistics, and biometrics to identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied to documents such as reviews and survey responses, social media, news media, and other online content for applications ranging from marketing to customer relationship management and clinical medicine.
        """
    )

    # Create image
    img = Image.open("image/pnn.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)

    st.write(
        """ 
             - Sentiment Analysis is the process of analyzing and evaluating a person's opinion about a particular object (positive, negative, or neutral opinion, etc.). This process can be carried out using rule-based methods, Machine Learning, or a hybrid approach (combining the two methods).
             - Sentiment Analysis is widely applied in practice, especially in business promotion activities. By analyzing user reviews of a product to see whether they are negative, positive, or highlight the product's limitations, companies can improve product quality, enhance their company's image, and strengthen customer satisfaction.
               [Read more...](https://monkeylearn.com/sentiment-analysis/)  
               
            ##### What is Streamlit?
             - Streamlit is a tool specifically designed for Machine Learning Engineers to create simple, user-friendly web interfaces for end-users. It is a complete product with high applicability.  
        """
    )
    # Create image
    img = Image.open("image/streamlit.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)

    st.write(
        """
             - Streamlit is known as the fastest way to turn your code into an application product, "The fastest way to build and share data apps."  
             - With its rich set of built-in functions that enhance app interactivity, Streamlit is the ideal tool for those who want to build a complete product.  
              [Read more...](https://streamlit.io/)       
        """
    )

    # Technologies and Tools
    st.write("#### 💻Technologies and Tools")
    st.write(
        """ 
             - Clean data: Using Python’s libraries like Pandas and Numpy.
             - Machine Learning: Using Python’s library Sklearn likes: Decision Tree, Random Forest, XGBoots,...
             - Analysis: Using Python’s libraries like Scipy, Pandas and Numpy.
             - Visualization: Creating informative visualizations using Python’s libraries like Matplotlib, Seaborn and Plotly.
             - User’s App: Using Streamlit to build and deloy app for end user.
        """
    )


elif choice == "📊About Project":
    st.subheader("📊**About Project**")

    # Summary Dataset
    st.write("#### I. Summary Dataset")
    st.dataframe(data_analysis.head(3))
    st.dataframe(data_analysis.tail(3))

    # Exploratory Data Analysis
    st.write("#### II. Exploratory Data Analysis")

    # Task 1: What is the monthly revenue?
    st.write(" ##### Task 1. Trend of restaurant/eatery distribution by district")

    restaurant_district = (
        data_analysis.groupby("District")["RestaurantID"]
        .nunique()
        .reset_index()
        .sort_values(by="RestaurantID", ascending=False)
    )
    lst_color = []
    for i in range(len(restaurant_district)):
        if i == 0:
            lst_color.append("#FBAB4A")
        else:
            lst_color.append("#C4C4C4")

    # Create a plotly figure object
    fig = Figure()
    # Add bars to the figure
    fig.add_traces(
        go.Bar(
            x=restaurant_district["District"],  # Use the extracted list
            y=restaurant_district["RestaurantID"],
            marker_color=lst_color,
        )
    )
    # Set title and layout options
    fig.update_layout(
        title="Count Restaurant for Each District",
        title_x=0,  # Center align title (optional)
        title_font_size=20,
        yaxis_title="Number of Restaurants",
        xaxis_ticktext=restaurant_district[
            "District"
        ],  # Use the extracted list for labels
        xaxis_tickangle=-90,  # Rotate x-axis labels for readability
        height=600,
        width=900,
        plot_bgcolor="white",  # Optional background color
    )
    # Show the plotly chart
    st.plotly_chart(fig)

    st.write(
        """ 
             - The dataset provides a sample of over 800 restaurants/eateries with user reviews. Among these, the highest concentration of restaurants is found in Binh Thanh District.  
             - It is observable that despite not being located in the central districts of Ho Chi Minh City, Binh Thanh has emerged as one of the areas with the most diverse and extensive culinary offerings.
        """
    )

    # Task 2. Customer Comment Trends by Region
    st.write(" ##### Task 2. Customer Comment Trends by Region")
    comment_district = (
        data_analysis.groupby("District")["Comment Tokenize"]
        .count()
        .reset_index()
        .sort_values(by="Comment Tokenize", ascending=False)
    )
    lst_color = []
    for i in range(len(comment_district)):
        if i == 0:
            lst_color.append("#FBAB4A")
        else:
            lst_color.append("#C4C4C4")
    # Create a plotly figure object
    fig = Figure()
    # Add bars to the figure
    fig.add_traces(
        go.Bar(
            x=comment_district["District"],  # Use the extracted list
            y=comment_district["Comment Tokenize"],
            marker_color=lst_color,
        )
    )
    # Set title and layout options
    fig.update_layout(
        title="Count Comment for Each District",
        title_x=0,  # Center align title (optional)
        title_font_size=20,
        yaxis_title="Number of Comment",
        xaxis_ticktext=comment_district[
            "District"
        ],  # Use the extracted list for labels
        xaxis_tickangle=-90,  # Rotate x-axis labels for readability
        height=600,
        width=900,
        plot_bgcolor="white",  # Optional background color
    )
    # Show the plotly chart
    st.plotly_chart(fig)
    st.write(
        """ 
             - While Binh Thanh District leads in terms of restaurant distribution within this dataset, District 1 receives the highest number of reviews for its restaurants.
             - Although the current reviews do not yet determine whether they are positive or negative for the districts, this clearly indicates high user engagement with restaurants in the downtown area.
             - Through the visualization of the two trends above, we can see that Binh Thanh and District 1 are always the two leading areas in terms of restaurant/eatery development on the ShopeeFood platform.
             - The further away from the city center, the sparser the distribution of online food and beverage services, as seen in districts like District 12, Hoc Mon, Binh Chanh, etc. This suggests that areas with high population density and high living standards (central or neighboring districts) drive a higher trend of online food and beverage service development.
        """
    )

    # Task 3. Operating Hour Trends of Restaurants/Eateries on the ShopeeFood Platform
    st.write(
        " ##### Task 3. Operating Hour Trends of Restaurants/Eateries on the ShopeeFood Platform"
    )
    time_open = (
        data_analysis.groupby("Time Open")["RestaurantID"]
        .nunique()
        .reset_index()
        .sort_values(by="RestaurantID", ascending=False)
    )
    lst_color = []
    for i in range(len(time_open)):
        if i == 0:
            lst_color.append("#FBAB4A")
        else:
            lst_color.append("#C4C4C4")
    # Create a plotly figure object
    fig = Figure()
    # Add bars to the figure
    fig.add_traces(
        go.Bar(
            x=time_open["Time Open"],  # Use the extracted list
            y=time_open["RestaurantID"],
            marker_color=lst_color,
        )
    )
    # Set title and layout options
    fig.update_layout(
        title="Count Restaurant for Each Time Open",
        title_x=0,  # Center align title (optional)
        title_font_size=20,
        yaxis_title="Number of Restaurant",
        # xaxis_ticktext=comment_district['District'],  # Use the extracted list for labels
        xaxis_tickangle=-90,  # Rotate x-axis labels for readability
        height=600,
        width=900,
        plot_bgcolor="white",  # Optional background color
    )
    # Show the plotly chart
    st.plotly_chart(fig)
    time_close = (
        data_analysis.groupby("Time Close")["RestaurantID"]
        .nunique()
        .reset_index()
        .sort_values(by="RestaurantID", ascending=False)
    )
    lst_color = []
    for i in range(len(time_close)):
        if i == 0:
            lst_color.append("#FBAB4A")
        else:
            lst_color.append("#C4C4C4")
    # Create a plotly figure object
    fig = Figure()
    # Add bars to the figure
    fig.add_traces(
        go.Bar(
            x=time_close["Time Close"],  # Use the extracted list
            y=time_close["RestaurantID"],
            marker_color=lst_color,
        )
    )
    # Set title and layout options
    fig.update_layout(
        title="Count Restaurant for Each Time Close",
        title_x=0,  # Center align title (optional)
        title_font_size=20,
        yaxis_title="Number of Restaurant",
        # xaxis_ticktext=comment_district['District'],  # Use the extracted list for labels
        xaxis_tickangle=-90,  # Rotate x-axis labels for readability
        height=600,
        width=900,
        plot_bgcolor="white",  # Optional background color
    )
    # Show the plotly chart
    st.plotly_chart(fig)
    st.write(
        """ 
             - The available data suggests that online restaurants/eateries tend to operate 24/24, which could be a unique characteristic of this business model to meet the around-the-clock needs of users.
        
        """
    )

    # Task 4. Food Price Trends
    st.write(" ##### Task 4. Food Price Trends")
    # Create image
    img = Image.open("image/task4_1.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    # Create image
    img = Image.open("image/task4_2.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)

    # Configure plot
    fig = go.Figure()
    # Create two bar traces, one for highest price, one for lowest price
    fig.add_trace(
        go.Bar(
            x=data_analysis["District"],
            y=data_analysis["Highest Price"],
            width=0.4,  # set bar width
            name="Highest Price",
            marker_color="#FBAB4A",  # set bar color
        )
    )
    fig.add_trace(
        go.Bar(
            x=data_analysis["District"],
            y=data_analysis["Lowest Price"],
            width=0.4,  # set bar width
            name="Lowest Price",
            marker_color="#C4C4C4",  # set bar color
        )
    )
    # Customize layout
    fig.update_layout(
        title_text="Highest and Lowest Prices by District",  # set chart title
        title_x=0,  # align title left
        title_font_size=20,
        # xaxis_title='District',  # set x-axis title
        yaxis_title="Price",  # set y-axis title
        xaxis_ticktext=data_analysis["District"],  # set district labels on x-axis
        xaxis_tickangle=-90,  # rotate x-axis labels for better readability
        # yaxis_gridline_pattern='dash',  # add dashed grid lines on y-axis
        legend_title_text="Price",  # set legend title
        legend_title_font_size=12,  # set legend title font size
        height=600,
        width=900,
        plot_bgcolor="white",
    )
    # Show the plotly chart
    st.plotly_chart(fig)
    st.write(
        """  
             - Distribution of Food Prices:
                 - 75% of the cheapest food items fall within the price range of 1,350 - 30,000 VND.
                 - 75% of the most expensive food items fall within the price range of 6,000 - 100,000 VND.
             - Price Outliers:
                 - There are a small number of price outliers in both the highest and lowest price categories, with the most expensive item costing up to 1,000,000 VND.
             - Price Variations Across Districts:
                 - Food prices do not vary significantly across different districts, with overall prices being quite similar.
                 - Interestingly, District 12, which has a very low number of online restaurants, has the highest average food price. This could be due to data outliers.
             - Impact of Location on Prices:
                 - Districts 1 and 11 have the second and third highest average food prices, respectively. This is likely due to their geographical location (central city) influencing pricing.
        """
    )

    # Task 5. Overview of User Review Categorization
    st.write(" ##### Task 5. Overview of User Review Categorization")
    # Create image
    img = Image.open("image/task5_1.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    # Create image
    img = Image.open("image/task5_2.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    st.write(
        """   
             - Data Imbalance: The imbalanced nature of the dataset, with a significant overrepresentation of positive reviews and 10-star ratings, poses a challenge for building a customer sentiment prediction model.
             - Limited Negative and Neutral Reviews: The scarcity of negative and neutral reviews hinders the ability to capture a broader range of customer sentiment and potentially biases the model towards positive opinions.
             - Rating-Review Inconsistency: The inconsistency between rating scores and review content further complicates the task of incorporating rating data into the model.
             - Questionable Value of Negative and Neutral Ratings: The distribution of ratings in the negative and neutral categories suggests that these ratings may not accurately reflect the sentiment expressed in the reviews.       
        """
    )

    # Task 6. Number of Reviews by Restaurant
    st.write(" ##### Task 6. Number of Reviews by Restaurant")
    # Create image
    img = Image.open("image/task6.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    st.write(
        """  
             - 50% of Restaurants with Fewer than 5 Reviews: A significant portion of restaurants on the platform have received very few reviews, indicating that either they are new or have not yet gained a strong customer base.
             - Maximum Reviews per Restaurant: 100: The highest number of reviews for a single restaurant is 100, suggesting that there are a small number of highly popular and well-established restaurants on the platform.
        """
    )

    # Task 7. Restaurants with 100 Reviews
    st.write(" ##### Task 7. Restaurants with 100 Reviews")
    # Create image 1
    img = Image.open("image/task7_1.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    # Create image 2
    img = Image.open("image/task7_2.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    # Create image 3
    img = Image.open("image/task7_3.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    # Create image 4
    img = Image.open("image/task7_4.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    # Create image 5
    img = Image.open("image/task7_5.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    # Create image 6
    img = Image.open("image/task7_6.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    # Create image 7
    img = Image.open("image/task7_7.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    st.write(
        """   
             - High Rating: 75% of the reviews for these restaurants have a rating of 6.6 or higher, indicating a strong overall customer satisfaction level.
             - Positive Review Content: The reviews frequently mention positive keywords like ngon, thích, lắm, thơm_ngon, đa dạng,... highlighting the positive attributes of the restaurants' food and service.
             - Location: The majority of these restaurants are located in District 1, a central and popular area, suggesting that they are easily accessible to a large customer base.
             - 24/7 Availability: The restaurants operate 24/7, catering to the needs of customers seeking late-night or early-morning dining options.
             - Affordable Pricing: The price range for dishes falls between 25,000 and 400,000 VND, making these restaurants accessible to a wide range of customers.
      """
    )

    # Task 8. Restaurants with Fewer than 5 Reviews
    st.write(" ##### Task 8. Restaurants with Fewer than 5 Reviews")
    # Create image 1
    img = Image.open("image/task8_1.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    # Create image 2
    img = Image.open("image/task8_2.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    # Create image 3
    img = Image.open("image/task8_3.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    # Create image 4
    img = Image.open("image/task8_4.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    # Create image 5
    img = Image.open("image/task8_5.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    # Create image 6
    img = Image.open("image/task8_6.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    # Create image 7
    img = Image.open("image/task8_7.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    st.write(
        """   
             - Decent Ratings: 75% of the reviews for these restaurants have a rating of 5.8 or higher, indicating a generally positive customer experience.
             - Positive Review Content: The reviews frequently mention positive keywords like ngon, chất lượng, thích, đậm đà,... highlighting the positive attributes of the restaurants' food and service.
             - Decentralized Locations: These restaurants are spread across various districts, with a notable presence in Bình Thạnh, Thủ Đức, and Gò Vấp, which are located near the city center.
             - 24/7 Availability: The restaurants operate 24/7, catering to the needs of customers seeking late-night or early-morning dining options.
             - Affordable Pricing: The price range for dishes falls between 50,000 VND and under 100,000 VND, making these restaurants accessible to a wide range of customers.
       """
    )

    # Task 9. Conclusion
    st.write(" ##### Task 9. Conclusion")
    st.write(
        """   
             - Geographic Location Drives Review Trends: Despite similarities in review content, operating hours, and pricing, the geographical location of restaurants appears to be a significant factor influencing user review trends.
             - Central Districts Dominate Orders: Data indicates that central city districts, such as District 1 and nearby areas like Bình Thạnh, have a significantly higher number of orders compared to other districts.
             - Proximity to City Center Affects Demand: This suggests that the demand for online food delivery tends to be higher in districts closer to the city center and decreases as the distance from the center increases.
             - Delivery Fees Impact Order Rates: Another potential factor is delivery fees, which are often higher in districts further from the city center, where population density and online food ordering demand are lower. This could explain why restaurants with positive ratings, reviews, and affordable prices might still experience fewer orders in these areas. der.      
        """
    )

elif choice == "📂Find Restaurant Information in Dataset":
    st.subheader("**📂Find Restaurant Information in Dataset**")
    st.write(
        "Please enter RestaurantID similar to the pattern below or select from the suggestions in the box:"
    )
    st.write(data_analysis["RestaurantID"].sample(3))
    restaurant_lst = data_analysis["RestaurantID"].drop_duplicates().values
    restaurantid = st.selectbox("Choose One RestaurantID", restaurant_lst)
    submitted = st.button("Submit")
    if submitted:
        st.write(" ##### Result:")
        st.write("Restaurant Information:")
        st.write(
            "- Restaurant Name:",
            data_analysis[data_analysis["RestaurantID"] == restaurantid][
                "Restaurant Name"
            ].iloc[0],
        )
        st.write(
            "- Address:",
            data_analysis[data_analysis["RestaurantID"] == restaurantid][
                "Address"
            ].iloc[0],
        )
        st.write(
            "- Time Open:",
            data_analysis[data_analysis["RestaurantID"] == restaurantid][
                "Time Open"
            ].iloc[0],
        )
        st.write(
            "- Time Close:",
            data_analysis[data_analysis["RestaurantID"] == restaurantid][
                "Time Close"
            ].iloc[0],
        )
        st.write(
            "- Price from",
            data_analysis[data_analysis["RestaurantID"] == restaurantid][
                "Lowest Price"
            ].iloc[0],
            "to",
            data_analysis[data_analysis["RestaurantID"] == restaurantid][
                "Highest Price"
            ].iloc[0],
            "VND",
        )
        st.write("--" * 100)
        st.write("Problem:")
        lst_negative_word = data_analysis[
            (data_analysis["Label"] == "Negative")
            & (data_analysis["RestaurantID"] == restaurantid)
        ]["Comment Tokenize"].apply(lambda x: find_negative_words(x))
        lst_negative_word = lst_negative_word.to_frame()
        lst_nega = []

        for i in range(len(lst_negative_word)):
            lst_nega.extend(lst_negative_word["Comment Tokenize"].iloc[i])
        lst_nega_unique = list(set(lst_nega))
        st.write("- List Adj Negative:", ", ".join(lst_nega_unique))
        lst_service_word_nega = []
        lst_food_word_nega = []
        lst_price_word_nega = []
        lst_word = []
        for word in lst_nega:
            if word in service_word_nega:
                lst_service_word_nega.append(word)
            elif word in food_word_nega:
                lst_food_word_nega.append(word)
            elif word in price_word_nega:
                lst_price_word_nega.append(word)
            else:
                lst_word.append(word)
        cnt_service_word_nega = len(lst_service_word_nega)
        cnt_food_word_nega = len(lst_food_word_nega)
        cnt_price_word_nega = len(lst_price_word_nega)
        df_word_nega = pd.DataFrame(
            {
                "Problem": ["Service", "Food", "Price"],
                "Key Word": [
                    lst_service_word_nega,
                    lst_food_word_nega,
                    lst_price_word_nega,
                ],
                "Count Word": [
                    cnt_service_word_nega,
                    cnt_food_word_nega,
                    cnt_price_word_nega,
                ],
            }
        )
        keyword_problem_unique = set(
            df_word_nega[
                df_word_nega["Count Word"] == df_word_nega["Count Word"].max()
            ]["Key Word"][1]
        )
        problem = df_word_nega[
            df_word_nega["Count Word"] == df_word_nega["Count Word"].max()
        ]["Problem"][1]
        st.write(
            f"- The restaurant is currently experiencing some issues with the {problem.upper()}, with some keywords appearing frequently in the comments: {', '.join(keyword_problem_unique)}"
        )
        st.write("--" * 100)

        st.write("Advantage")
        lst_positive_word = data_analysis[
            (data_analysis["Label"] == "Negative")
            & (data_analysis["RestaurantID"] == restaurantid)
        ]["Comment Tokenize"].apply(lambda x: find_positive_words(x))
        lst_positive_word = lst_positive_word.to_frame()
        lst_posi = []
        for i in range(len(lst_positive_word)):
            lst_posi.extend(lst_positive_word["Comment Tokenize"].iloc[i])
        lst_posi_uniqe = list(set(lst_posi))
        st.write("- List Adj Positive:", ", ".join(lst_posi_uniqe))

        lst_service_word_posi = []
        lst_food_word_posi = []
        lst_price_word_posi = []
        lst_word = []
        for word in lst_posi:
            if word in service_word_posi:
                lst_service_word_posi.append(word)
            elif word in food_word_posi:
                lst_food_word_posi.append(word)
            elif word in price_word_posi:
                lst_price_word_posi.append(word)
            else:
                lst_word.append(word)
        cnt_service_word_posi = len(lst_service_word_posi)
        cnt_food_word_posi = len(lst_food_word_posi)
        cnt_price_word_posi = len(lst_price_word_posi)
        df_word_posi = pd.DataFrame(
            {
                "Advantage": ["Service", "Food", "Price"],
                "Key Word": [
                    lst_service_word_posi,
                    lst_food_word_posi,
                    lst_price_word_posi,
                ],
                "Count Word": [
                    cnt_service_word_posi,
                    cnt_food_word_posi,
                    cnt_price_word_posi,
                ],
            }
        )
        keyword_advantage_unique = set(
            df_word_posi[
                df_word_posi["Count Word"] == df_word_posi["Count Word"].max()
            ]["Key Word"][1]
        )
        advantage = df_word_posi[
            df_word_posi["Count Word"] == df_word_posi["Count Word"].max()
        ]["Advantage"][1]
        st.write(
            f"- The restaurant excels with the {advantage.upper()}, with some keywords appearing frequently in the comments: {', '.join(keyword_advantage_unique)}"
        )
        st.write("--" * 100)

        st.write("Statistics by Charts:")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # Create countplot of Label
        sb.countplot(
            data=data_analysis[data_analysis["RestaurantID"] == restaurantid],
            x="Label",
            ax=axes[0],
        )
        axes[0].set_title("Number for each Label")
        # Create hisplot of Rating
        sb.histplot(
            data=data_analysis[data_analysis["RestaurantID"] == restaurantid],
            x="Rating",
            ax=axes[1],
        )
        axes[1].set_title("Rating Distribution")
        st.pyplot(fig)

        text_nega = " ".join(
            data_analysis[
                (data_analysis["RestaurantID"] == restaurantid)
                & (data_analysis["Label"] == "Negative")
            ]["Comment Tokenize"]
        )
        if len(text_nega) == 0:
            st.write(
                f"Insufficient word count in Comment to generate a word cloud of Negative Group."
            )
            # break
        else:
            fig, axes = plt.subplots(figsize=(15, 10))
            wordcloud = WordCloud(
                background_color="white", width=800, height=400, max_words=100
            ).generate(text_nega)
            axes.imshow(wordcloud, interpolation="bilinear")
            axes.set_title(f"Customers of Negative Group say", fontsize=30)
            axes.axis("off")
            st.pyplot(fig)

        text_posi = " ".join(
            data_analysis[
                (data_analysis["RestaurantID"] == restaurantid)
                & (data_analysis["Label"] == "Positive")
            ]["Comment Tokenize"]
        )
        if len(text_posi) == 0:
            st.write(
                f"Insufficient word count in Comment to generate a word cloud of Positive Group."
            )
            # break
        else:
            fig, axes = plt.subplots(figsize=(15, 10))
            wordcloud = WordCloud(
                background_color="white", width=800, height=400, max_words=100
            ).generate(text_posi)
            axes.imshow(wordcloud, interpolation="bilinear")
            axes.set_title(f"Customers of Positive Group say", fontsize=30)
            axes.axis("off")
            st.pyplot(fig)

# Create New Predict
else:
    st.subheader("**🔎New Predict**")

    # Select data
    st.write("##### I. Select Data")
    flag = False
    lines = None
    type = st.radio(
        " ###### *Do you want to Input data or Upload data?*",
        options=("Input", "Upload"),
    )

    def enter_your_comment(text):
        # Create df with comment
        df = pd.DataFrame({"Comment": text}, index=[0])

        # transform data using nlp function
        df["Comment Tokenize"] = df["Comment"].apply(xt.stepByStep)
        with open(
            "/mount/src/sentimentanalysis_deloy_streamlit/sentiment_analysis.pkl", "rb"
        ) as f:
            model = pickle.load(f)
        with open("model/tfidf.pkl", "rb") as g:
            tfidf = pickle.load(g)
        # tfidf
        X_test = tfidf.transform(df["Comment Tokenize"])

        y_pred = model.predict(X_test)

        df["Label"] = y_pred
        df["Label"] = df["Label"].map({0: "Negative", 1: "Positive", 2: "Neutral"})
        df = df[["Comment", "Label"]]
        return df

    if type == "Input":
        comment = st.text_input("Enter Your Comment:")
        if st.button("Predict"):
            # Result
            st.write("##### II. Result")
            result = enter_your_comment(comment)
            st.write(result)
            extract_csv_file = result.to_csv(index=False, encoding="utf-8")
            st.download_button(
                label="Download predictions as CSV file",
                data=extract_csv_file,
                file_name="predictions.csv",
                mime="csv",
            )
    else:
        st.write(
            """
    ##### Note:
     - ###### Please provide only the data file in CSV format.
     - ###### Please submit the data file in the following format:
    """
        )
        st.write(
            data_analysis[
                ["RestaurantID", "Restaurant Name", "UserID", "User", "Comment"]
            ].head(5)
        )
        uploaded_file = st.file_uploader(
            "###### *Select your file data:*", type=["csv"]
        )

        def sentiment(df):
            with open("model/xgboots_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("model/tfidf.pkl", "rb") as g:
                tfidf = pickle.load(g)
            df["Comment Tokenize"] = df["Comment"].apply(xt.stepByStep)
            X_test = tfidf.transform(df["Comment Tokenize"])
            y_pred = model.predict(X_test)
            df["Label"] = y_pred
            df["Label"] = df["Label"].map({0: "Negative", 1: "Positive", 2: "Neutral"})
            df = df.drop(columns="Comment Tokenize")
            return df

        if uploaded_file is not None:
            st.write("##### Your data:")
            # Read file data
            data_new = pd.read_csv(uploaded_file, sep=",")
            st.write(data_new)
            df_predict = sentiment(data_new)
            df_predict = df_predict.reset_index(drop=True)
            submitted = st.button("Predict")

            if submitted:
                st.write("##### II. Result")
                st.write(df_predict)
                csv = df_predict.to_csv(index=False)
                st.download_button(
                    label="Download predictions as CSV file",
                    data=csv,
                    file_name="predictions.csv",
                    mime="csv",
                )
