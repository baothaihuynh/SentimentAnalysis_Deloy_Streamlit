import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from helpers.find_adj_word import find_negative_words, find_positive_words
from wordcloud import WordCloud, STOPWORDS

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


def find_restaurant(restaurantid, df):
    data = df
    # restaurantid = restaurantid
    # lst_idrestaurant = list(data["RestaurantID"].unique())
    # if restaurantid not in lst_idrestaurant:
    # print("This restaurant currently has no information!")
    # else:
    print("Restaurant Information:")
    print(
        "- Restaurant Name:",
        data[data["RestaurantID"] == restaurantid]["Restaurant Name"].iloc[0],
    )
    print("- Address:", data[data["RestaurantID"] == restaurantid]["Address"].iloc[0])
    print(
        "- Time Open:", data[data["RestaurantID"] == restaurantid]["Time Open"].iloc[0]
    )
    print(
        "- Time Close:",
        data[data["RestaurantID"] == restaurantid]["Time Close"].iloc[0],
    )
    print(
        "- Price from",
        data[data["RestaurantID"] == restaurantid]["Lowest Price"].iloc[0],
        "to",
        data[data["RestaurantID"] == restaurantid]["Highest Price"].iloc[0],
        "VND",
    )
    print("--" * 100)
    lst_negative_word = data[
        (data["Label"] == "Negative") & (data["RestaurantID"] == restaurantid)
    ]["Comment Tokenize"].apply(lambda x: find_negative_words(x))
    lst_negative_word = lst_negative_word.to_frame()
    lst_nega = []
    for i in range(len(lst_negative_word)):
        lst_nega.extend(lst_negative_word["Comment Tokenize"].iloc[i])
    lst_nega_unique = list(set(lst_nega))
    print("- List Adj Negative:", ", ".join(lst_nega_unique))
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
        df_word_nega[df_word_nega["Count Word"] == df_word_nega["Count Word"].max()][
            "Key Word"
        ][1]
    )
    problem = df_word_nega[
        df_word_nega["Count Word"] == df_word_nega["Count Word"].max()
    ]["Problem"][1]
    print(
        f"- The restaurant is currently experiencing some issues with the {problem.upper()}, with some keywords appearing frequently in the comments: {', '.join(keyword_problem_unique)}"
    )
    print("--" * 100)
    print("Advantage")
    lst_positive_word = data[
        (data["Label"] == "Negative") & (data["RestaurantID"] == restaurantid)
    ]["Comment Tokenize"].apply(lambda x: find_positive_words(x))
    lst_positive_word = lst_positive_word.to_frame()
    lst_posi = []
    for i in range(len(lst_positive_word)):
        lst_posi.extend(lst_positive_word["Comment Tokenize"].iloc[i])
    lst_posi_uniqe = list(set(lst_posi))
    print("- List Adj Positive:", ", ".join(lst_posi_uniqe))

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
        df_word_posi[df_word_posi["Count Word"] == df_word_posi["Count Word"].max()][
            "Key Word"
        ][1]
    )
    advantage = df_word_posi[
        df_word_posi["Count Word"] == df_word_posi["Count Word"].max()
    ]["Advantage"][1]
    print(
        f"- The restaurant excels with the {advantage.upper()}, with some keywords appearing frequently in the comments: {', '.join(keyword_advantage_unique)}"
    )
    print("--" * 100)

    print("Statistics by Charts")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Create countplot of Label
    sb.countplot(data=data[data["RestaurantID"] == restaurantid], x="Label", ax=axes[0])
    axes[0].set_title("Number for each Label")
    # Create hisplot of Rating
    sb.histplot(data=data[data["RestaurantID"] == restaurantid], x="Rating", ax=axes[1])
    axes[1].set_title("Rating Distribution")
    # Create wordcloud
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    for i, label in enumerate(["Negative", "Positive"]):
        text = " ".join(
            data[(data["RestaurantID"] == restaurantid) & (data["Label"] == label)][
                "Comment Tokenize"
            ]
        )
        if len(text) == 0:
            print(
                f"Insufficient word count in Comment to generate a word cloud of {label}."
            )
            break
        else:
            wordcloud = WordCloud(
                background_color="white", width=400, height=300, max_words=100
            ).generate(text)
            axes[i].imshow(wordcloud, interpolation="bilinear")
            axes[i].set_title(f"Customers of {label} Group say", fontsize=18)
            axes[i].axis("off")
