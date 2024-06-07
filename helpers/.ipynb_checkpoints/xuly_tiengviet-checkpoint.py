from underthesea import word_tokenize, sent_tokenize, pos_tag
import regex
import re

##LOAD EMOJICON
file = open('stopwords/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('stopwords/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('stopwords/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('stopwords/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('stopwords/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

def process_text(text, emoji_dict, teen_dict, wrong_lst):
    text = text.lower()
    text = text.replace("’",'')
    text = regex.sub(r'\.+', ".", text)
    new_sentence =''
    for sentence in sent_tokenize(text):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        ###### DEL wrong words   
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '                    
    text = new_sentence  
    #print(document)
    ###### DEL excess blank space
    text = regex.sub(r'\s+', ' ', text).strip()
    return text

def removeStopWords(text, stop_words):
    text = text.split()
    result = []
    for word in text:
        if word not in stop_words:
            result.append(word)
    return " ".join(result)


def wordtokenize(text):
    return word_tokenize(text, format="text")


def process_postag_thesea(text):
    new_document = ""
    for sentence in sent_tokenize(text):
        sentence = sentence.replace(".", "")
        ## POS tag
        lst_word_type = ["N", "NP", "A", "AB", "AY", "V", "VB", "VY", "R", "M"]
        # lst_word_type = [ 'N', 'NP', 'A', 'AB' , 'AY' , 'ABY', 'V', 'VB', 'VY', 'R' , 'M', 'I']
        # print(pos_tag(word_tokenize(sentence, format="text")))

        sentence = " ".join(
            word[0] if word[1].upper() in lst_word_type else ""
            for word in pos_tag(word_tokenize(sentence, format="text"))
        )
        new_document = new_document + sentence + " "
    # Delete excess blank space
    new_document = regex.sub(r"\s+", " ", new_document).strip()
    return new_document


def removeSpecialChar(text):
    return regex.sub(r"[^\w\s]", "", text)

# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
# Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
# Ví dụ: "ngonnnn" thành "ngon", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)
# Áp dụng hàm chuẩn hóa cho văn bản
# print(normalize_repeated_characters(example))

def stepByStep(text):
    with open("stopwords/vietnamese-stopwords.txt", "r", encoding="utf-8") as file:
        stop_words = file.read()
    stop_words = stop_words.split("\n")

    text = str(text)
    # text = process_text(text, emoji_dict, teen_dict, wrong_lst)
    # text = wordtokenize(text.lower())
    text = process_postag_thesea(text.lower())
    text = removeSpecialChar(text)
    text = removeStopWords(text, stop_words)
    text = normalize_repeated_characters(text)
    return text


# text = "Áo Ba Lỗ"
# text = stepByStep(text)
# print(text)
# print(type(text))
