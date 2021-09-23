###############################################
# Association Rule Learning Recommender
###############################################

# Gerekli Kütüphaneler ve Ayarlar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False) # çıktının tek satırda olması için
from mlxtend.frequent_patterns import apriori, association_rules

###############################
# Data Preprocessing
###############################

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df = df.head(700)
df.info()
df.head()

# OUTLIERS
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Ekisik gözlemlerin silinmesi
df.dropna(inplace=True)
df.isnull().sum()
# İadelerin atılması
df = df[~ df["Invoice"].str.contains("C", na=False)]
# Quantity ve Price değişkenlerinin 0 dan büyük olan değerlerinin alınması
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
# Aykırı gözlemlerin baskılanması
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.shape
df.describe().T

# Almanya müşterilerinin seçimi
df_de = df[df["Country"] == "Germany"]
df_de.head()

#######################################
# ARL Veri Yapısının Hazırlanması
#######################################
df_de.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:3]
#                                   Quantity
# Description  50'S CHRISTMAS GIFT BAG LARGE  DOLLY GIRL BEAKER  I LOVE LONDON MINI BACKPACK
# Invoice
# 536527                                 NaN                NaN                          NaN
# 536840                                 NaN                NaN                          NaN
# 536861                                 NaN                NaN                          NaN
# 536967                                 NaN                NaN                          NaN
# 536983                                 NaN                NaN                          NaN

df_de.groupby(["Invoice", "StockCode"]).agg({"Quantity": "sum"}).unstack().iloc[0:10, 0:10]


# Eğer ürün alındıysa 1, alınmadıysa 0 kodlanacak.
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

de_inv_pro_df = create_invoice_product_df(df_de)

##############################################
# Birlikte alınama olasılıklarının görülmesi
##############################################
frequent_itemsets = apriori(de_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)
#        support                                           itemsets
# 318   0.818381                                          (POSTAGE)
# 390   0.245077              (ROUND SNACK BOXES SET OF4 WOODLAND )
# 2149  0.225383     (POSTAGE, ROUND SNACK BOXES SET OF4 WOODLAND )
# 388   0.157549               (ROUND SNACK BOXES SET OF 4 FRUITS )
# 2147  0.150985      (POSTAGE, ROUND SNACK BOXES SET OF 4 FRUITS )
#         ...                                                ...
# 3475  0.010941  (CHILDRENS CUTLERY DOLLY GIRL , PLASTERS IN TI...
# 3477  0.010941  (CHILDRENS CUTLERY DOLLY GIRL , REGENCY CAKEST...
# 3478  0.010941  (CHILDRENS CUTLERY DOLLY GIRL , PLASTERS IN TI...
# 3480  0.010941  (CHILDRENS CUTLERY DOLLY GIRL , REGENCY CAKEST...
# 6721  0.010941  (RED SPOT CERAMIC DRAWER KNOB, ROUND SNACK BOX...

##########################################
# Birliktelik kurallarının çıkarılması
##########################################
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()
rules.sort_values("lift", ascending=False).head()

#                                              antecedents                                        consequents  antecedent support  consequent support   support  confidence  lift  leverage  conviction
# 42699  (PACK OF 20 SKULL PAPER NAPKINS, PACK OF 6 SKU...  (SET/6 RED SPOTTY PAPER PLATES, SET/6 RED SPOT...            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
# 35429  (SET OF 4 KNICK KNACK TINS DOILY , ROUND STORA...  (STORAGE TIN VINTAGE LEAF, STORAGE TIN VINTAGE...            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
# 30472  (SET/6 RED SPOTTY PAPER CUPS, PACK OF 6 SKULL ...  (PACK OF 20 SKULL PAPER NAPKINS, PACK OF 6 SKU...            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
# 30473  (PACK OF 6 SKULL PAPER CUPS, PACK OF 6 SKULL P...  (PACK OF 20 SKULL PAPER NAPKINS, SET/6 RED SPO...            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
# 30468  (PACK OF 20 SKULL PAPER NAPKINS, SET/6 RED SPO...  (PACK OF 6 SKULL PAPER CUPS, PACK OF 6 SKULL P...            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
#

rules.sort_values(['confidence', 'lift'], ascending =[False, False])
rules.head()


# Ürün ID si girildiğinde ürünün ismini veren fonksiyon
def check_id(dataframe, product_id):
    product_name = dataframe[dataframe["StockCode"] == product_id][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_de, 21987) # ['PACK OF 6 SKULL PAPER CUPS']
check_id(df_de, 23235) # ['STORAGE TIN VINTAGE LEAF']
check_id(df_de, 22747) # ["POPPY'S PLAYHOUSE BATHROOM"]

####################################################################
# Sepetinde ürün olan kullanıcılar için önerilerinin yapılması
####################################################################

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]


a = arl_recommender(rules, 22492, 3)
b = arl_recommender(rules, 23235, 3)
c = arl_recommender(rules, 22747, 3)

# Önerilen ürünlerin isimleri
basket_product = [a, b, c]
for product in basket_product:
    check_id(df_de, product)
