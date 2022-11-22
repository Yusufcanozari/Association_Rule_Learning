import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("../input/armuts/armut_data.csv")

df["hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["newdate"] =df["CreateDate"].dt.strftime("%Y-%m")

df["SepetID"] = df["UserId"].astype(str) + "_" + df["newdate"].astype(str)

df_1 =df.groupby(["SepetID","hizmet"])["hizmet"].count().unstack().fillna(0).applymap(lambda x:1 if x >0 else 0)

freq_itemset = apriori(df_1,min_support = 0.01,use_colnames = True)
rules = association_rules(freq_itemset,metric="support",min_threshold = 0.01)


def script():
    df["hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
    df["CreateDate"] = pd.to_datetime(df["CreateDate"])
    df["newdate"] = df["CreateDate"].dt.strftime("%Y-%m")
    df["SepetID"] = df["UserId"].astype(str) + "_" + df["newdate"].astype(str)
    df_1 = df.groupby(["SepetID", "hizmet"])["hizmet"].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    freq_itemset = apriori(df_1, min_support=0.01, use_colnames=True)
    rules = association_rules(freq_itemset, metric="support", min_threshold=0.01)
    return rules

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(script(), "2_0", 10)