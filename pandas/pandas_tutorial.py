import pandas as pd
from datetime import datetime, timedelta

# a = pd.DataFrame({'Yes' : [50, 21], 'No' : [131, 2]})
# b = pd.DataFrame({'Bob' : ['I liked it', 'It was awful'], 'Sue' : ['Pretty good', 'Bland']}, index=["Product A", "Product B"])
# print(a)
# print(b)
# c = pd.Series([1, 2, 3, 4, 5])
# d = pd.Series([30, 40, 50], index = ['2020', '2021', '2022'], name='Product A')
# print(c)
# print(d)

wine_reviews = pd.read_csv(r"C:\Repositories\machine_learning\datasets\winemag-data-130k-v2.csv", index_col=0)
reviews = wine_reviews

# pd.set_option('display.max_rows', 5)

# print(wine_reviews.shape)
# print(wine_reviews.head())

# # Save a DataFrame to a csv
# wine_reviews.to_csv(r"C:\Repositories\machine_learning\datasets\saved-wine-reviews.csv")

# print(wine_reviews.country)
# print(wine_reviews["country"])
# print(wine_reviews["country"][0])

# Select the 0th row
# print(wine_reviews.iloc[0])

# Select a column ( first parameter specifies range x:y row, second parameter specifies column )
# print(wine_reviews.iloc[:, 0])

# Select first 3 rows of 0th column
# print(wine_reviews.iloc[0:3, 0])

# Select 2nd and 3rd entries in 0th column
# print(wine_reviews.iloc[1:3])

# Select specific entries in 0th column
# print(wine_reviews.iloc[[0, 1, 2], 0])

# Select last 5 entries of all columns, aka last 5 rows
# print(wine_reviews.iloc[-5:])

# Label-based selection, selects the 0th element in the 'country' column
# print(wine_reviews.loc[0, 'country'])

# Select all entries in the following columns : ['taster_name', 'taster_twitter_handle', 'points']
# print(wine_reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']])

# Set a specific column as the index column ( if logically that makes more sense )
# print(wine_reviews.set_index("title"))

# Does a row have Italy as its country
# print(wine_reviews.country == "Italy")

# Display rows where the country is Italy
# print(wine_reviews.loc[wine_reviews.country == "Italy"])

# Display better than average wines from Italy 
# Wines are rated betweenb 80 and 100, so display wines with rating 90+ from Italy
# print(wine_reviews.loc[(wine_reviews.country == "Italy") & (wine_reviews.points >= 90)])

# Wines from Italy OR better than average wines
# print(wine_reviews.loc[(wine_reviews.country == "Italy") | (wine_reviews.points >= 90)])

# Wines where country is in ['Italy', 'France']
# print(wine_reviews.loc[wine_reviews.country.isin(['Italy', 'France'])])

# Wines where price isn't missing
# print(wine_reviews.loc[wine_reviews.price.notnull()])

# ASSIGN DATA
# Assign constant value for all values in a column
# wine_reviews["critic"] = "everyone"
# print(wine_reviews["critic"])

# Assign an iterable of values
# wine_reviews["index_backwards"] = range(len(wine_reviews), 0, -1)
# print(wine_reviews["index_backwards"])

#########################################################

# SUMMARY FUNCTIONS and MAPS

# Type-aware method, that shows different stats for different types of data
# print(wine_reviews.points.describe())

# print(wine_reviews.taster_name.describe())

# Average number of points a wine was given
# print(wine_reviews.points.mean())

# To see a list of unique values in a column
# print(wine_reviews.taster_name.unique())

# To see unique values, and their respective frequencies
# print(wine_reviews.taster_name.value_counts())


# MAPS

# Map and apply, don't affect the original DataFrames or Series, they're called on

# Map method is for transforming a Series, one element at a time
# Re-mean the scores the wines received to 0 ( You subtract the mean of all points, from each entry )
# wine_reviews_points_mean = wine_reviews.points.mean()
# mean_0_wine_reviews_points = wine_reviews.points.map(lambda p : p - wine_reviews_points_mean)

# print(wine_reviews.points)
# print(mean_0_wine_reviews_points)

# def remean_points(row):
#     row.points = row.points - wine_reviews_points_mean
#     return row

# Apply method is for transforming a DataFrame, one row at a time
# wine_reviews_with_remeaned_points = wine_reviews.apply(remean_points, axis="columns")

# print(wine_reviews.points)
# print(wine_reviews_with_remeaned_points.points)

# print(wine_reviews.head(1))

# wine_reviews_points_mean = wine_reviews.points.mean()
# mean_0_wine_reviews_points = wine_reviews.points - wine_reviews_points_mean

# print(wine_reviews.points)
# print(mean_0_wine_reviews_points)

# Series + string + Series works ( pandas broadcasts the "-" to all the rows)
# location = wine_reviews.country + "-" + wine_reviews.region_1

# fstring doesn't work, as it assumes the series are strings
# location = f"{wine_reviews.country} - {wine_reviews.region_1}"

# print(location)

# Find median of a points
# points_median = wine_reviews.points.median()

# Find unique countries
# unique_countries = wine_reviews.country.unique()

# Wine with the highest points-to-price ratio
# print(wine_reviews.points)
# print(wine_reviews.price)

# bargain_idx = (wine_reviews.points / wine_reviews.price).idxmax()
# print(bargain_idx)

# bargain_wine = wine_reviews.loc[bargain_idx, 'title']
# print(bargain_wine)

# def retrieve_bargain_wine(reviews):
#     reviews.points_to_price = reviews.points / reviews.price
#     reviews.points_to_price.max()

#     return reviews.loc[reviews.points_to_price == reviews.points_to_price.max()]
    
# print(retrieve_bargain_wine(wine_reviews))

# There are only so many words you can use when describing a bottle of wine. 
# Is a wine more likely to be "tropical" or "fruity"? 
# Create a Series descriptor_counts 
# counting how many times each of these two words appears in the description column in the dataset. 
# (For simplicity, let's ignore the capitalized versions of these words.)

# print(reviews.description)

# Count the number of times the string "tropical" appears in these descriptions

##############################################

# for i in range(10) :
#     my_method_start = datetime.utcnow()
#     tropical_count = 0
#     fruity_count = 0

#     for description in reviews.description:
#         if "tropical" in description :
#             tropical_count += 1
#         if "fruity" in description :
#             fruity_count += 1

#     descriptor_counts = pd.Series([tropical_count, fruity_count], index=["tropical", "fruity"])
#     my_method_end = datetime.utcnow()
#     # print(descriptor_counts)
#     my_method_time_in_milliseconds = (my_method_end - my_method_start).total_seconds() * 1000
#     print(f"Time taken : {my_method_time_in_milliseconds}")

################################################################

# for i in range(10):
#     pd_method_start = datetime.utcnow()
#     n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum() 
#     n_fruity = reviews.description.map(lambda desc : "fruity" in desc).sum()
#     descriptor_counts = pd.Series([n_trop, n_fruity], index=["tropical", "fruity"])
#     pd_method_end = datetime.utcnow()
#     # print(descriptor_counts)
#     pd_method_time_in_milliseconds = (pd_method_end - pd_method_start).total_seconds() * 1000
#     print(f"Time taken : {pd_method_time_in_milliseconds}")

################################################################

# for i in range(10):
#     vec_method_start = datetime.utcnow()
#     n_trop = reviews.description.str.contains("tropical").sum()
#     n_fruity = reviews.description.str.contains("fruity").sum()
#     descriptor_counts = pd.Series([n_trop, n_fruity], index=["tropical", "fruity"])
#     vec_method_end = datetime.utcnow()
#     vec_method_time_ms = (vec_method_end - vec_method_start).total_seconds() * 1000
#     print(f"Time taken : {vec_method_time_ms:.2f} ms")

#######################################################

# for i in range(10):
#     fast_contains_start = datetime.utcnow()
#     n_trop = reviews.description.str.contains("tropical", regex=False).sum()
#     n_fruity = reviews.description.str.contains("fruity", regex=False).sum()
#     descriptor_counts = pd.Series([n_trop, n_fruity], index=["tropical", "fruity"])
#     fast_contains_end = datetime.utcnow()
#     fast_contains_ms = (fast_contains_end - fast_contains_start).total_seconds() * 1000
#     print(f"Time taken : {fast_contains_ms:.2f} ms")

#########################################################

# We'd like to host these wine reviews on our website, 
# but a rating system ranging from 80 to 100 points is too hard to understand - 
# we'd like to translate them into simple star ratings. 
# A score of 95 or higher counts as 3 stars, a score of at least 85 but less than 95 is 2 stars. Any other score is 1 star.
# Also, the Canadian Vintners Association bought a lot of ads on the site, 
# so any wines from Canada should automatically get 3 stars, regardless of points.
# Create a series star_ratings with the number of stars corresponding to each review in the dataset.

def generate_star_rating(row):
    if row.points >= 95 or row.country == "Canada":
        return 3
    elif row.points >= 85 :
        return 2
    else :
        return 1

reviews_with_stars = reviews.copy()
reviews_with_stars["stars"] = reviews.apply(generate_star_rating, axis="columns")

star_ratings = reviews_with_stars.stars

print(star_ratings)