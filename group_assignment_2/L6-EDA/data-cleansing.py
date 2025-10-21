# Databricks notebook source
# DBTITLE 0,--i18n-8c6d3ef3-e44b-4292-a0d3-1aaba0198525
# MAGIC %md 
# MAGIC
# MAGIC # Data Exploration 
# MAGIC
# MAGIC Imagine that we would like to build a ML model to predict the price of a listing based on lodaded data in raw area. 
# MAGIC
# MAGIC Our primary goal at this stage is to conduct data exploration to identify and resolve potential issues with the data, such as invalid types, missing values, and anomalies. 
# MAGIC
# MAGIC Please run <code>load_athens_airbnb_data</code> notebook to get dataset into <code>airbnb.raw</code> schema

# COMMAND ----------

# DBTITLE 0,--i18n-969507ea-bffc-4255-9a99-2306a594625f
# MAGIC %md 
# MAGIC
# MAGIC ## Load Listings Dataset
# MAGIC
# MAGIC Let's load the Airbnb Athens listing dataset in.

# COMMAND ----------


raw_df = spark.read.table('airbnb.raw.listings')
display(raw_df)

# COMMAND ----------

raw_df.columns

# COMMAND ----------

# DBTITLE 0,--i18n-94856418-c319-4915-a73e-5728fcd44101
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC For the sake of simplicity, only keep certain columns from this dataset.

# COMMAND ----------

columns_to_keep = [
    'host_id','host_since','host_is_superhost','latitude','longitude','property_type','room_type','accommodates','bathrooms','bathrooms_text','bedrooms','beds','amenities','price','minimum_nights','maximum_nights', 'number_of_reviews','review_scores_rating','license','instant_bookable','reviews_per_month'
]

base_df = raw_df.select(columns_to_keep)
display(base_df)

# COMMAND ----------

# DBTITLE 0,--i18n-a12c5a59-ad1c-4542-8695-d822ec10c4ca
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC  
# MAGIC ## Fixing Data Types
# MAGIC
# MAGIC Take a look at the schema above. You'll notice that the **`price`** field got picked up as string. For our task, we need it to be a numeric (double type) field. 
# MAGIC
# MAGIC Let's fix that.

# COMMAND ----------

from pyspark.sql.functions import col, translate

fixed_price_df = base_df.withColumn("price", translate(col("price"), "$,", "").cast("double"))

display(fixed_price_df)

# COMMAND ----------

# DBTITLE 0,--i18n-4ad08138-4563-4a93-b038-801832c9bc73
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Summary statistics
# MAGIC
# MAGIC Two options:
# MAGIC * **`describe`**: count, mean, stddev, min, max
# MAGIC * **`summary`**: describe + interquartile range (IQR)
# MAGIC
# MAGIC **Question:** When to use IQR/median over mean? Vice versa?

# COMMAND ----------

# MAGIC %md
# MAGIC **Answer:** We use median/IQR when distributions are skewed or have outliers (e.g., price). The median shows the center of the data, and the IQR shows how spread out the values are without being influenced by outliers. We use mean/std when distributions are roughly symmetric without heavy tails.

# COMMAND ----------

display(fixed_price_df.describe())

# COMMAND ----------

display(fixed_price_df.summary())

# COMMAND ----------

# DBTITLE 0,--i18n-bd55efda-86d0-4584-a6fc-ef4f221b2872
# MAGIC %md 
# MAGIC
# MAGIC ### Explore Dataset with Data Profile
# MAGIC
# MAGIC The **Data Profile** feature in Databricks notebooks offers valuable insights and benefits for data analysis and exploration. By leveraging Data Profile, users gain a comprehensive overview of their **dataset's characteristics, statistics, and data quality metrics**. This feature enables data scientists and analysts to understand the data distribution, identify missing values, detect outliers, and explore descriptive statistics efficiently.
# MAGIC
# MAGIC There are two ways of viewing Data Profiler. The first option is the UI.
# MAGIC
# MAGIC - After using `display` function to show a data frame, click **+** icon next to the *Table* in the header. 
# MAGIC - Click **Data Profile**. 
# MAGIC
# MAGIC
# MAGIC
# MAGIC This functionality is also available through the dbutils API in Python, Scala, and R, using the dbutils.data.summarize(df) command. We can also use **`dbutils.data.summarize(df)`** to display Data Profile UI.
# MAGIC
# MAGIC Note that this features will profile the entire data set in the data frame or SQL query results, not just the portion displayed in the table

# COMMAND ----------

dbutils.data.summarize(fixed_price_df)

# COMMAND ----------

# DBTITLE 0,--i18n-e9860f92-2fbe-4d23-b728-678a7bb4734e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Extreme values
# MAGIC
# MAGIC Let's take a look at the *min* and *max* values of the **`price`** column.

# COMMAND ----------

display(fixed_price_df.select("price").describe())

# COMMAND ----------

display(fixed_price_df.select('price'))

# COMMAND ----------

# DBTITLE 0,--i18n-4a8fe21b-1dac-4edf-a0a3-204f170b05c9
# MAGIC %md 
# MAGIC There are some super-expensive listings, and it's up to theSubject Matter Experts to decide what to do with them.
# MAGIC Let's see first how many listings we can find where the *price* is extream.

# COMMAND ----------

fixed_price_df.filter(col("price") >= 10000).count()

# COMMAND ----------

# DBTITLE 0,--i18n-bf195d9b-ea4d-4a3e-8b61-372be8eec327
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC We have 1 listing that has extream values and lets remove it.

# COMMAND ----------

pos_prices_df = fixed_price_df.filter(col("price") < 10000)

# COMMAND ----------

# DBTITLE 0,--i18n-dc8600db-ebd1-4110-bfb1-ce555bc95245
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's take a look at the *min* and *max* values of the *minimum_nights* column:

# COMMAND ----------

display(pos_prices_df)

# COMMAND ----------

display(pos_prices_df.select("minimum_nights").describe())

# COMMAND ----------

display(pos_prices_df
        .groupBy("minimum_nights").count()
        .orderBy(col("count").desc(), col("minimum_nights"))
       )

# COMMAND ----------

# DBTITLE 0,--i18n-5aa4dfa8-d9a1-42e2-9060-a5dcc3513a0d
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC A minimum stay of 90 days seems to be a reasonable limit here due to . Let's filter out those records where the *minimum_nights* is greater than 90.

# COMMAND ----------

min_nights_df = pos_prices_df.filter(col("minimum_nights") <= 90)

display(min_nights_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Fix bathrooms
# MAGIC
# MAGIC It seems like bathroom column contains null values and information regarding bathrooms counts contained in bathrooms_text column.

# COMMAND ----------

display(min_nights_df.select('bathrooms', 'bathrooms_text'))

# COMMAND ----------

# MAGIC %md
# MAGIC Then we counted raw text categories with groupBy('bathrooms_text') → checked what values are present in the 'bathrooms_text' column to understand how to parse the number of bathrooms effectively and correctly
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F

vc_raw = (
    min_nights_df
    .groupBy('bathrooms_text')
    .count()
    .orderBy(F.desc('count'))
)
display(vc_raw)

# COMMAND ----------


from pyspark.sql.functions import element_at, split

# this code will fail because of anusual data points
# please uncoment and try to figure out how to fix this. 

old_bathrooms_df = min_nights_df

old_bathrooms_df = (
  old_bathrooms_df
  .withColumn('bathrooms', element_at(split('bathrooms_text', ' '), 1).cast('double'))
  .drop('bathrooms_text')
)

# bathrooms_df.count()
# display(old_bathrooms_df)



# COMMAND ----------

# MAGIC %md
# MAGIC We hit an error when counting nulls → Spark’s full scan encountered “Half-bath”, which can’t be cast to double, so the job failed.

# COMMAND ----------

# if you run this cell, you will encounter the error

# null_bathrooms = old_bathrooms_df.filter(F.col("bathrooms").isNull()).count()
# print(f"Number of null values in bathrooms column: {null_bathrooms}")

# COMMAND ----------

# MAGIC %md
# MAGIC Then we implemented a robust parser: normalize text, regexp_extract the number, special-case “half-bath” → 0.5, and try_cast everything else → produced a clean numeric bathrooms.

# COMMAND ----------

bathrooms_df = (
    min_nights_df
    .withColumn("bath_norm", F.trim(F.lower(F.col("bathrooms_text"))))
    .withColumn("num_str", F.regexp_extract(F.col("bath_norm"), r"(\d+(?:\.\d+)?)", 1))
    .withColumn(
        "bathrooms",
        F.when(F.col("bathrooms").isNotNull(), F.col("bathrooms").cast("double"))
         .when(F.col("bath_norm").rlike(r"\bhalf[- ]bath"), F.lit(0.5))
         .otherwise(F.expr("try_cast(num_str as double)"))
    )
    .drop("bathrooms_text","bath_norm","num_str")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Then we verified remaining nulls in the cleaned column.

# COMMAND ----------

null_bathrooms = bathrooms_df.filter(F.col("bathrooms").isNull()).count()
print(f"Number of null values in bathrooms column: {null_bathrooms}")

# COMMAND ----------

# MAGIC %md
# MAGIC Then we parsed the old text into numeric bins (bathrooms_num) and aggregated counts (count_old).

# COMMAND ----------

old_group = (
    min_nights_df
    .withColumn("bt", F.trim(F.lower(F.col("bathrooms_text"))))
    .withColumn("n_str", F.regexp_extract(F.col("bt"), r"(\d+(?:\.\d+)?)", 1))
    .withColumn("is_half", F.col("bt").rlike(r"\bhalf[- ]bath(s)?\b"))
    .withColumn(
        "bathrooms_num",
        F.when(F.col("is_half"), F.lit(0.5))
         .when(F.col("n_str") != "", F.col("n_str").cast("double"))
         .otherwise(F.lit(None).cast("double"))
    )
    .groupBy("bathrooms_num")
    .agg(F.count("*").alias("count_old"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Then we aggregated the new numeric column to counts (count_new) for comparison.

# COMMAND ----------

new_dist = (
    bathrooms_df
    .groupBy("bathrooms")
    .agg(F.count("*").alias("count_new"))
    .orderBy("bathrooms")
)

# COMMAND ----------

new_group = new_dist.withColumnRenamed("bathrooms", "bathrooms_num")

# COMMAND ----------

o = old_group.alias("o")
n = new_group.alias("n")

# COMMAND ----------

# MAGIC %md
# MAGIC Then we joined with null-safe equality and added a boolean matched flag → counts match across bins, confirming the parsing is consistent.

# COMMAND ----------

comparison = (
    o.join(n, o["bathrooms_num"].eqNullSafe(n["bathrooms_num"]), "full")
     .select(
        F.coalesce(o["bathrooms_num"], n["bathrooms_num"]).alias("bathrooms_num"),
        o["count_old"],
        n["count_new"]
     )
     .select(
        "bathrooms_num",
        F.coalesce(F.col("count_old"), F.lit(0)).alias("count_old"),
        F.coalesce(F.col("count_new"), F.lit(0)).alias("count_new")
     )
     .withColumn("matched", F.col("count_old") == F.col("count_new"))
     .orderBy("bathrooms_num")
)

display(comparison)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Fix boolean columns types 
# MAGIC
# MAGIC There are two columns `host_is_superhost` and `instant_bookable` that potentially may be usefull, but they are encoded as 't' or 'f' values.
# MAGIC Lets change it to boolean type. 
# MAGIC

# COMMAND ----------

boolean_df = (
  bathrooms_df
  .withColumn('instant_bookable', col('instant_bookable') == 't')
  .withColumn('host_is_superhost', col('host_is_superhost') == 't')
)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Fix amenities
# MAGIC
# MAGIC We can notice that `amenities` column contains a list of available items but it looks like string list which is not easy to work with. 
# MAGIC PySpark has a number of build-in methods to work with arrays: __array_*__
# MAGIC Lets convert amenities into propper aray of strings column.

# COMMAND ----------


from pyspark.sql.functions import explode, lower

amenities_df = boolean_df.withColumn('amenities', split(translate('amenities', '\\]\\[\\"', ''), ','))

display(
  amenities_df
  .select(explode('amenities'))
  .withColumn('item', lower('col'))
  .groupBy('item').count()
  .sort('count')
)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Combine all above transformations
# MAGIC

# COMMAND ----------

def prepare_athens_listings(raw_df):
    cols = [
        'host_id','host_since','host_is_superhost','latitude','longitude','property_type','room_type',
        'accommodates','bathrooms','bathrooms_text','bedrooms','beds','amenities','price',
        'minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','license',
        'instant_bookable','reviews_per_month'
    ]
    df = raw_df.select(*cols)

    df = df.withColumn('price', F.translate(F.col('price'), '$,', '').cast('double'))

    df = (
        df
        .withColumn('bath_norm', F.trim(F.lower(F.col('bathrooms_text'))))
        .withColumn('num_str', F.regexp_extract(F.col('bath_norm'), r'(\d+(?:\.\d+)?)', 1))
        .withColumn(
            'bathrooms',
            F.when(F.col('bathrooms').isNotNull(), F.col('bathrooms').cast('double'))
             .when(F.col('bath_norm').rlike(r'\bhalf[- ]bath'), F.lit(0.5))
             .otherwise(F.expr('try_cast(num_str as double)'))
        )
        .drop('bath_norm','num_str')
    )

    df = (
        df
        .withColumn('instant_bookable', F.col('instant_bookable') == F.lit('t'))
        .withColumn('host_is_superhost', F.col('host_is_superhost') == F.lit('t'))
    )

    df = df.withColumn(
        'amenities',
        F.transform(
            F.split(F.translate('amenities', '\\]\\[\\"', ''), ','),
            lambda x: F.trim(F.lower(x))
        )
    )

    df = df.withColumn('host_since', F.to_date('host_since'))

    def blank(colname):
        return (F.col(colname).isNull()) | (F.trim(F.col(colname)) == "")

    rules = [
        (F.col('price').isNull(), 'price_missing'),
        ((F.col('price') <= 0) | (F.col('price') >= 10000), 'price_out_of_range'),

        (F.col('minimum_nights').isNull(), 'min_nights_missing'),
        ((F.col('minimum_nights') <= 0) | (F.col('minimum_nights') > 90), 'min_nights_out_of_range'),
        (F.col('maximum_nights').isNull(), 'max_nights_missing'),
        ((F.col('maximum_nights') < F.col('minimum_nights')), 'max_lt_min_nights'),
        ((F.col('maximum_nights') > 3650), 'max_nights_unrealistic'),

        (F.col('accommodates').isNull() | (F.col('accommodates') < 1), 'accommodates_invalid'),
        (F.col('beds').isNull() | (F.col('beds') < 1), 'beds_invalid'),
        (F.col('bedrooms').isNotNull() & F.col('beds').isNotNull() & (F.col('bedrooms') > F.col('beds')), 'bedrooms_gt_beds'),
        ((F.col('bedrooms') > 20), 'bedrooms_unrealistic'),

        (F.col('bathrooms').isNull(), 'bathrooms_missing'),

        (blank('property_type'), 'property_type_missing'),
        (blank('room_type'), 'room_type_missing'),
        (blank('license'), 'license_missing'),

        ((F.col('latitude').isNull()) | (F.col('longitude').isNull()), 'missing_latlon'),
        (~((F.col('latitude') >= -90) & (F.col('latitude') <= 90)), 'lat_out_of_range'),
        (~((F.col('longitude') >= -180) & (F.col('longitude') <= 180)), 'lon_out_of_range'),

        (((F.col('review_scores_rating') < 0) | (F.col('review_scores_rating') > 5)), 'review_score_out_of_range'),
        ((F.col('reviews_per_month') < 0), 'reviews_per_month_negative')
    ]

    reason_cols = [F.when(cond, F.lit(label)) for cond, label in rules]
    df = df.withColumn('invalid_reasons', F.array(*reason_cols))
    df = df.withColumn('invalid_reasons', F.expr("filter(invalid_reasons, x -> x is not null)"))
    df = df.withColumn('is_valid', F.size(F.col('invalid_reasons')) == 0)

    return df

# COMMAND ----------

df = prepare_athens_listings(raw_df)

# COMMAND ----------

display(df)

# COMMAND ----------

display(
    df.select(F.explode_outer('invalid_reasons').alias('reason'))
      .filter(F.col('reason').isNotNull())           # keep only invalid reasons
      .groupBy('reason')
      .count()
      .orderBy(F.desc('count'))
)

# COMMAND ----------

# DBTITLE 0,--i18n-25a35390-d716-43ad-8f51-7e7690e1c913
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Handling Null Values
# MAGIC
# MAGIC There are a lot of different ways to handle null values. Sometimes, null can actually be a key indicator of the thing you are trying to predict (e.g. if you don't fill in certain portions of a form, probability of it getting approved decreases).
# MAGIC
# MAGIC Some ways to handle nulls:
# MAGIC * Drop any records that contain nulls
# MAGIC * Numeric:
# MAGIC   * Replace them with mean/median/zero/etc.
# MAGIC * Categorical:
# MAGIC   * Replace them with the mode
# MAGIC   * Create a special category for null
# MAGIC   
# MAGIC **If you do ANY imputation techniques for categorical/numerical features, you MUST include an additional field specifying that field was imputed.**

# COMMAND ----------

# to do: 
# Work in team and brainstorm the best way to visualize missing values in dataset 

# COMMAND ----------

import math
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker

# COMMAND ----------

# MAGIC %md
# MAGIC ### Column-wise missingness

# COMMAND ----------

def missing_by_column(df):
    n = df.count()
    if n == 0:
        return spark.createDataFrame([], "column string, null_count long, non_null_count long, pct_missing double")

    null_exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
    null_row = df.agg(*null_exprs)

    stack_expr = ", ".join([f"'{c}', `{c}`" for c in df.columns])
    long_df = null_row.selectExpr(f"stack({len(df.columns)}, {stack_expr}) as (column, null_count)")

    result = (
        long_df
        .withColumn("total_rows", F.lit(n))
        .withColumn("non_null_count", F.col("total_rows") - F.col("null_count"))
        .withColumn("pct_missing", F.round((F.col("null_count") / F.col("total_rows")) * 100.0, 2))
        .orderBy(F.desc("pct_missing"))
    )

    return result

# COMMAND ----------

len(df.columns)

# COMMAND ----------

def treat_blanks_as_null(df, cols):
    out = df
    for c in cols:
        if c in df.columns:
            out = out.withColumn(c, F.when(F.trim(F.col(c)) == "", F.lit(None)).otherwise(F.col(c)))
    return out

df = treat_blanks_as_null(df, ["license", "room_type", "property_type"])

# COMMAND ----------

missing_cols = missing_by_column(df)
display(missing_cols)

# COMMAND ----------

mc_pd = missing_cols.toPandas().sort_values("pct_missing", ascending=False).reset_index(drop=True)

TOP_N = None
if TOP_N is not None:
    mc_pd = mc_pd.head(TOP_N)

norm = colors.Normalize(vmin=0, vmax=max(1.0, mc_pd["pct_missing"].max()))
cmap = cm.get_cmap("Blues")
bar_colors = [cmap(norm(v)) for v in mc_pd["pct_missing"]]

def wrap(s, width=28):
    return "\n".join(textwrap.wrap(str(s), width=width)) if isinstance(s, str) else s

labels = [wrap(c) for c in mc_pd["column"]]

h = max(3.5, 0.45 * len(mc_pd))
fig, ax = plt.subplots(figsize=(11, h))

ax.barh(labels, mc_pd["pct_missing"], color=bar_colors, edgecolor="none")

ax.invert_yaxis()
ax.set_title("Column-wise Missingness", pad=12, fontsize=14)
ax.set_xlabel("% missing", labelpad=8)
ax.set_ylabel("Column", labelpad=8)
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100.0, decimals=0))

ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
ax.grid(axis="y", visible=False)

max_pct = mc_pd["pct_missing"].max()
x_pad = max(3.0, 0.05 * max_pct)
ax.set_xlim(0, max_pct + x_pad)

for i, v in enumerate(mc_pd["pct_missing"].values):
    label = f"{v:.1f}%"
    if v >= 10:
        ax.text(v - 0.5, i, label, va="center", ha="right", color="white", fontsize=10)
    else:
        ax.text(v + 0.6, i, label, va="center", ha="left", color="black", fontsize=10)

plt.tight_layout()
# plt.savefig("/dbfs/FileStore/missing_by_column_pretty.png", dpi=160, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Row-wise missingness distribution

# COMMAND ----------

def missing_by_row(df):
    miss_cols = [F.when(F.col(c).isNull(), 1).otherwise(0) for c in df.columns]
    per_row = df.select(F.aggregate(F.array(*miss_cols), F.lit(0), lambda acc, x: acc + x).alias("missing_count"))
    hist = (per_row
            .groupBy("missing_count")
            .count()
            .orderBy("missing_count"))
    return hist

# COMMAND ----------

missing_rows = missing_by_row(df)
display(missing_rows)

# COMMAND ----------

mr_pd = (missing_rows
         .toPandas()
         .sort_values("missing_count")
         .reset_index(drop=True))

total_rows = mr_pd["count"].sum()
mr_pd["pct_rows"] = 100.0 * mr_pd["count"] / total_rows

norm = colors.Normalize(vmin=mr_pd["missing_count"].min(), vmax=mr_pd["missing_count"].max())
cmap = cm.get_cmap("Blues").reversed()
bar_colors = [cmap(norm(v)) for v in mr_pd["missing_count"]]

w = max(8, 0.8 * len(mr_pd))
h = 5
fig, ax = plt.subplots(figsize=(w, h))

bars = ax.bar(mr_pd["missing_count"], mr_pd["count"], color=bar_colors, edgecolor="none", width=0.8)

ax.set_title("Row-wise Missingness (Count of NULLs per Row)", pad=12, fontsize=14)
ax.set_xlabel("# missing values in a row", labelpad=8)
ax.set_ylabel("Rows", labelpad=8)

ax.set_xticks(mr_pd["missing_count"])
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))

ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
ax.grid(axis="x", visible=False)

ymax = mr_pd["count"].max()
top_pad = 0.12
ax.set_ylim(0, ymax * (1 + top_pad))

for rect, cnt, p in zip(bars, mr_pd["count"], mr_pd["pct_rows"]):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    label = f"{cnt:,} ({p:.1f}%)"
    ax.text(x, y + 0.01 * ymax, label,
            ha="center", va="bottom", color="black", fontsize=10, clip_on=False)

plt.tight_layout()
# plt.savefig("/dbfs/FileStore/missing_by_row_labels_above.png", dpi=160, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missingness by group

# COMMAND ----------

def missing_by_group(df, group_col):
    if group_col not in df.columns:
        raise ValueError(f"{group_col} not in DataFrame")
    agg_exprs = [F.avg(F.when(F.col(c).isNull(), 1.0).otherwise(0.0)).alias(c) for c in df.columns if c != group_col]
    wide = df.groupBy(group_col).agg(*agg_exprs)
    other_cols = [c for c in df.columns if c != group_col]
    stack_expr = ", ".join([f"'{c}', `{c}`" for c in other_cols])
    long = wide.selectExpr(group_col, f"stack({len(other_cols)}, {stack_expr}) as (column, frac_missing)")
    out = long.withColumn("pct_missing", F.col("frac_missing") * 100.0).drop("frac_missing") \
              .orderBy(group_col, F.desc("pct_missing"))
    return out

# COMMAND ----------

missing_by_room = missing_by_group(df, "room_type")
display(missing_by_room)

# COMMAND ----------

display(missing_by_group(df, "property_type"))

# COMMAND ----------

mbg_pd = missing_by_room.toPandas()
pivot = mbg_pd.pivot(index="room_type", columns="column", values="pct_missing")

TOP_COLS = 15
if TOP_COLS is not None:
    keep_cols = pivot.max(axis=0).sort_values(ascending=False).head(TOP_COLS).index
    pivot = pivot[keep_cols]

def wrap(s, width=18):
    return "\n".join(textwrap.wrap(str(s), width)) if isinstance(s, str) else s

x_labels = [wrap(c) for c in pivot.columns]
y_labels = list(pivot.index)

n_rows, n_cols = pivot.shape
fig_w = max(8, 0.7 * n_cols)
fig_h = max(3.5, 0.6 * n_rows)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

vmin, vmax = 0.0, 100.0
im = ax.imshow(pivot.values, cmap=cm.get_cmap("Blues"), vmin=vmin, vmax=vmax, aspect="auto")

ax.set_title("Missingness by Group (% of NULLs per Column)", pad=12, fontsize=14)
ax.set_xticks(np.arange(n_cols), labels=x_labels)
ax.set_yticks(np.arange(n_rows), labels=y_labels)

plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

ax.set_xlabel("Column", labelpad=8)
ax.set_ylabel("room_type", labelpad=8)
ax.grid(False)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100.0, decimals=0))
cbar.ax.set_ylabel("% missing", rotation=270, labelpad=15)

if n_rows * n_cols <= 150:
    data = pivot.values
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            txt_color = "white" if v >= 60 else "black"
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", color=txt_color, fontsize=9)

plt.tight_layout()
# plt.savefig("/dbfs/FileStore/missing_by_group_heatmap.png", dpi=160, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pairwise co-missingness

# COMMAND ----------

def pairwise_missing_corr(df, cols=None):
    cols = cols or df.columns
    ind = df.select([F.col(c).isNull().cast("int").alias(c) for c in cols])
    n = df.count()
    if n == 0:
        return spark.createDataFrame([], "col_i string, col_j string, jaccard double, co_missing long, either_missing long")

    from itertools import combinations
    pair_names, pair_exprs = [], []
    for a, b in combinations(cols, 2):
        name = f"{a}__{b}"
        pair_names.append((a, b, name))
        pair_exprs.append(F.sum(F.col(a) * F.col(b)).alias(name))

    col_sums_exprs = [F.sum(F.col(c)).alias(c) for c in cols]
    agg_row = ind.agg(*col_sums_exprs, *pair_exprs).collect()[0]

    rows = []
    for a, b, name in pair_names:
        co = agg_row[name]
        sa = agg_row[a]
        sb = agg_row[b]
        either = sa + sb - co
        j = float(co) / float(either) if either else None
        rows.append((a, b, j, int(co), int(either)))

    return spark.createDataFrame(rows, ["col_i","col_j","jaccard","co_missing","either_missing"]) \
                .orderBy(F.desc("jaccard"))

# COMMAND ----------

subset = ["price","bathrooms","bedrooms","beds","reviews_per_month","review_scores_rating","license","host_since"]
pw = pairwise_missing_corr(df, subset)
display(pw)

# COMMAND ----------

pw_pd = pw.toPandas().copy()

pw_pd["jaccard"] = pw_pd["jaccard"].astype(float)

cols = []
for c in list(pw_pd["col_i"]) + list(pw_pd["col_j"]):
    if c not in cols:
        cols.append(c)

M = np.full((len(cols), len(cols)), np.nan, dtype=float)
idx = {c:i for i, c in enumerate(cols)}
for _, r in pw_pd.iterrows():
    i, j = idx[r["col_i"]], idx[r["col_j"]]
    val = r["jaccard"]
    M[i, j] = val
    M[j, i] = val

def wrap(s, width=14):
    return "\n".join(textwrap.wrap(str(s), width)) if isinstance(s, str) else s

x_labels = [wrap(c) for c in cols]
y_labels = [wrap(c) for c in cols]

n = len(cols)
fig_w = max(7, 0.75 * n)
fig_h = max(6, 0.75 * n)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

cmap = cm.get_cmap("Purples")
im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

ax.set_title("Pairwise Co-missingness (Jaccard of NULLs)", pad=12, fontsize=14)
ax.set_xticks(np.arange(n), labels=x_labels)
ax.set_yticks(np.arange(n), labels=y_labels)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

ax.set_xlabel("Column", labelpad=8)
ax.set_ylabel("Column", labelpad=8)

ax.grid(False)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
cbar.ax.set_ylabel("Jaccard", rotation=270, labelpad=15)

if n * n <= 144:
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if math.isnan(v):
                continue
            txt_color = "white" if v >= 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=txt_color, fontsize=9)

plt.tight_layout()
# plt.savefig("/dbfs/FileStore/pairwise_comissing_heatmap.png", dpi=160, bbox_inches="tight")
plt.show()

# COMMAND ----------

# to do: 

# create a new schema called 'validatated' in airbnb catalog and save your validated listing dataset.

# COMMAND ----------

spark.sql("""
CREATE SCHEMA IF NOT EXISTS airbnb.validatated
COMMENT 'Validated & profiling outputs for Airbnb Athens listings'
""")

# COMMAND ----------

valid_df   = df.filter(F.col("is_valid"))
invalid_df = df.filter(~F.col("is_valid"))

# COMMAND ----------

(valid_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("airbnb.validatated.athens_listings_valid"))

(invalid_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("airbnb.validatated.athens_listings_invalid"))
