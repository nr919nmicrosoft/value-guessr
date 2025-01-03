import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import decimal
from typing import List
from ands.query_utils import QueryManager
qm = QueryManager()

# Title and Description
st.title("ValueGuessr")
st.markdown("""
Welcome to the **Value Guessr Challenge!** Select your dataset, features, target, and scope, and see how well you can predict impression value. 
Optimize your selections for the best performance!
""")

# Sidebar for inputs
st.sidebar.header("Configuration Panel")

# Fact Table Selection
fact_table = st.sidebar.selectbox(
    "Select Fact Table:", ["agg_platform_seller_analytics"], index=0
)

# Feature Selection
features = st.sidebar.multiselect(
    "Select Features:",
    [
        "seller_sudoku_row", "seller_member_group", "primary_width", "primary_height", 
        "format", "content_category", "tag_position", "tag_media_subtypes", "tag_is_resizable",
        "call_type", "is_prebid_server_included" 
         
    ],
    default=['seller_sudoku_row', 'primary_width', 'primary_height', 'format', 'content_category']
)

# Target Selection
target_column = st.sidebar.selectbox("Select Target:", ["sRPM", "esRPM"], index=0)

# Market Scope
market = st.sidebar.selectbox("Select Market:", ["US", "GB", "FR", "DE", "AU", "JP", "BR"], index=0)

# Additional Scope
scope = st.sidebar.text_area("Enter Additional Scope (Optional):", value="")

# Lookback Window
days = st.sidebar.number_input(
    label="Lookback Window in days:",  # Required label argument
    min_value=1,                                      # Minimum allowed value (1 day)
    max_value=365,                                    # Maximum allowed value (365 days)
    step=1,                                           # Increment step
    format="%i"                                       # Integer format
)

# Ad Requests Threshold
ad_requests_threshold = st.sidebar.selectbox(
    label="Ad Requests Threshold",  # Required label argument
    options=[1, 10, 100, 1000, 10000, 100000, 1000000] # List of allowed values
)

# Dataset Information
st.subheader("Dataset Configuration")
st.write(f"**Fact Table:** {fact_table}")
st.write(f"**Features:** {features}")
st.write(f"**Target:** {target_column}")
st.write(f"**Market:** {market}")
st.write(f"**Lookback Window:** {days} days")
st.write(f"**Ad Requests Threshold:** {ad_requests_threshold}")
if scope:
    st.write(f"**Additional Scope:** {scope}")

dimensions = []

if 'seller_sudoku_row' in features:
    dimensions.append('seller_member_id')
if 'format' in features:
    dimensions.append('media_type')
if ('tag_position' in features) or ('tag_media_subtypes' in features) or ('tag_is_resizable' in features):
    dimensions.append('tag_id')
if 'seller_member_group' in features:
    dimensions.append('seller_member_group_id')
if 'content_category' in features:
    dimensions.append('inventory_url_id')

# Add any other elements from 'features' to the dimensions list
join_requiring_features = {
    'seller_sudoku_row', 
    'format',
    'tag_position', 
    'tag_media_subtypes', 
    'tag_is_resizable', 
    'seller_member_group', 
    'content_category'
}

for feature in features:
    if feature not in join_requiring_features:
        dimensions.append(feature)

# Placeholder to mimic dataset retrieval and preprocessing (Replace this with your actual fetch_and_enrich_data function)
def fetch_and_enrich_data(
    fact_table: str,
    dimensions: List[str],
    market: str,
    timeframe: int,
    ad_requests_threshold: int = 1
) -> pd.DataFrame:
    """
    Fetches aggregated data from a fact table and enriches it with additional dimensions.

    Parameters:
        fact_table (str): The name of the fact table.
        dimensions (List[str]): List of dimension columns to include in the GROUP BY clause.
        market (str): Country code to filter the dataset (e.g., "FR").
        timeframe (int): Number of days for the timeframe condition (e.g., 30).
        additional_conditions (str): Additional conditions for the WHERE clause (optional).
        ad_requests_threshold (int): Conditions for the HAVING clause (optional), minimal volume of ad_requests per inventory piece.
        query_executor (callable): A function to execute the query (e.g., `qm.clickhouse_query`).

    Returns:
        pd.DataFrame: Enriched dataframe.
    """

    # Construct the SELECT clause
    select_clause = ",\n    ".join(dimensions) + ",\n    SUM(imps) AS ad_requests,\n    SUM(biddable_imps) AS biddable_imps,\n    SUM(CASE WHEN is_delivered THEN imps ELSE 0 END) AS delivered_imps,\n    SUM(seller_revenue) / 1000 AS reseller_revenue"

    # Construct the GROUP BY clause
    group_by_clause = ",\n    ".join(dimensions)

    # Construct the WHERE clause
    where_conditions = [
        f"geo_country = '{market}'",
        f"DATE(ymdh) >= today() - INTERVAL {timeframe} DAY",
        "smg.seller_member_group_id != 2",
        "imp_type != 5",
        "deal_id = 0"
    ]
    where_clause = " AND\n    ".join(where_conditions)

    having_conditions = f" SUM(imps)>={ad_requests_threshold}"

    # Construct the final query
    query = f"""
    SELECT
        {select_clause}
    FROM {fact_table} a
    LEFT JOIN sup_common_member_seller_member_group smg 
        ON smg.member_id = a.seller_member_id
    WHERE
        {where_clause}
    GROUP BY
        {group_by_clause}
    HAVING
        {having_conditions}
    """

    # Execute the query and fetch the result
    aggregated_data = qm.clickhouse_query(query)

    # Step 2: Enrich the DataFrame
    enriched_df = aggregated_data.copy()

    if "seller_member_id" in aggregated_data.columns:
        seller_member_id_tuple = tuple(aggregated_data["seller_member_id"].unique())
        dim_member_data = qm.clickhouse_query(f"""
            SELECT 
                member_id AS seller_member_id,
                sudoku_row AS seller_sudoku_row
            FROM dim_member
            WHERE member_id IN {seller_member_id_tuple}
        """)
        enriched_df = enriched_df.merge(dim_member_data, how="left", on="seller_member_id")

    if "seller_member_group_id" in aggregated_data.columns:
        seller_member_group_id_tuple = tuple(aggregated_data["seller_member_group_id"].unique())
        common_seller_member_group_data = qm.mysql_query(f"""
            SELECT 
                id AS seller_member_group_id,
                name AS seller_member_group
            FROM common.seller_member_group
            WHERE id IN {seller_member_group_id_tuple}
        """)
        enriched_df = enriched_df.merge(common_seller_member_group_data, how="left", on="seller_member_group_id")

    if "tag_id" in aggregated_data.columns:
        tag_id_tuple = tuple(aggregated_data["tag_id"].unique())
        api_tinytag_data = qm.mysql_query(f"""
            SELECT 
                id AS tag_id,
                position AS tag_position,
                media_subtypes AS tag_media_subtypes,
                is_resizable AS tag_is_resizable
            FROM api.tinytag
            WHERE id IN {tag_id_tuple}
        """)
        # Get the intersection of the columns in api_tinytag_data that are present in both dimensions and features
        columns_to_merge = [col for col in api_tinytag_data.columns if col in dimensions or col in features]
        enriched_df = enriched_df.merge(api_tinytag_data[columns_to_merge], how="left", on="tag_id")

    if "inventory_url_id" in aggregated_data.columns:
        inventory_url_id_tuple = tuple(aggregated_data["inventory_url_id"].unique())
        api_inventory_url_data = qm.mysql_query(f"""
            SELECT 
                id AS inventory_url_id,
                content_category_id AS content_category_id
            FROM api.inventory_url
            WHERE id IN {inventory_url_id_tuple}
        """)
        enriched_df = enriched_df.merge(api_inventory_url_data, how="left", on="inventory_url_id")

    if "content_category_id" in enriched_df.columns:
        def categorize_content_category(content_category_id):
            categories = {
                'Sports': {8556, 8362, 29, 53645, 53614, 8560, 8562, 8570, 8532, 8568, 8542, 53757, 8574, 8566, 8544, 8526, 8548, 8524, 8564, 8538, 8576, 8550, 8536, 8572, 8530, 8540},
                'Gaming': {8206, 53567, 8202, 13, 64681, 64245, 64250, 64246, 64248, 8212, 8198, 64241, 8508, 64254, 64238, 64244, 64243, 64247, 64256, 64678, 64683, 64251, 64679, 8144, 8208, 8194},
                'News': {20, 19, 8368, 7992, 8348, 8350, 64252, 8364, 53587, 8356, 8354, 53672, 53610, 54169},
                'Tech': {8330, 8048, 8444, 8142, 8458, 9, 8452, 8138, 8202, 30, 8140, 17, 8148, 8154},
                'Auto': {8096, 53680, 6, 64718, 8040, 8022, 8024, 8032, 8026, 8046, 8044, 8028, 8042, 8520, 8584},
                'Finance': {11, 53593, 53648},
                'Business': {8350, 8, 53577, 8104, 8128, 8116, 53608, 8098, 8102},
                'Travel': {31, 53584, 8600, 8578, 8598, 64682, 8590, 53604},
                'Healthcare': {14, 8220, 8354, 8222, 8246, 8248, 8224, 8228, 8462, 8226, 8230},
                'Soccer': {8564, 8362, 29},
                'Luxury': {8490},
                'Food and Drink': {12, 53621, 64253, 8186, 53659, 8182, 8190, 8178, 8188},
                'Shopping': {27, 53606, 53633, 8496, 8512},
                'Cycling': {8538, 8016, 8362, 29},
                'Basketball': {8526, 8362, 29},
                'Entertainment': {10, 7996, 53569, 53609},
                'Hockey': {8548},
                'Job': {8478, 8118, 8150},
                'Music': {8004, 53578, 64240, 8508, 53607, 64679, 54279},
                'Beauty and Fashion': {7, 8504, 53677, 8054, 8070, 8408, 8064},
                'Lifestyle': {53594, 18, 53568},
                'Health and Fitness': {8254, 8222, 8234, 64239, 8246, 8504, 8232, 53603, 14, 8354, 53588, 8066},
            }
            for category, ids in categories.items():
                if content_category_id in ids:
                    return category
            return 'Other'

        enriched_df['content_category'] = enriched_df['content_category_id'].apply(categorize_content_category)
        enriched_df = enriched_df.drop(columns=['content_category_id'])

    if "media_type" in enriched_df.columns:
        def categorize_media_type(media_type):
            if media_type in {1, 3}:  
                return "Banner"
            elif media_type in {7, 8, 23, 64}:
                return 'Video'
            elif media_type in {65, 66}:
                return 'Native'
            elif media_type in {67, 68, 69, 70}:
                return 'Audio'
            else:
                return 'Other'

        enriched_df['format'] = aggregated_data['media_type'].apply(categorize_media_type)
        enriched_df = enriched_df.drop(columns=['media_type'])

    # Drop all columns ending with '_id'
    columns_to_drop = [col for col in enriched_df.columns if col.endswith('_id')]
    enriched_df = enriched_df.drop(columns=columns_to_drop)

    # Handle NaN values in groupby columns based on their data type
    group_columns = [col for col in enriched_df.columns if col not in ['ad_requests', 'biddable_imps', 'delivered_imps', 'reseller_revenue']]
    
    for col in group_columns:
        if enriched_df[col].dtype == 'object':  # String columns
            enriched_df[col] = enriched_df[col].fillna('None')
        elif pd.api.types.is_numeric_dtype(enriched_df[col]):  # Numeric columns
            enriched_df[col] = enriched_df[col].fillna(0)

    # Ensure NaN values in the aggregation columns are handled (optional: fill with 0 for summing)
    enriched_df['ad_requests'] = enriched_df['ad_requests'].fillna(0)
    enriched_df['biddable_imps'] = enriched_df['biddable_imps'].fillna(0)
    enriched_df['delivered_imps'] = enriched_df['delivered_imps'].fillna(0)
    enriched_df['reseller_revenue'] = enriched_df['reseller_revenue'].fillna(0)

    # Perform a group by operation and sum the relevant columns
    enriched_df = enriched_df.groupby(group_columns, as_index=False).agg({
        'ad_requests': 'sum',
        'biddable_imps': 'sum',
        'delivered_imps': 'sum',
        'reseller_revenue': 'sum'
    })

    return enriched_df, aggregated_data

def analyze_and_preprocess(dataset, target_column, scope=None):
    # Step 1: Apply scope filter if provided
    if scope:
        dataset = dataset.query(scope)

    columns_to_analyze = [col for col in dataset.columns if col not in ['ad_requests', 'biddable_imps', 'delivered_imps', 'reseller_revenue']]

    # Step 2: Group by the specified columns before analysis and target calculation
    grouped_dataset = dataset.groupby(columns_to_analyze).agg({
        "reseller_revenue": 'sum',
        "ad_requests": 'sum',
        "biddable_imps": 'sum',
        "delivered_imps": 'sum'
    }).reset_index()

    # Ensure that all relevant columns are converted to float to avoid issues with decimal.Decimal
    numeric_columns = ['reseller_revenue', 'ad_requests', 'biddable_imps', 'delivered_imps']
    for col in numeric_columns:
        grouped_dataset[col] = grouped_dataset[col].apply(lambda x: float(x) if isinstance(x, decimal.Decimal) else x)

    # Step 3: Analyze dataset
    analysis_results = {}
    for column in columns_to_analyze:
        column_data = grouped_dataset[column]
        column_analysis = {
            "datatype": column_data.dtype,
            "cardinality": column_data.nunique(),
            "null_percentage": column_data.isnull().mean() * 100
        }
        if pd.api.types.is_numeric_dtype(column_data):
            column_analysis["min"] = column_data.min()
            column_analysis["max"] = column_data.max()
            column_analysis["mean"] = column_data.mean()
        else:
            column_data = column_data.astype(str)
            column_analysis["min"] = column_data.min()
            column_analysis["max"] = column_data.max()
            column_analysis["mean"] = None
        analysis_results[column] = column_analysis

    analysis_df = pd.DataFrame(analysis_results).T
    #print("Column Analysis:")
    #print(analysis_df)

    # Step 4: Add calculated target columns
    grouped_dataset["sRPM"] = 1000000 * grouped_dataset["reseller_revenue"] / grouped_dataset["ad_requests"]
    grouped_dataset["esRPM"] = 1000000 * grouped_dataset["reseller_revenue"] / (
        grouped_dataset["delivered_imps"] +
        (grouped_dataset["biddable_imps"] - grouped_dataset["delivered_imps"]) * 0.5 +
        (grouped_dataset["ad_requests"] - grouped_dataset["biddable_imps"]) * 0.1
    )
    if target_column=='sRPM':
        overall_target = 1000000 * grouped_dataset["reseller_revenue"].sum() / grouped_dataset["ad_requests"].sum()
    elif target_column=='esRPM':
        overall_target = 1000000 * grouped_dataset["reseller_revenue"].sum() / (
        grouped_dataset["delivered_imps"].sum() +
        (grouped_dataset["biddable_imps"].sum() - grouped_dataset["delivered_imps"].sum()) * 0.5 +
        (grouped_dataset["ad_requests"].sum() - grouped_dataset["biddable_imps"].sum()) * 0.1
    )

    # Step 5: Preprocessing setup
    categorical_cols = [col for col in columns_to_analyze if grouped_dataset[col].dtype == 'object' or grouped_dataset[col].dtype == 'string']
    numeric_cols = [col for col in columns_to_analyze if pd.api.types.is_numeric_dtype(grouped_dataset[col])]

    categorical_transformer = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_transformer = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler()) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[ 
            ('cat', categorical_transformer, categorical_cols), 
            ('num', numeric_transformer, numeric_cols),
        ]
    )

    # Step 6: Apply preprocessing
    preprocessed_data = preprocessor.fit_transform(grouped_dataset)

    # Convert to DataFrame
    if isinstance(preprocessed_data, np.ndarray):
        dense_data = preprocessed_data
    else:
        dense_data = preprocessed_data.toarray()

    categorical_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    all_feature_names = list(categorical_feature_names) + numeric_cols

    preprocessed_df = pd.DataFrame(dense_data, columns=all_feature_names)
    preprocessed_df.index = grouped_dataset.index
    preprocessed_df[target_column] = grouped_dataset[target_column].values

    # Step 7: Model training and evaluation
    X = preprocessed_df.drop(columns=[target_column])
    y = preprocessed_df[target_column]

    print(preprocessed_df.shape)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_cv_pred = model.predict(X_cv)
    cv_rmse = np.sqrt(mean_squared_error(y_cv, y_cv_pred))
    cv_r_squared = r2_score(y_cv, y_cv_pred)
    cv_mape = np.mean(np.abs((y_cv - y_cv_pred) / y_cv.mean())) * 100
    print(f"Cross-validation RMSE: {cv_rmse}")
    print(f"Cross-validation R-squared: {cv_r_squared:.4f}")
    print(f"Cross-validation MAPE: {cv_mape:.2f}%")

    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r_squared = r2_score(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test.mean())) * 100
    print(f"Test RMSE: {test_rmse}")
    print(f"Test R-squared: {test_r_squared:.4f}")
    print(f"Test MAPE: {test_mape:.2f}%")

    test_results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_test_pred,
        'Error': y_test - y_test_pred
    }, index=y_test.index)

    dataset_with_results = grouped_dataset.copy()
    dataset_with_results['Predicted'] = np.nan
    dataset_with_results.loc[test_results.index, 'Predicted'] = test_results['Predicted']

    grouped_dataset_with_results = dataset_with_results.groupby(categorical_cols + numeric_cols).agg({
        target_column: 'mean',
        'Predicted': 'mean'
    }).reset_index()

    #print("Grouped Dataset with Results:")
    #print(grouped_dataset_with_results)

    return analysis_df, overall_target, preprocessed_df, grouped_dataset_with_results, cv_rmse, cv_mape, cv_r_squared, test_rmse, test_mape, test_r_squared


# Button to trigger the chain of functions
if st.button("Train Model"):

    # Fetch the dataset
    dataset, aggregated_data = fetch_and_enrich_data(fact_table, dimensions=dimensions, market=market, timeframe=days)

    # Preprocess the dataset
    analysis_df, overall_target, preprocessed_df, grouped_dataset_with_results, cv_rmse, cv_mape, cv_r_squared, test_rmse, test_mape, test_r_squared = analyze_and_preprocess(dataset, target_column, scope=scope)

    # Display Performance Metrics
    st.success("Model evaluation completed! Challenge yourself to improve the performance further.")
    
    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Cross-validation RMSE", f"{cv_rmse:.2f}", delta=None, delta_color="normal")
    col2.metric("Cross-validation MAPE", f"{cv_mape:.2f}%", delta=None, delta_color="normal")
    col3.metric("Cross-validation R-squared", f"{cv_r_squared:.4f}", delta=None, delta_color="normal")

    col1, col2, col3 = st.columns(3)

    col1.metric("Test RMSE", f"{test_rmse:.2f}", delta=None, delta_color="normal")
    col2.metric("Test MAPE", f"{test_mape:.2f}%", delta=None, delta_color="normal")
    col3.metric("Test R-squared", f"{test_r_squared:.4f}", delta=None, delta_color="normal")
    
    st.write("### Features Details")
    st.write(analysis_df.head())

    # Target column details
    st.write("### Target Details")
    if target_column in preprocessed_df.columns:
        st.write(f"Overall: {overall_target}")
        st.write(f"Min: {preprocessed_df[target_column].min()}")
        st.write(f"Max: {preprocessed_df[target_column].max()}")
    else:
        st.write(f"Target column '{target_column}' not found in the preprocessed data.")

    st.write("### Dataset Sample")
    st.write(dataset.head(10))

    st.write("### Training Dataset Sample")
    st.write(preprocessed_df.head(10))

    st.write(f"### Training Dataset Shape: {preprocessed_df.shape}")

    st.write("### Test Results Sample")
    st.write(grouped_dataset_with_results.head(1000))