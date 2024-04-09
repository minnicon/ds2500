import statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import random


FILE = "Indicators_of_Anxiety_or_Depression_Based_on_Reported_Frequency_of_Symptoms_During_Last_7_Days.csv"

# read in csv, clean data: organize by age and by symptoms of depressive 
# disorder, anxiety disorder, or both 

# analyze difference between years 2020-2021 (??)
# covid years were 2020-2021 so analyze differences between 2020-2021 and 2022-2023
# use "value" column

# use linear regression, find correlation between age and anxiety/depression levels

# knn classification - predict and classify anxiety/depression levels in 2024, 
# and can compare it with actual data from first two months of 2024

def clean_data(df):
    '''
    Take in a dataframe and sort by age, clean data to so that 
    'Time Period' only includes the year and not the entire date.
    Additionally, remove any rows with NaN values in critical columns.
    '''
    # Convert 'Time Period Start Date' to datetime and extract the year
    df['Year'] = pd.to_datetime(df['Time Period Start Date']).dt.year

    # Define a mapping from age ranges to a sortable numerical value
    age_sort_order = {
        '18 - 29 years': 1,
        '30 - 39 years': 2,
        '40 - 49 years': 3,
        '50 - 59 years': 4,
        '60 - 69 years': 5,
        '70 - 79 years': 6,
        '80 years and above': 7
    }

    # Map 'Subgroup' to the sort order and create a new column for sorting
    df['Age Sort'] = df['Subgroup'].map(age_sort_order)

    # Ensure data related to age is selected and sorted
    cleaned_df = df[df['Group'] == 'By Age'].sort_values(by=['Age Sort', 'Year'])

    # Drop the 'Age Sort' column as it's no longer needed after sorting
    cleaned_df = cleaned_df.drop(columns=['Age Sort'])

    # Remove rows with NaN values in the 'Value' column to avoid errors in analysis
    cleaned_df = cleaned_df.dropna(subset=['Value'])

    # Convert age groups to numerical values
    cleaned_df = modify_age(cleaned_df)

    return cleaned_df


def prepare_features_labels(df, exclude_year=2024):
    """
    Prepare features and labels for the classifier, removing any rows with NaN values and optionally excluding data for a specific year.
    """
    if exclude_year:
        df = df[df['Year'] != exclude_year]  # Exclude the specified year from the dataset

    features = df[['AgeEncoded', 'Year']]
    labels = df['Value']
    return features, labels

def modify_age(df):
    # Convert age groups to numerical values
    age_encoder = LabelEncoder()
    df['AgeEncoded'] = age_encoder.fit_transform(df['Subgroup'])
    return df
    
    return df_cleaned

def discretize_values(df, bins, labels):
    """
    Discretize the 'Value' column into categories.
    Args:
    - df: DataFrame containing the 'Value' column.
    - bins: A list of bin edges for categorization.
    - labels: A list of labels for the bins.
    
    Returns:
    - DataFrame with the 'Value' column replaced with discretized categories.
    """
    df['Value'] = pd.cut(df['Value'], bins=bins, labels=labels, include_lowest=True)
    return df

def classifier(df):
    """
    Train a KNN classifier to predict categorized anxiety/depression levels based on age groups and year.
    """
    
    # Prepare features and labels
    features, labels = prepare_features_labels(df)
    
    # Encode labels if they are categorical
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
    
    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)
    
    # Predictions
    predictions = knn.predict(X_test)
    
    # Evaluation
    print(confusion_matrix(Y_test, predictions))
    print(classification_report(Y_test, predictions))
    
    return knn, le  # Return the trained model and label encoder for further use
    
def mean_within_age_group(df, year):
    ''' Calculate mean for each age subgroup in the dataset for given year
    '''
    means = []
    subgroups = df['AgeEncoded'].unique()
    # drop NaN values
    df = df.dropna(subset=['Value'])
    for age in subgroups:
        subset = df[(df['AgeEncoded'] == age) & (df['Year'] == year)]
        mean = statistics.mean(subset['Value'])
        means.append(mean)
    return means, subgroups

def compare_periods_mean(df):
    '''
    Compares the mean anxiety and depression levels between 2020-2021 and 2022-2023.
    Args:
    - df: Cleaned DataFrame with at least 'Year' and 'Value' columns.

    Returns:
    - A dictionary with mean values for each period and a comparison result.
    '''
    # Filter the DataFrame for the specified years
    period1_df = df[df['Year'].isin([2020, 2021])]
    period2_df = df[df['Year'].isin([2022, 2023])]

    # Calculate the mean values for each period
    mean_period1 = period1_df['Value'].mean()
    mean_period2 = period2_df['Value'].mean()

    # Compare the means and determine the trend
    trend = "increased" if mean_period2 > mean_period1 else "decreased"

    return {
        "2020-2021 Mean": mean_period1,
        "2022-2023 Mean": mean_period2,
        "Trend": trend
    }

def mean_median_per_year(df):
    ''' Calculate mean and median values for each year (2020-2024)
    '''
    means = []
    medians = []
    years = df['Year'].unique()
    # drop NaN values
    df = df.dropna(subset=['Value'])
    for year in years:
        subset = df[df['Year'] == year]
        mean = statistics.mean(subset['Value'])
        median = statistics.median(subset['Value'])
        means.append(mean)
        medians.append(median)
    return means, medians, years
    
def analysis_2024(actual, predictions, label_encoder):
    """
    Compares predicted values for 2024 against actual values.
    Calculates and prints the accuracy of the predictions.
    
    Args:
    - actual: Actual labels for 2024.
    - predictions: Predicted labels for 2024.
    - label_encoder: LabelEncoder object used for encoding labels.
    
    Returns:
    - Accuracy of the predictions.
    """
    # Decode labels to compare accurately
    actual_decoded = label_encoder.inverse_transform(actual)
    predictions_decoded = label_encoder.inverse_transform(predictions)
    
    # Calculate accuracy
    accuracy = np.mean(actual_decoded == predictions_decoded)
    print(f"Accuracy of 2024 predictions: {accuracy*100:.2f}%")
    
    return accuracy

def linear_regression(df):
    """
    Perform enhanced multi-linear regression to understand how anxiety and depression levels 
    are influenced by age groups across the years 2020-2023.
    """
    # Prepare features (Age group and Year) and labels (Value)
    # Make sure 'Year' and 'AgeEncoded' are used as features to capture the multi-linear regression model
    features, labels = df[['AgeEncoded', 'Year']], df['Value']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Evaluation
    print(f"Mean squared error: {round(mean_squared_error(y_test, predictions),4)}")
    print(f"R-squared: {round(r2_score(y_test, predictions),4)}")
    
    # Plotting to visualize predictions against actual values
    # For plotting purposes, we'll average the predictions and actual values by Age Group and Year to make it interpretable.
    X_test['Predictions'] = predictions
    X_test['Actual'] = y_test
    grouped_pred = X_test.groupby(['AgeEncoded', 'Year'])['Predictions'].mean().reset_index()
    grouped_actual = X_test.groupby(['AgeEncoded', 'Year'])['Actual'].mean().reset_index()
    
    # We plot the actual vs predicted values for a visual comparison
    plt.figure(figsize=(10, 6))
    subgroups = df['Subgroup'].unique()
    color = ["purple", "blue", "orange", "red", "magenta", "green", "brown"]
    for age in sorted(df['AgeEncoded'].unique()):
        age_pred = grouped_pred[grouped_pred['AgeEncoded'] == age]
        age_actual = grouped_actual[grouped_actual['AgeEncoded'] == age]
        plt.plot(age_pred['Year'], age_pred['Predictions'], color=color[age], marker='o', linestyle='-', label=f'Pred Age: {subgroups[age]}')
        plt.plot(age_actual['Year'], age_actual['Actual'], color=color[age], marker='x', linestyle='--', label=f'Actual Age')
    
    plt.xlabel('Year')
    plt.ylabel('Anxiety/Depression Levels')
    plt.title('Predicted vs Actual Anxiety/Depression Levels by Age Group and Year')
    plt.xticks(range(min(age_pred['Year'].unique()), max(age_pred['Year'].unique())+2, 1))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
        
def compare_years(df):
    '''
    analyze difference between years 2020-2021 
    covid years were 2020-2021 so analyze differences between 2020-2021 and 2022-2023
    use "value" column
    '''
    # correlation (btw age groups and anxiety/depression levels), mean, sd, and variance in each year
    years = df['Year'].unique()
    stats = {}
    for year in years:
        mean, subgroups = mean_within_age_group(df, year)
        corr = statistics.correlation(subgroups, mean)
        sd = statistics.stdev(mean)
        var = statistics.variance(mean)
        stats[year] = [corr, sd, var]
    return stats

def correlation_matrix(df):
    ''' 
    Create an enhanced correlation matrix to see which age group is most correlated 
    with anxiety or depression across all years. 
    '''
    # Prepare the data: 'Value' as the target variable and 'AgeEncoded' for age groups
    # Ensure to include data from all years
    data_for_corr = df[['Value', 'AgeEncoded']]
    subgroups = df['Subgroup'].unique()
    
    # One-hot encode 'AgeEncoded' to get separate columns for each age group
    age_dummies = pd.get_dummies(data_for_corr['AgeEncoded'], prefix='AgeGroup')
    data_for_corr = pd.concat([data_for_corr[['Value']], age_dummies], axis=1)
    
    # Calculate the correlation matrix
    correlation_matrix = data_for_corr.corr()
    
    # Save list of xlabels and ylabels for plotting
    labels = ['Value']
    labels.extend(subgroups[i] for i in range(len(subgroups)))
        
    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, xticklabels=labels, yticklabels=labels)
    plt.title("Correlation Matrix for Age Groups and Anxiety/Depression Levels")
    plt.show()

    
def change_over_time(df):
    ''' Show change in median and average age over time 
    '''
    mean, median, years = mean_median_per_year(df)
    plt.plot(years, mean, color="blue", label="Mean")
    plt.plot(years, median, color="magenta", label="Median")
    plt.ylabel("Anxiety/Depression Levels")
    plt.xticks(range(min(years), max(years)+1, 1))
    plt.xlabel("Year")
    plt.title("Median and Mean Anxiety/Depression Levels Over Time")
    plt.legend()
    plt.show()
    

def main():
    # Assuming 'FILE' is the path to your CSV file
    data = pd.read_csv(FILE)
    
    # Clean and prepare the data
    cleaned_data = clean_data(data)
    
    # Linear regression analysis
    linear_regression(cleaned_data)
    
    # Compare difference in stats in each year
    stats_dct = compare_years(cleaned_data)
    for key in stats_dct: 
        print(f"Correlation between age groups and mean anxiety depression levels in {key}: {round(stats_dct[key][0],4)}") 
        print(f"Standard Deviation of mean anxiety/depression levels: {round(stats_dct[key][1],4)}")
        print(f"Variance of mean anxiety/depression levels: {round(stats_dct[key][2],4)}")
        
    # Create a correlation matrix for age group and anxiety/depression levels
    corr_matrix = correlation_matrix(cleaned_data)
    
    # Compare mean and median anxiety/depression levels in each year of the dataset (plot)
    median_mean_per_year = change_over_time(cleaned_data)
    
    # New: Compare mean anxiety and depression levels between 2020-2021 and 2022-2023
    period_comparison = compare_periods_mean(cleaned_data)
    print(f"Mean anxiety/depression levels 2020-2021: {round(period_comparison['2020-2021 Mean'],4)}")
    print(f"Mean anxiety/depression levels 2022-2023: {round(period_comparison['2022-2023 Mean'],4)}")
    print(f"Trend from 2020-2021 to 2022-2023: {period_comparison['Trend']}")
    

if __name__ == "__main__":
    main()
