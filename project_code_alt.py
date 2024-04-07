import statistics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


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
    '''
    # Correct: Use df instead of data
    df['Year'] = pd.to_datetime(df['Time Period Start Date']).dt.year

    # Define a mapping from age ranges to a sortable numerical value based on the minimum age
    age_sort_order = {
        '18 - 29 years': 1,
        '30 - 39 years': 2,
        '40 - 49 years': 3,
        '50 - 59 years': 4,
        '60 - 69 years': 5,
        '70 - 79 years': 6,
        '80 years and above': 7
    }

    # Map the 'Subgroup' to the sort order and create a new column for sorting
    df['Age Sort'] = df['Subgroup'].map(age_sort_order)

    # Filter for age-related data and sort by our new 'Age Sort' column, then by 'Year'
    cleaned_df = df[df['Group'] == 'By Age'].sort_values(by=['Age Sort', 'Year'])

    # Drop the 'Age Sort' column as it's no longer needed after sorting
    cleaned_df = cleaned_df.drop(columns=['Age Sort'])

    return cleaned_df

def prepare_features_labels(df):
    """
    Prepare features and labels for the classifier, removing any rows with NaN values.
    """
    # Drop rows with NaN values in the 'Value' column or in any column used as features
    df_cleaned = df.dropna(subset=['Value', 'Subgroup', 'Year'])
    
    # Convert age groups to numerical values
    age_encoder = LabelEncoder()
    df_cleaned['AgeEncoded'] = age_encoder.fit_transform(df_cleaned['Subgroup'])
    features = df_cleaned[['AgeEncoded', 'Year']]  # Using 'Year' as an additional feature
    labels = df_cleaned['Value']
    
    return features, labels


def classifier(df):
    """
    Train a KNN classifier to predict 'Value' based on age groups and year.
    """
    # Prepare features and labels
    features, labels = prepare_features_labels(df)
    
    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Initialize and train classifier
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)
    
    # Predictions and evaluation
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, zero_division=1)
    
    return report
    
def mean_within_category(df):
    subgroups = df['Subgroup'].unique()
    means = []
    for age in subgroups:
        subset = df[df['Subgroup'] == age]
        mean = statistics.mean(subset['Value'])
        means.append(mean)
    return means, subgroups
    
def analysis_2020(df):
    ''' Find correlation between age groups and mean anxiety/depression levels in 2020
        For each age subgroup, calculate mean
    '''
    mean, subgroups = mean_within_category(df)
    correlation = statistics.correlation(subgroups, mean)
    sd = statistics.stdev(mean)
    var = statistics.variance(mean)
    return correlation, sd, var
    
def analysis_2024(true_preds, year_pred):
    ''' Find variance and sd calculations comparing predicted value in 2024 to actual values
    '''
    return var, sd

def linear_regression():
    ''' Perform linear regression on dataset and plot
    '''
    pass

def compare_years():
    '''
    analyze difference between years 2020-2021 (??)
    covid years were 2020-2021 so analyze differences between 2020-2021 and 2022-2023
    use "value" column
    '''
    pass

def main():
    # Assuming 'FILE' is the path to your CSV file
    data = pd.read_csv(FILE)
    
    # Clean and prepare the data
    cleaned_data = clean_data(data)
    
    # Run the classifier and get the classification report
    classifier_report = classifier(cleaned_data)
    
    # Print the classification report to evaluate the model's performance
    print(classifier_report)
    
    # predict values for 2024
    # report, pred, true_pred = classifier(cleaned_data, cleaned_data)
    
    # standard deviation and variance analysis for predicted 2024 values
    # var_2024, sd_2024 = analysis_2024(true_pred, pred)
    
    # Find correlation between age groups and mean anxiety/depression levels in 2020
    # Include standard deviation and variance analysis calculations
    # corr_2020, sd_2020, var_2020 = analysis_2020(cleaned_data)
    
if __name__ == "__main__":
    main()
    
