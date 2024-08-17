import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np

HOUSE_PRICE_PATH = "A2_Data/unprocessed/House_Price.csv"
FIGURE_SAVE_PATH = "figures/figure"

def fileName(name:str) -> str:
    return FIGURE_SAVE_PATH + "_" + name + ".png"


# strategy decides where on the x axis the bins will be (like wat range)
# strategy = quantile, kmeans (in 1d, with n_bins centroids)
# KBinsDiscretizer(n_bin=7, encode='ordinal', strategy='quantile')

def summaryStats(data:pd.DataFrame):
    numFeatures = data.columns.size
    numInstances = data.shape[0]
    numCategorical = data.select_dtypes(include=['object', 'category']).shape[1]
    numNumerical = data.select_dtypes(include=['float64', 'int64']).shape[1]

    return pd.DataFrame({
        'Metrics': ['Number of Features', 'Number of Instances', 'Number of Categorical Features', 'Number of Numerical Features'],
        'Count': [numFeatures, numInstances, numCategorical, numNumerical]
    })


def analyseFeatureDistributions(data:pd.DataFrame, topFeatures):
    results = []

    for feature in topFeatures:
        # determine the number of bins, 
        # bw <- 2 * IQR(x) / length(x)^(1/3)
        # numbOfBins = (max-min)/bw
        bw = 2 * (data[feature].quantile(0.75) - data[feature].quantile(0.25)) / data[feature].size**(1/3)
        numBins = int(np.ceil((data[feature].max() - data[feature].min()) / bw))

        # plot histogram with n bins (x=feature, y= salePrice)
        plt.figure(figsize=(10, 6))
        sns.histplot(data[feature], bins=numBins, binwidth=bw) #, kde=True (kde is the line)
        plt.title('Distribution of ' + feature)
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # save to file
        plt.savefig(fileName(feature + "_hist"))  

        # the shape of their distributions with skewness and kurtosis 
        # (use Scipy for obtaining skewness and kurtosis values)

        skewness = skew(data[feature].dropna())
        kurt = kurtosis(data[feature].dropna())

        results.append({
            'Feature': feature,
            'Skewness': skewness,
            'Kurtosis': kurt
        })

    return pd.DataFrame(results)


def analyseMissingValues(data:pd.DataFrame):
    missing_data = data.isnull().sum()
    missing_data = missing_data[missing_data > 0]

    percentage_missing = (missing_data/len(data)) * 100

    missing_sum = pd.DataFrame({
        "Values": missing_data,
        "Percentage": percentage_missing
    })

    sorted = missing_sum.sort_values(by="Percentage", ascending=False)
    return sorted


def hierarchicalClustering(data: pd.DataFrame, top_features: list):
    numeric_data = data[top_features]
    numeric_data = numeric_data.fillna(numeric_data.mean())
    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).fillna(numeric_data.mean())
    
    # Normalize the data
    numeric_data = (numeric_data - numeric_data.mean()) / numeric_data.std()

    # Aggregate data by neighborhood
    aggregated_data = numeric_data.groupby(data['Neighborhood']).mean()
    
    # Check unique neighborhoods
    unique_neighborhoods = data['Neighborhood'].unique()
    print("Unique Neighborhoods: " + str(unique_neighborhoods.size))
    
    # Perform hierarchical clustering using Ward's method
    Z = linkage(aggregated_data, method='ward')
    
    # Plot the dendrogram with neighborhood labels
    plt.figure(figsize=(15, 10))
    dendrogram(Z, labels=aggregated_data.index, leaf_rotation=90, leaf_font_size=10)
    plt.title('Dendrogram of House Prices by Neighborhood')
    plt.xlabel('Neighborhood')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.savefig(fileName("dendrogram"))

def findTopCorrelations(data: pd.DataFrame, target: str, n: int):
    # Select numerical features
    numerical_features = data.select_dtypes(include=['float64', 'int64'])
    
    # Calculate Pearson correlation with the target variable
    correlation_matrix = numerical_features.corr()
    correlations = correlation_matrix[target].drop(labels=[target])
    
    # Get the top N features with the highest correlation
    top_features = correlations.abs().sort_values(ascending=False).head(n)
    
    # Create a DataFrame to display the results
    top_features_df = pd.DataFrame(top_features).reset_index()
    top_features_df.columns = ['Feature', 'Correlation with ' + target]
    
    return top_features_df


def printResultsForQuestion(results, question):
    print("Results for Question " + question)
    print(results)
    print("")
    print("")


if __name__ == "__main__":
    data = pd.read_csv(HOUSE_PRICE_PATH)

    # Question 1a
    stats = summaryStats(data)
    printResultsForQuestion(stats, "1a")
    
    # Question 1b
    topFeatures =  findTopCorrelations(data, "SalePrice", 5)
    printResultsForQuestion(topFeatures, "1b")
    top5NumericalFeatures = topFeatures["Feature"].tolist()

    # Question 1c
    featureDistrib = analyseFeatureDistributions(data, top5NumericalFeatures)
    printResultsForQuestion(featureDistrib, "1c")
    
    # Question 1d
    missingValues = analyseMissingValues(data)
    printResultsForQuestion(missingValues, "1d")

    # Question 3
    hierarchicalClustering(data, top5NumericalFeatures)

