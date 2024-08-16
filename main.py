import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

HOUSE_PRICE_PATH = "A2_Data/House_Price.csv"
FIGURE_SAVE_PATH = "figures/figure"

def fileName(name:str) -> str:
    return FIGURE_SAVE_PATH + "_" + name + ".png"


# strategy decides where on the x axis the bins will be (like wat range)
# strategy = quantile, kmeans (in 1d, with n_bins centroids)
# KBinsDiscretizer(n_bin=7, encode='ordinal', strategy='quantile')

def printInfo(data:pd.DataFrame):
    print("Num of Features: " + str(data.columns.size))

    maxCount = 0

    for column in list(data.columns):
        if data[column].size > maxCount: # count rows (get max row count)
            maxCount = data[column].size


    print("Num of Instances: " + str(maxCount))


def analyseFeatureDistributions(data:pd.DataFrame, topFeatures):

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

        print("Feature: " + feature)
        print("Skewness: " + str(skewness))
        print("Kurtosis: " + str(kurt))


def analyseMissingValues(data:pd.DataFrame):
    missing_data = data.isnull().sum()
    missing_data = missing_data[missing_data > 0]

    percentage_missing = (missing_data/len(data)) * 100

    missing_sum = pd.DataFrame({
        "Values": missing_data,
        "Percentage": percentage_missing
    })

    sorted = missing_sum.sort_values(by="Percentage", ascending=False)
    print(sorted)


def hierarchicalClustering(data:pd.DataFrame):
    data = data.drop(['SalePrice'])
    cleaned_data = data.select_dtypes(include=['float64'])

    linkMatrix = linkage(cleaned_data)

    plt.figure(figsize=(10, 6))
    d = dendrogram(linkMatrix)
    plt.title("fghfh")
    plt.show()


def make_graphs(data):
    # Set up the matplotlib figure for larger visualizations
    plt.figure(figsize=(10, 6))

    # Distribution of Sale Prices
    sns.histplot(data['SalePrice'], kde=True)
    plt.title('Distribution of House Sale Prices')
    plt.xlabel('Sale Price ($)')
    plt.ylabel('Frequency')
    plt.grid(True)
    #plt.show()
    plt.savefig(fileName("histogram"))

    # Correlation Matrix Heatmap
    plt.figure(figsize=(15, 10))
    numeric_data = data.select_dtypes(include=[float,int])
    corr_matrix = numeric_data.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of All Features')
    #plt.show()
    plt.savefig(fileName("correlation_matrix"))

    # Scatter plot of GrLivArea vs. SalePrice
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='GrLivArea', y='SalePrice', data=data)
    plt.title('Above Grade (Ground) Living Area vs. Sale Price')
    plt.xlabel('Above Grade Living Area (Square Feet)')
    plt.ylabel('Sale Price ($)')
    plt.grid(True)
    #plt.show()
    plt.savefig(fileName("scatter_plot"))

    # Box plot of Neighborhood vs. SalePrice
    plt.figure(figsize=(15, 10))
    sns.boxplot(x='Neighborhood', y='SalePrice', data=data)
    plt.title('Neighborhood vs. Sale Price')
    plt.xlabel('Neighborhood')
    plt.ylabel('Sale Price ($)')
    plt.xticks(rotation=90)
    plt.grid(True)
    #plt.show()
    plt.savefig(fileName("box_plot"))

    # Year Built vs. SalePrice
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='YearBuilt', y='SalePrice', data=data)
    plt.title('Year Built vs. Sale Price')
    plt.xlabel('Year Built')
    plt.ylabel('Sale Price ($)')
    plt.grid(True)
    #plt.show()
    plt.savefig(fileName("line_plot"))

    print("Saved graphs to " + fileName("x"))



if __name__ == "__main__":
    data = pd.read_csv(HOUSE_PRICE_PATH)

    # These are from orange
    top5NumericalFeatures = ["OverallQual","GrLivArea",
                             "GarageCars","GarageArea",
                             "TotalBsmtSF","SalePrice"] # saleperice in here jus for comparison
    
    # Question 1c
    analyseFeatureDistributions(data, top5NumericalFeatures)
    
    # Question 1d
    analyseMissingValues(data)

    # Question 3
    hierarchicalClustering(data)
    
    #make_graphs(data)
    printInfo(data)