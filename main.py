import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

HOUSE_PRICE_PATH = "A2_Data/House_Price.csv"
FIGURE_SAVE_PATH = "figures/figure"

def fileName(name:str) -> str:
    return FIGURE_SAVE_PATH + "_" + name + ".png"


# strategy decides where on the x axis the bins will be (like wat range)
# strategy = quantile, kmeans (in 1d, with n_bins centroids)
# KBinsDiscretizer(n_bin=7, encode='ordinal', strategy='quantile')

def printInfo():
    data = pd.read_csv(HOUSE_PRICE_PATH)

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



if __name__ == "__main__":
    printInfo()