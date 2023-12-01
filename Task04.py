
import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.spatial import distance

def NormalizeData(data):            # function Min Max Normalizes the data
    normalizedData = []
    dataMax = max(data)
    dataMin = min(data)
    
    for x in data:
        normalizedData.append((x-dataMin)/(dataMax-dataMin))
    return normalizedData

def MahalanobisDistance(x, xi, inverseCovariance):          # Subtask A: design a distance function                
    mahalanobisDistance = distance.mahalanobis(x, xi, VI=inverseCovariance)
    return mahalanobisDistance

def OutlierDetection(data, alpha):                          # Subtask B: Distance Based Outtlier Detection
    # Find average, and inverse covariance to calculate Mahalanobis distance
    averageDay = np.mean(data, axis = 0)
    covariance = np.cov(data, rowvar=False)
    inverseCovariance = np.linalg.inv(covariance)

    #calculate the distance from every day in the data to the average day
    distances = []
    for index, row  in data.iterrows():
        mahalanobisDistance = MahalanobisDistance(row, averageDay, inverseCovariance)
        distances.append(mahalanobisDistance)

    # determine the chi squared threshold by using alpha 
    chiSqrThreshold = chi2.ppf(1-alpha, df=len(data.columns))

    # calculate the outlier score for a day by dividing its distance from average day by the chi squared threshold
    outlierScores = []
    for distance in distances:
        outlierScore = distance / chiSqrThreshold
        outlierScores.append(outlierScore)
    
    return outlierScores

def main():
    # read in the data
    weatherData = pd.read_csv("Basel_Weather.csv")

    labels = ['precipitation', 'sunshine', 'temp_mean', 'humidity']
    normalUsefulData = pd.DataFrame()

    # separate the data we want to make our outlier detection on and then normalize the data
    for label in labels:
        normalUsefulData[label] = NormalizeData(weatherData[label])

    # Subtask C: apply outlier detection to the dat with three diff hyperparameter
    alphas = [.01, .05, .1]
    for alpha in alphas:
        outlierScores = OutlierDetection(normalUsefulData, alpha)
        outlierPDArray = pd.DataFrame({"OLS": outlierScores})

        outputLabels = ['DATE', 'precipitation', 'sunshine', 'temp_mean', 'humidity']
        outputData = weatherData[outputLabels]
        dataWithOutlierScores = pd.concat([outputData, outlierPDArray], axis=1)
        
        # Subtask D: sort the data by OLS
        sortedData = dataWithOutlierScores.sort_values(by="OLS", ascending=False)
        topOutliers = sortedData.head(6)
        bottomOutliers = sortedData.tail(2)

        # Print the top 6 and bottom 2
        print("Results for Hyperparameter Setting Alpha = " + str(alpha))
        print("Top 6 Outliers:")
        print(str(topOutliers))
        print("Bottom 2 Normal:")
        print(str(bottomOutliers))
        print("\n")
    

main()

