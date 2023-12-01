import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

# SUBTASK A: 
# create a purity function
def Purity(a, b, areOutliersPresent):
    confusionMat = np.array([[sum((a == i) & (b == j)) for j in np.unique(b)] for i in np.unique(a)])
    totalElements = np.sum(confusionMat)
    numOutliers = 0
    
    # if outliers are present then count them
    if areOutliersPresent:
        for num in a:
            if num == 0:
                numOutliers +=1
    
    totalCounts = np.sum(np.max(confusionMat, axis=1))
    purity = totalCounts / totalElements
    
    if areOutliersPresent:
        outlierPercent = numOutliers / totalElements
        return round(purity, 2), round(outlierPercent, 2)
    else:
        return round(purity, 2)
    
def CreateScatterPlot(x, y, classifcation, labels):
    title = labels[0]
    xtitle = labels[1]
    ytitle=labels[2]
    plt.figure()
    scatterPlot = plt.scatter(x, y, c=classifcation, s=15)
    plt.colorbar(scatterPlot)
    plt.axis('equal')
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def RandomSearch(data, groundTruth, minPtsRange, epsilonRange, maxIterations):    
    minClusters = 2
    maxClusters = 17
    maxPurity = 0.0
    bestDBSCAN = None
    bestEpsilon = 0.0
    bestMinPoints = 0

    for i in range(maxIterations):
        iMinPoints = np.random.randint(minPtsRange[0], minPtsRange[1])
        iEpsilon = np.random.uniform(epsilonRange[0], epsilonRange[1])
        iDBSCAN = DBSCAN(min_samples=iMinPoints, eps=iEpsilon)

        iPrediction = iDBSCAN.fit_predict(data) + 1
        iPurity = Purity(iPrediction, groundTruth, True)
        iOutlier = iPurity[1]

        iClusters = len(set(iPrediction))-1

        if iPurity[0] > maxPurity and minClusters <= iClusters <=  maxClusters and iOutlier < 0.2 :
            maxPurity = iPurity[0]
            bestDBSCAN = iDBSCAN
            bestEpsilon = iEpsilon
            bestMinPoints = iMinPoints

    print("Best MinPts: " + str(bestMinPoints) + "\n")
    print("Best Epsilon: " + str(bestEpsilon) + "\n")
    return bestDBSCAN


def main():
    # Step 0: here we are setting up the data and printing a scatter plot so I can see the data
    print(" Subtask A \n")
    weatherData = pd.read_csv("Basel_Weather.csv")
    complex8Data = pd.read_csv("Complex8.csv", names=['X', 'Y', 'Cluster'])

    # I need to separate the data so its usable in the plot function
    complex8X = complex8Data['X'].values
    complex8Y = complex8Data['Y'].values
    complex8Cluster = complex8Data['Cluster'].values
    labels = ['Complex8 Data', 'a', 'b']
    CreateScatterPlot(complex8X, complex8Y, complex8Cluster, labels)
    
    humidityClass = weatherData["humidity_class"].values
    desiredWeatherCols = ["cloud_cover", "global_radiation", "precipitation", "sunshine", "temp_mean", "temp_min", "temp_max", "humidity"]
    adjustedWeatherData = weatherData[desiredWeatherCols]

    # changing humidity class from strings to corresponding integers
    for index in range(len(humidityClass)):
        if humidityClass[index] == "Low":
            humidityClass[index] = 1
        elif humidityClass[index] =="Mid":
            humidityClass[index] = 2
        elif humidityClass[index] == "High": 
            humidityClass[index] = 3
        else:
            humidityClass[index] = 0

    # here we are calling and printing the purity function
    print("Given Purity with Outliers: " + str(Purity(complex8Cluster, complex8Cluster, True)) + "\n")
    print("Given Purity without Outliers: " + str(Purity(complex8Cluster, complex8Cluster, False)) + "\n")
    
    # SUBTASK B: 
    print("\n Subtask B \n")
    
    # set up and run kmeans from the scikit-learn lib
    dataColumns = ['X', 'Y']
    kMeansData = complex8Data[dataColumns]
    
    # creating 4 kmeans object, fitting our data, then storing it 
    kMeans8First = KMeans(n_clusters=8, init="random", n_init=10,  max_iter=300, random_state=42)
    kMeans8Second = KMeans(n_clusters=8, init="random", n_init=10,  max_iter=300, random_state=99)
    kMeans12First = KMeans(n_clusters=12, init="random", n_init=10,  max_iter=300, random_state=42)
    kMeans12Second = KMeans(n_clusters=12, init="random", n_init=10,  max_iter=300, random_state=99)
    
    kMeans8First.fit(kMeansData)
    kMeans8Second.fit(kMeansData)
    kMeans12First.fit(kMeansData)
    kMeans12Second.fit(kMeansData)

    predicted8First = kMeans8First.labels_
    predicted8Second = kMeans8Second.labels_
    predicted12First = kMeans12First.labels_
    predicted12Second = kMeans12Second.labels_

    print8First = "K=8 First Round \nPurity with outliers: " + str(Purity(predicted8First, complex8Cluster, True)) +  ". Purity without Outliers: " + str(Purity(predicted12First, complex8Cluster, False)) + "\n"
    print8Second = "K=8 Second Round \nPurity with outliers: " + str(Purity(predicted8Second, complex8Cluster, True)) +  ". Purity without Outliers: " + str(Purity(predicted8Second, complex8Cluster, False)) + "\n"
    print12First = "K=12 First Round \nPurity with outliers: " + str(Purity(predicted12First, complex8Cluster, True)) +  ". Purity without Outliers: " + str(Purity(predicted12First, complex8Cluster, False)) + "\n"
    print12Second = "K=12 Second Round \nPurity with outliers: " + str(Purity(predicted12Second, complex8Cluster, True)) +  ". Purity without Outliers: " + str(Purity(predicted12Second, complex8Cluster, False)) + "\n"

    print(print8First) 
    print(print8Second) 
    print(print12First) 
    print(print12Second) 
    
    # put those plots up
    labels8First = ["K=8 First Iteration", "a", "b"]
    labess8Second = ["K=8 Second Iteration", "a", "b"]
    labels12First = ["K=12 First Iteration", "a", "b"]
    labels12Second = ["K=12 Second Iteration", "a", "b"]

    CreateScatterPlot(complex8X, complex8Y, predicted8First, labels8First)
    CreateScatterPlot(complex8X, complex8Y, predicted8Second, labess8Second)
    CreateScatterPlot(complex8X, complex8Y, predicted12First, labels12First)
    CreateScatterPlot(complex8X, complex8Y, predicted12Second, labels12Second)
    
    # SUBTASK C: Develop the search procedure 
    print("\n Subtask C \n")
    task3Data = np.hstack((complex8X.reshape(-1, 1), complex8Y.reshape(-1, 1)))

    minPointsRange = [2,4]
    epsilonRange = [13.6,15.0]

    print("MinPoints Range : " + str(minPointsRange) + "\n")
    print("Epsilon Range: " + str(epsilonRange) + "\n")
    
    bestComplexDBSCAN = RandomSearch(task3Data, complex8Cluster, minPointsRange, epsilonRange, 500)
    bestClassifications = bestComplexDBSCAN.fit_predict(task3Data)
    bestClassesOutlierAdjusted = bestClassifications + 1
    print("Number of Classes: " + str(max(bestClassesOutlierAdjusted)) + "\n")
    
    
    print("DBSCAN Purity with Outliers: " + str(Purity(bestClassesOutlierAdjusted, complex8Cluster, True)) + "\n")
    labelsComplex8Dbscan = ["DBSCAN", "a", "b"]
    CreateScatterPlot(complex8X, complex8Y, bestClassesOutlierAdjusted, labelsComplex8Dbscan)
    
    # SUBTASK D: Run Kmeans k=3 + k=5 for BWD
    print("\n Subtask D \n")

    # bwd  k=3 setup
    bwdK3 = KMeans(n_clusters=3, init="random", n_init=10,  max_iter=300, random_state=42)
    bwdK3.fit(adjustedWeatherData)
    predictedbwdK3 = bwdK3.labels_
    predictedbwdK3OutlierAdj = predictedbwdK3 + 1
    
    # bwd k=5 setup
    bwdK5 = KMeans(n_clusters=5, init="random", n_init=10,  max_iter=300, random_state=42)
    bwdK5.fit(adjustedWeatherData)
    predictedbwdK5 = bwdK5.labels_
    predictedbwdK5OutlierAdj = predictedbwdK5 + 1

    # I had trouble with plotting humidty so multiplying by 100 turns humidity into a percent and better for scaling the graph
    adjustedHumidityValues = adjustedWeatherData["humidity"].values * 100.0

    labelsBwdData = ["BWD Data", "humidity percent", "temp_mean"]
    labelsBwdk3 = ["BWD K=3", "humidity percent", "temp_mean"]
    labelsBwdk5 = ["BWD K=5", "humidity percent", "temp_mean"]

    CreateScatterPlot(adjustedHumidityValues, adjustedWeatherData["temp_mean"].values, humidityClass, labelsBwdData)
    CreateScatterPlot(adjustedHumidityValues, adjustedWeatherData["temp_mean"].values, predictedbwdK3OutlierAdj, labelsBwdk3)
    CreateScatterPlot(adjustedHumidityValues, adjustedWeatherData["temp_mean"].values, predictedbwdK5OutlierAdj, labelsBwdk5)
    
    # purity calculations
    print("k=3 Purity: " + str(Purity(predictedbwdK3OutlierAdj, humidityClass, False)) + "\n")
    print("k=5 Purity: " + str(Purity(predictedbwdK5OutlierAdj, humidityClass, False)) + "\n")

    # cluster centroids
    centroidBWDK3 = bwdK3.cluster_centers_
    centroidBWDK5 = bwdK5.cluster_centers_

    print("k=3 Cluster Centroid:\n" + str(centroidBWDK3) + "\n")
    print("k=5 Cluster Centroid:\n" + str(centroidBWDK5) + "\n")

    # SSE
    sseBWDK3 = bwdK3.inertia_
    sseBWDK5 = bwdK5.inertia_

    print("k=3 SSE: " + str(sseBWDK3) + "\n")
    print("k=5 SSE: " + str(sseBWDK5) + "\n")
    
    # SUBTASK E
    print("\n Subtask E\n")
    minPointsRange = [17,21]
    epsilonRange = [2.18,2.31]

    print("MinPoints Range : " + str(minPointsRange) + "\n")
    print("Epsilon Range : " + str(epsilonRange) + "\n")

    bestBwdDBSCAN = RandomSearch(adjustedWeatherData, humidityClass, minPointsRange, epsilonRange, 500)
    bestBwdClassifications = bestBwdDBSCAN.fit_predict(adjustedWeatherData)
    bestBwdClassesOutlierAdjusted = bestBwdClassifications + 1

    print("Number of Classes: " + str(max(bestBwdClassesOutlierAdjusted)) + "\n")

    labelsBwdDbscan = ["BWD Dbscan", "humidity percent", "temp_mean"]
    CreateScatterPlot(adjustedHumidityValues, adjustedWeatherData["temp_mean"].values, bestBwdClassesOutlierAdjusted, labelsBwdDbscan)

    print("BWD Dbscan Purity with Outliers: " + str(Purity(bestBwdClassesOutlierAdjusted, humidityClass, True)) + "\n")

main()
