package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {
	// Load and parse the data from csv files
	fmt.Println("Loading data...")

	trainData, err := base.ParseCSVToInstances("data/fashion-mnist_train.csv", true)
	if err != nil {
		panic(err)
	}

	testData, err := base.ParseCSVToInstances("data/fashion-mnist_testsmall.csv", true)
	if err != nil {
		panic(err)
	}

	fmt.Println("Very data, much mnist, wow")

	// Create a new KNN classifier with arbitrary values
	classifier := knn.NewKnnClassifier("euclidean", "kdtree", 5)


	// Train the classifier
	fmt.Println("Many training...")
	classifier.Fit(trainData)

	// Make predictions for the test data
	fmt.Println("Very predict...")
	predictions, err := classifier.Predict(testData)
	if err != nil {
		panic(err)
	}

	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}

	fmt.Println(evaluation.GetSummary(confusionMat))

}
