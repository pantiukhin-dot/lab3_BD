using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;

namespace StopWordsClustering
{
    // Class representing stop word data with columns for word and language
    public class StopWordData
    {
        [LoadColumn(0)] // First column in the CSV corresponds to the word
        public string Word { get; set; }

        [LoadColumn(1)] // Second column in the CSV corresponds to the language
        public string Language { get; set; }
    }

    // Class representing the prediction results
    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")] // Predicted cluster ID
        public uint PredictedClusterId { get; set; }
        public float[] Score { get; set; } // Cluster distance scores
    }

    class Program
    {
        static void Main(string[] args)
        {
            // Path to the folder containing stop words
            string dataDirectory = @"C:\Users\archive\";
            string[] files = Directory.GetFiles(dataDirectory, "*.txt");

            // Load data from text files and combine into a single list
            var combinedData = files.SelectMany(file =>
            {
                var language = Path.GetFileNameWithoutExtension(file); // Infer language from file name
                var words = File.ReadAllLines(file).Where(line => !string.IsNullOrWhiteSpace(line));
                return words.Select(word => new StopWordData { Word = word.Trim(), Language = language });
            }).ToList();

            // Save the combined data to a CSV file for ML.NET processing
            string outputCsv = @"C:\Users\stop_words.csv";
            File.WriteAllLines(outputCsv, combinedData.Select(d => $"{d.Word},{d.Language}"));
            Console.WriteLine($"Data saved to {outputCsv}");

            // Initialize ML.NET context
            MLContext mlContext = new MLContext();

            // Load data from the CSV file
            IDataView data = mlContext.Data.LoadFromTextFile<StopWordData>(
                path: outputCsv,
                hasHeader: false,
                separatorChar: ',');

            // Build a pipeline for text clustering
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(StopWordData.Word))
                .Append(mlContext.Clustering.Trainers.KMeans(
                    featureColumnName: "Features",
                    numberOfClusters: 5)); // Specify the number of clusters

            // Train the model
            var model = pipeline.Fit(data);

            // Make predictions using the trained model
            var predictions = model.Transform(data);
            var clusteredData = mlContext.Data.CreateEnumerable<ClusterPrediction>(
                predictions, reuseRowObject: false).ToList();

            // Combine original data with cluster predictions
            var clusteredWords = combinedData.Zip(clusteredData, (original, prediction) => new
            {
                original.Word,
                original.Language,
                prediction.PredictedClusterId
            });

            // Display clustering results
            Console.WriteLine("\nClustering Results:");
            foreach (var cluster in clusteredWords.GroupBy(c => c.PredictedClusterId))
            {
                Console.WriteLine($"Cluster {cluster.Key}:");
                foreach (var item in cluster)
                {
                    Console.WriteLine($"  Word: {item.Word}, Language: {item.Language}");
                }
                Console.WriteLine();
            }

            Console.WriteLine("Clustering process completed.");
        }
    }
}
