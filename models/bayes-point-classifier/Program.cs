using System;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using M = Microsoft.ML.Probabilistic.Models;
using At = Microsoft.ML.Probabilistic.Models.Attributes;
using D = Microsoft.ML.Probabilistic.Distributions;
using A = Microsoft.ML.Probabilistic.Algorithms;
using System.IO;
using System.Text;
namespace model
{
    class Program
    {
        static void Main(string[] args)
        {

            // Print out args
            Console.WriteLine("Arguments:");
            foreach (string s in args)
            {
                Console.WriteLine(s);
            }
   

            // Reading in the data
            string dataDir = args[0];
            string datasetFilename = dataDir+args[1];
            Console.WriteLine("Reading in data from " + datasetFilename);
            string[] lines = File.ReadAllLines(datasetFilename);
            bool[] isSetosaLabel = new bool[lines.Length];
            double[] featureVal = new double[lines.Length];

            Console.WriteLine("Looping through lines");

            for (int i = 0; i < lines.Length; i++)
            {
                Console.WriteLine("Line " + i);
                string[] strArray = lines[i].Split('|');
                Console.WriteLine("strArray length: " + strArray.Length);
                Console.WriteLine("strArray[0]: " + strArray[0]);
                Console.WriteLine("strArray[1]: " + strArray[1]);
                isSetosaLabel[i] = strArray[1] == "1,0";
                featureVal[i] = Convert.ToDouble(strArray[0]);
                // featureVal[i] = float.Parse(strArray[0]);
            }

            // Creating the model
            int numberOfSample = lines.Length;
            Console.WriteLine("Number of samples: " + numberOfSample);
            Range n = new Range(numberOfSample).Named("n");

            // Make sure that the range across flowers is handled sequentially
            n.AddAttribute(new At.Sequential());

            // Variables

            // The feature - x
            M.VariableArray<double> featureValues = M.Variable.Array<double>(n).Named("featureValue").Attrib(new At.DoNotInfer());
            // The label - y
            M.VariableArray<bool> isSetosa = M.Variable.Array<bool>(n).Named("isSetosa");

            // The weight - w
            M.Variable<double> weight = M.Variable.GaussianFromMeanAndVariance(0,1).Named("weight");     
            // The threshold
            M.Variable<double> threshold = M.Variable.GaussianFromMeanAndVariance(0,10).Named("threshold");

            // Loop over Ns
            using (M.Variable.ForEach(n))
            {
                var score = (featureValues[n] * weight).Named("score");

                var noisyScore = M.Variable.GaussianFromMeanAndVariance(score, 100).Named("noisyScore");
                isSetosa[n] = noisyScore > threshold;
            }

            /********* observations *********/
            isSetosa.ObservedValue = isSetosaLabel;
            featureValues.ObservedValue = featureVal;
            /*******************************/

            /********** inference **********/
            var engine = new M.InferenceEngine(new A.ExpectationPropagation());
            // var InferenceEngine = new InferenceEngine(new VariationalMessagePassing());
            engine.NumberOfIterations = 50;
            // engine.ShowFactorGraph = true;

            D.Gaussian postWeight = engine.Infer<D.Gaussian>(weight);
            D.Gaussian postThreshold = engine.Infer<D.Gaussian>(threshold);
            /*******************************/

            Console.WriteLine(postWeight);
            Console.WriteLine(postThreshold);

            // write outputs to file
            var results = new StringBuilder();

            results.AppendLine("variable;mean;variance");
            var line = string.Format("postWeight;{0};{1}", postWeight.GetMean(), postWeight.GetVariance());
            results.AppendLine(line.Replace(',', '.'));
            line = string.Format("postThreshold;{0};{1}", postThreshold.GetMean(), postThreshold.GetVariance());
            results.AppendLine(line.Replace(',', '.'));

            File.WriteAllText(dataDir+"results.csv", results.ToString());
            
        }
    }
}
