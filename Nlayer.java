
/*
* Alex Zhong
* Created 10 April 2024
* 
* This file trains and/or runs an N-layer Network and adheres to the design document provided by Dr. Eric Nelson.
* Calculation of the error function for training uses gradient descent and is optimized with backpropagation.
* 
* ==== Methods ====
* fileToStrArray(String)
* loadConfigParams(String)
* loadConfigParams()
* loadConfigHelper(String[])
* echoConfigParams()
* allocateMemoryTrain()
* allocateMemoryRun()
* populateArrays()
* populateInputs()
* loadInputs()
* loadOutputs()
* readWeights()
* writeWeights()
* populateWeightsRandom()
* setWeightsHardCode()
* runTrain(int, double[])
* train()
* runRun(double[])
* runCases()
* printTrainResults()
* printRunResults()
* f(double)
* fDeriv(double)
* sigmoid(double)
* sigmoidDeriv(double)
* hyperbolicTangent(double)
* hyperbolicTangentDeriv(double)
* rand(double double)
* main(String[])
*/

import java.io.*;
import java.util.*;

public class Nlayer
{
/*
* Declaration of configuration parameters
* 
* cases ---- number of test cases for training (positive)
* maxIter -- maximum number of training iterations (positive)
* randMin -- minimum value for the random assignment of weight values
* randMax -- maximum value for the random assignment of weight values
* lambda --- lambda value for training
* error ---- error cutoff for training (between 0 to 1)
*
* training - true if training, false if running
* save ----- true if weights will be saved after, false otherwise
* weightPopulation - how the weights will be populated: "LOAD" for loading from file, "RAND" for random, "SET" for hard-code
*/
   int cases, maxIter;
   double randMin, randMax, lambda, error;
   boolean training, save;
   int weightPopulation;

/*
* Declaration of network activations and training-related variables/arrays
*
* layers ----- number of layers of the network
* N ---------- number of activations per layer (length: layers)
* n ---------- the current layer being worked on (makes progression to N-layer easier)
* a ---------- 2D array of activation values (length per row x: N[m])
* weights ---- 3D array of weights (length based on a[])
*
* theta ------ 2D array of theta values (length per row n: N[n])
* psi -------- 2D array of psi values   (length per row n: N[n])
* omega ------ omega value for training
*
* thetaRun --- theta accumulator for the running process
*
* caseError  - the error for an individual test case of a training iteration
* totalError - the error for all four cases of a training iteration
*
* inputs ----- the inputs of the neural network
* eOutputs --- the expected outputs of the network
* cOutputs --- the calculated outputs after running the network
*/
   int layers, n;
   int[] N;
   double[][] a;
   double[][][] weights;
   double[][] theta, psi;
   double caseError, totalError;
   double[][] inputs, eOutputs, cOutputs;

/*
* Instance values for the training process; printed for the user
* 
* iter ------ current iteration of the training process
* keepAlive - number of iterations per iteration message
* avgError -- average error of the current iteration
*/
   int iter, keepAlive;
   double avgError;

/*
* Global constants for the purpose of choosing how to load weights
*
* RAND - randomized weights
* LOAD - load weights from file
* SET -- hard-code weights with the setWeightshardCode() method
* 
* the variables below are for hard-coded weights
* M ---- layer "M" (inputs)
* K ---- layer "K" (first hidden)
* J ---- layer "J" (second hidden)
* I ---- layer "I" (outputs) 
*/
   final int RAND = 0;
   final int LOAD = 1;
   final int SET  = 2;
   final int M = 0;
   final int K = 1;
   final int J = 2;
   final int I = 3;

/*
* DEFAULTCONTROl - the default control file for the network that holds configuration parameters
*
* controlFile ---- String to potentially hold the file name for the control file
* weightsFile ---- String to potentially hold the file name for loaded weights
* inputsFile ----- String to hold the file name for the inputs array
* outputsFile ---- String to hold the file name for the eOutputs array
* newWeightsFile - String to potentially hold the file destination for writing/saving weights
*/
   final String DEFAULTCONTROL = "control";
   String controlFile, weightsFile, inputsFile, outputsFile, newWeightsFile;

/*
* fileToStrArray converts a String file into a String array
* 
* @param fileName the name of the file to be converted
* @return the String array representation of the file
*/
   public String[] fileToStrArray(String fileName) throws IOException
   {
      File f = new File(fileName);
      BufferedReader br = new BufferedReader(new FileReader(f));

      int lines = 0;
      while (br.readLine() != null) { lines++; }
      br.close();

      String[] file = new String[lines];
      br = new BufferedReader(new FileReader(f));
      for (int ind =  0; ind < lines; ind++)
      {
         file[ind] = br.readLine();
      }
      br.close();

      return file;
   } // public String[] fileToStrArray(String fileName) throws IOException

/*
* loadConfigParams loads the configuration parameters of the network using an external file
* @param control the name of a potential control file
*/
   public void loadConfigParams(String control) throws IOException
   {
      controlFile = control;
      String[] config = fileToStrArray(controlFile);
      loadConfigHelper(config);
   } // public void loadConfigParams(String control) throws IOException

/*
* loadConfigParams loads the configuration parameters of the network using an external file
*/
   public void loadConfigParams() throws IOException
   {
      controlFile = DEFAULTCONTROL;
      String[] config = fileToStrArray(controlFile);
      loadConfigHelper(config);
   } // public void loadConfigParams() throws IOException

/*
* loadConfigHelper actually sets configuration variables to the values in an external control file
* 
* @param config a String representation of the external control file
*/
   public void loadConfigHelper(String config[])
   {
      int line = 0;

      StringTokenizer st = new StringTokenizer(config[line], " ");
      layers = Integer.parseInt(st.nextToken());

      line++;
      st = new StringTokenizer(config[line], " ");
      N = new int[layers];
      for (n = 0; n < layers; n++)
      {
         N[n] = Integer.parseInt(st.nextToken());
      }

      line++;
      st = new StringTokenizer(config[line], " ");
      weightPopulation = Integer.parseInt(st.nextToken());

      line++;
      weightsFile = config[line];

      line++;
      st = new StringTokenizer(config[line], " ");
      training = Boolean.parseBoolean(st.nextToken());

      line++;
      st = new StringTokenizer(config[line], " ");
      cases = Integer.parseInt(st.nextToken());

      line++;
      inputsFile = config[line];

      line++;
      outputsFile = config[line];

      line++;
      st = new StringTokenizer(config[line], " ");
      randMin = Double.parseDouble(st.nextToken());
      randMax = Double.parseDouble(st.nextToken());
      if (training)
      {
         lambda  = Double.parseDouble(st.nextToken());
         error   = Double.parseDouble(st.nextToken());
         maxIter = Integer.parseInt(st.nextToken());
      } // if (training)

      line++;
      st = new StringTokenizer(config[line], " ");
      save = Boolean.parseBoolean(st.nextToken());

      line++;
      newWeightsFile = config[line];

      line++;
      st = new StringTokenizer(config[line], " ");
      keepAlive = Integer.parseInt(st.nextToken());
   } // public void loadConfigHelper(String config[])

/*
* echoConfigParams prints configuration parameters prior to training/running
*/
   public void echoConfigParams()
   {
      System.out.printf("Reading control file \"%s\"%n", controlFile);
      System.out.printf("Reading inputs  file \"%s\"%n", inputsFile);
      System.out.printf("Reading outputs file \"%s\"%n", outputsFile);
      if (weightPopulation == 1)
      {
         System.out.printf("Reading weights file \"%s\"%n", weightsFile);
      }
      else if (weightPopulation == 0)
      {
         System.out.printf("Randomly populating weights%n");
      }
      reportWeights();
      System.out.printf("%nInitializing N-layer network with node layout ");

      for (n = 0; n < layers; n++)
      {
         System.out.printf("%s ", N[n]);
      }

      if (training)
      {
         System.out.printf("%nWeights Range: %s to %s%n", randMin, randMax);
         System.out.printf("Test Cases: %d%n", cases);
         System.out.printf("Max Iterations: %,d%n", maxIter);

         if (keepAlive != 0)
         {
            System.out.printf("Iterations per Keep-Alive Message: %d%n", keepAlive);
         }

         System.out.printf("Lambda: %.1f%n", lambda);
         System.out.printf("Error Cutoff: %s%n", error);

         System.out.printf("%nTraining...%n%n");
      } // if (training)

      else
      {
         System.out.printf("Weights Range: %s to %s%n", randMin, randMax);
         System.out.printf("Test Cases: %s%n", cases);
         System.out.printf("%nRunning...%n%n");
      }
   } // public void echoConfigparams()

/*
* allocateMemoryTrain creates the arrays required for the training process and allocates memory
*/
   public void allocateMemoryTrain()
   {
      a = new double[layers][];
      for (n = 0; n < layers; n++)
      {
         a[n] = new double[N[n]];
      }

      weights = new double[layers - 1][][];
      for (n = 0; n < layers - 1; n++) // no weights for the output (last) layer
      {
         weights[n] = new double[N[n]][N[n + 1]];
      }

      theta = new double[layers][];
      for (n = 1; n < layers; n++) // no theta for input (first) layer
      {
         theta[n] = new double[N[n]];
      }

      psi = new double[layers][];
      for (n = 1; n < layers; n++) // no psi for input (first) layer
      {
         psi[n] = new double[N[n]];
      }

      inputs     = new double[cases][N[0]]; // 0 for input activations
      eOutputs   = new double[cases][N[layers - 1]]; // layers - 1 for output activations
      cOutputs   = new double[cases][N[layers - 1]];

      iter = 0;
   } //public void allocateMemoryTrain()

/*
* allocateMemoryRun creates the arrays required for the running process and allocates memory
*/
   public void allocateMemoryRun()
   {
      a = new double[layers][];
      for (n = 0; n < layers; n++)
      {
         a[n] = new double[N[n]];
      }

      weights = new double[layers - 1][][];
      for (n = 0; n < layers - 1; n++) // no weights for the output (last) layer
      {
         weights[n] = new double[N[n]][N[n + 1]];
      }

      inputs     = new double[cases][N[0]]; // 0 for input activations
      eOutputs   = new double[cases][N[layers - 1]]; // layers - 1 for output activations
      cOutputs   = new double[cases][N[layers - 1]];
   } //public void allocateMemoryRun()

/*
* populateArrays populates the inputs, expected outputs, and weights arrays
*/
   public void populateArrays() throws IOException
   {
      loadInputs();
      loadOutputs();

      if (weightPopulation == LOAD)
      {
         readWeights();
      }
      else if (weightPopulation == RAND)
      {
         populateWeightsRandom();
      }
      else if (weightPopulation == SET)
      {
         setWeightsHardCode();
      }
   } // public void populateArrays()

/*
* loadInputs loads the inputs array ƒrom an external file
*/
   public void loadInputs() throws IOException
   {
      String[] file = fileToStrArray(inputsFile);
      
      StringTokenizer st = new StringTokenizer(file[0], " ");

      int tempInt;
      for (int c = 0; c < cases; c++)
      {
         st = new StringTokenizer(file[c]," "); // first line was param
         
         FileInputStream fstream = new FileInputStream(st.nextToken());

         // Convert our input stream to a DataInputStream
         DataInputStream in = new DataInputStream(fstream);

         for (int k = 0; k < N[0]; k++)
         {
            tempInt = (int) in.readInt();
            inputs[c][k] = ((double) tempInt) / 255.0; // 255 is the max value for RGB
         }

         in.close();
      } // for (int c = 0; c < cases; c++)
   } // public void loadInputs() throws IOException
   
/*
* loadOutputs loads the outputs array ƒrom an external file
*/
   public void loadOutputs() throws IOException
   {
      String[] file = fileToStrArray(outputsFile);
      
      StringTokenizer st = new StringTokenizer(file[0], " ");
      if (Integer.parseInt(st.nextToken()) != cases ||
          Integer.parseInt(st.nextToken()) != N[layers - 1]) // output activations number
      {
         throw new ArrayIndexOutOfBoundsException("Outputs file does not match configuration parameters");
      }

      for (int c = 0; c < cases; c++)
      {
         st = new StringTokenizer(file[c + 1]," "); // first line was param
         int k = 0; // number of inputs
         while (st.hasMoreTokens())
         {
            eOutputs[c][k] = Double.parseDouble(st.nextToken());
            k++;
         }
      } // for (int c = 0; c < cases; c++)
   } // public void loadOutputs() throws IOException

/*
* readWeights sets the weight arrays to the weight values in a specified file
*/
public void readWeights() throws IOException
{
   String[] file = fileToStrArray(weightsFile);

   int line = 0;
   StringTokenizer st = new StringTokenizer(file[line], " ");
   for (n = 0; n < layers; n++)
   {
      if (Integer.parseInt(st.nextToken()) != N[n])
      {
         throw new ArrayIndexOutOfBoundsException("Weights file does not match configuration parameters");
      }
   }

   for (n = 0; n < layers - 1; n++)
   {
      for (int k = 0; k < N[n]; k++)
      {
         line++;
         st = new StringTokenizer(file[line]," ");
         int j = 0;
         while (st.hasMoreTokens())
         {
            weights[n][k][j] = Double.parseDouble(st.nextToken());
            j++;
         }
      } // for (int k = 0; k < N[n]; k++)
   } // for (n = 0; n < layers - 1; n++)
} // public void readWeights()

/*
* writeWeights writes the weights array to a specified file
*/
public void writeWeights() throws IOException
{
   if (save)
   {
      File f = new File(newWeightsFile);
      BufferedWriter bw = new BufferedWriter (new FileWriter(f, false));

      for (n = 0; n < layers; n++)
      {
         bw.write(String.valueOf(N[n]  + " "));
      }
      bw.newLine();

      for (n = 0; n < layers - 1; n++)
      {
         for (int k = 0; k < N[n]; k++)
         {
            for (int j = 0; j < N[n + 1]; j++)
            {
               bw.write(String.valueOf(weights[n][k][j]) + " ");
            }
            bw.newLine();
         }
      } // for (n = 0; n < layers - 1; n++)

      bw.close();
   } // if (save)
} // public void writeWeights()

/*
* populateWeightsRandom randomly populates the weights array with random values
* The random values are between the specified range of randMin and randMax
*/
   public void populateWeightsRandom()
   {
      for (n = 0; n < layers - 1; n++)
      {
         for (int k = 0; k < N[n]; k++)
         {
            for (int j = 0; j < N[n + 1]; j++)
            {
               weights[n][k][j] = rand(randMin, randMax);
            }
         }
      } // for (n = 0; n < layers - 1; n++)
   } // public void populateWeightsRandom()

/*
* setWeightsHardCode will set the weights to pre-determined values, as well as the activations per layer
* This method is for running purposes only
*/
   public void setWeightsHardCode()
   {
      n = M;
      N[n]  = 2;
      n = K;
      N[n]  = 2;
      n = I;
      N[n]  = 1;

      n = K;
      weights[n][0][0] = -3.3;
      weights[n][0][1] = -1.5;
      weights[n][1][0] = 6.8;
      weights[n][1][1] = 0.3;
      n = J;
      weights[n][0][0] = 9.1;
      weights[n][1][0] = -22.5;
   } // public void setWeightsHardCode

/*
* runTrain takes in an array of inputs, then runs the network based on current weights; for training purposes
* 
* @param inputs the activations for the first layer
*/
   public void runTrain(int caseIter, double[] inputs)
   {
      double omega;
      a[0] = inputs; // pointer to the inputs for the specific test case, 0 for input activations

      for (n = 1; n < layers - 1; n++)
      {
         for (int j = 0; j < N[n]; j++)
         {
            theta[n][j] = 0.0;

            for (int k = 0; k < N[n - 1]; k++)
            {
               theta[n][j] += a[n - 1][k] * weights[n - 1][k][j];
            }

            a[n][j] = f(theta[n][j]);
         } // for (int j = 0; j < N[n]; j++)
      } // for (n = 1; n < layers - 1; n++)

      n = layers - 1; // layers - 1 (aka I) for output layer

      for (int i = 0; i < N[n]; i++) // i is used for output activation instead of j/k
      {
         theta[n][i] = 0.0;

         for (int j = 0; j < N[n - 1]; j++) // j here specifically refers to layer before i
         {
            theta[n][i] += a[n - 1][j] * weights[n - 1][j][i];
         }
         a[n][i] = f(theta[n][i]);

         omega  = eOutputs[caseIter][i] - a[n][i];  // ωi = (Ti − Fi)
         psi[n][i] = omega * fDeriv(theta[n][i]);   // ψ0 = ωi * ƒ'(Θi)
      } // for (int i = 0; i < N[n]; i++)
   } // public void runTrain(double[] inputs)

/*
* train uses gradient (steepest) descent to train the network
*/
   public void train()
   {
      do // while ((iter < maxIter) && (avgError > error));
      {
         double omega;
         totalError = 0.0;
         for (int caseIter = 0; caseIter < inputs.length; caseIter++)
         {
            caseError = 0.0;
            
            runTrain(caseIter, inputs[caseIter]);

            for (n = layers - 2; n > 1; n--) // layers - 2 is second to last layer (right before output layer)
            {
               for (int k = 0; k < N[n]; k++)
               {
                  omega = 0.0;
   
                  for (int j = 0; j < N[n + 1]; j++)
                  {
                     omega            += psi[n + 1][j] * weights[n][k][j];
                     weights[n][k][j] += lambda * a[n][k] * psi[n + 1][j];
                  }
   
                  psi[n][k] = omega * fDeriv(theta[n][k]);
               } // for (int k = 0; k < N[n]; k++)
            } // for (n = layers - 2; n > 1; n--)
            

            n = 1; // layer before input (first) layer
            for (int k = 0; k < N[n]; k++)
            {
               omega = 0.0;

               for (int j = 0; j < N[n + 1]; j++)
               {
                  omega            += psi[n + 1][j] * weights[n][k][j];
                  weights[n][k][j] += lambda * a[n][k] * psi[n + 1][j];
               }

               psi[n][k] = omega * fDeriv(theta[n][k]);

               for (int m = 0; m < N[n - 1]; m++)
               {
                  weights[n - 1][m][k] += lambda * a[n - 1][m] * psi[n][k];
               }
            } // for (int k = 0; k < N[n]; k++)

            runRun(inputs[caseIter]);      // run again with updated weights for error calculation

            n = layers - 1;                // output activations layer (I)

            for (int i = 0; i < N[n]; i++) // i is used here for output activations
            {
               omega      = eOutputs[caseIter][i] - a[n][i];
               caseError += (omega * omega) / 2;
            }
            totalError += caseError;
         } // for (int caseIter = 0; caseIter < inputs.length; caseIter++)

         avgError = totalError / cases; // calculate average error
         iter++;

         if ((keepAlive != 0) && (iter % keepAlive == 0)) // iterations is a multiple of keep-alive
         {
            System.out.printf("Iteration %d, Error = %.17f\n", iter, avgError);
         }

      } while ((iter < maxIter) && (avgError > error));
   } // public void train()

/*
* runRun takes in an array of inputs, then runs the network based on current weights without a theta array
* 
* @param inputs the activations for the first layer
*/
   public void runRun(double[] inputs)
   {
      double thetaRun;
      a[0] = inputs; // pointer to the inputs for the specific test case, 0 for input activations

      for (n = 1; n < layers; n++)
      {
         for (int j = 0; j < N[n]; j++)
         {
            thetaRun = 0.0;

            for (int k = 0; k < N[n - 1]; k++)
            {
               thetaRun += a[n - 1][k] * weights[n - 1][k][j];
            }

            a[n][j] = f(thetaRun);
         } // for (int j = 0; j < N[n]; j++)
      } // for (n = 1; n < layers - 1; n ++)
   } // public void runRun(double[] inputs)

/*
* runCases runs each test case (row) of the inputs[][] array
*/
   public void runCases()
   {
      for (int ind = 0; ind < cases; ind++)
      {
         runRun(inputs[ind]);

         for (int i = 0; i < N[layers - 1]; i++) // output activations layer (layers - 1)
         {
            cOutputs[ind][i] = a[layers - 1][i]; // set calculated output array for that specific test case
         }
      }
   } // public void runCases()

/*
* printTrainResults displays total iterations, average error, the inputs, and the expected and calculated outputs
*/
   public void printTrainResults()
   {
      if (iter >= maxIter)  System.out.printf("Maximum iterations reached.%n");
      if (avgError < error) System.out.printf("Error cutoff reached.%n");

      System.out.printf("Total Iterations: %,d%n", iter);
      System.out.printf("Average Error: %.17f%n%n", avgError);

      System.out.printf("Table below in order of: Expected Outputs, Calculated Outputs%n");
      for (int c = 0; c < cases; c++)
      {
         // for (double input : inputs[c])
         // {
         //    System.out.printf("%s ", input);
         // }
         // System.out.printf("  ");

         for (double eOut : eOutputs[c])
         {
            System.out.printf("%s ", eOut);
         }
         System.out.printf("  ");

         for (double cOut : cOutputs[c])
         {
            System.out.printf("%.3f ", cOut);
         }
         System.out.printf("%n");
      } // for (int c = 0; c < cases; c++)
   } // public void printTrainResults()

/*
* printRunResults displays the inputs followed by expected and calculated outputs
*/
   public void printRunResults()
   {
      System.out.printf("Table below in order of: Expected Outputs, Calculated Outputs%n%n");
      for (int c = 0; c < cases; c++)
      {
         // for (double input : inputs[c])
         // {
         //    System.out.printf("%s ", input);
         // }
         // System.out.printf("  ");

         for (double eOut : eOutputs[c])
         {
            System.out.printf("%s ", eOut);
         }
         System.out.printf("  ");

         for (double cOut : cOutputs[c])
         {
            System.out.printf("%.3f ", cOut);
         }
         System.out.printf("%n");
      } // for (int c = 0; c < cases; c++)
   } // public void printRunResults()

/*
* reportWeights prints if the weights were saved as well as where they were saved to
*/
   public void reportWeights()
   {
      if (save)
      {
         System.out.printf("Weights will save to file \"%s\"%n", newWeightsFile);
      }
      else
      {
         System.out.printf("Weights will not be saved.%n");
      }
   } // public void reportWeights()

/*
* f applies the activation function to the given input
* 
* @param input a double value to be operated on
* @return the result of the activation function on the given input (ie: sigmoid)
*/
   public double f(double input)
   {
      return sigmoid(input);
   }

/*
* fDeriv applies the derivative of the activation function to the given input
* 
* @param input a double value to be operated on
* @return the result of the derivative of the activation function on the given input (ie: sigmoidDeriv)
*/
   public double fDeriv(double input)
   {
      return sigmoidDeriv(input);
   }

/*
* sigmoid applies the sigmoid function to the given input
* 
* @param input a double value to be operated on
* @return the sigmoid function of the input
*/
public double sigmoid(double input)
{
   return 1.0 / (1.0 + Math.exp(-input));
}

/*
* sigmoidDeriv applies the sigmoid derivative to the given input
* 
* @param input a double value to be operated on
* @return the derivative of sigmoid of the input
*/
public double sigmoidDeriv(double input)
{
   double sig = sigmoid(input); // only need to calculate sigmoid once
   return sig * (1.0 - sig);
}

/*
* hyperbolicTangent applies the hyperbolic tangent function to the given input
* 
* @param input a double value to be operated on
* @return the hyperbolic tangent function of the input
*/
public double hyperbolicTangent(double input)
{
   double s = (input > 0) ? 1.0 : -1.0;
   double exp = Math.exp(-s * 2 * input);

   return s * ((1.0 - exp) / (1.0 + exp));
}

/*
* hyperbolicTangentDeriv applies the hyperbolic tangent derivative to the given input
* 
* @param input a double value to be operated on
* @return the derivative of the hyperbolic tangent of the input
*/
public double hyperbolicTangentDeriv(double input)
{
   double hTan = hyperbolicTangent(input);
   return 1.0 - hTan * hTan; // ƒ'(x) = 1 - ƒ(x) ^ 2
}

/*
* rand generates a random value between a range
*
* @param min the lower bound of the range
* @param max the upper bound of the range
* @return a random number between the specified range
*/
   public double rand(double min, double max)
   {
      return min + (max - min) * Math.random();
   }

/*
* Main method to run/train
* 
* @param args arguments from the command line
*/
   public static void main(String[] args) throws IOException
   {
      Nlayer p = new Nlayer();

      if (args.length != 0)
      {
         p.loadConfigParams(args[0]);
      }
      else
      {
         p.loadConfigParams();
      }

      if (p.training)
      {
         p.allocateMemoryTrain();
         p.populateArrays();
         p.echoConfigParams();
         p.train();
         p.runCases(); // run again to set the calculated outputs array
         p.printTrainResults();
         p.writeWeights();
      }
      else
      {
         p.allocateMemoryRun();
         p.populateArrays();
         p.echoConfigParams();
         p.runCases();
         p.printRunResults();
         p.writeWeights();
      }
   } // public static void main(String[] args)

} // public class ABCD