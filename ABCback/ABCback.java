

import java.io.*;
import java.util.*;

public class ABCback
{
/*
* Declaration of configuration parameters
* 
* nInputs -- number of input activations/nodes
* nHidden -- number of hidden activations/nodes
* nOutputs - number of output activations/nodes
* cases ---- number of test cases for training (positive)
* maxIter -- maximum number of training iterations (positive)
* randMin -- minimum value for the random assignment of weight values
* randMax -- maximum value for the random assignment of weight values
* lambda --- lambda value for training
* error ---- error cutoff for training (between 0 to 1)
* training - true if training, false if running
* svae ----- true if weights will be saved after, false otherwise
* weightPopulation - how the weights will be populated: "LOAD" for loading from file, "RAND" for random, "SET" for hard-code
*/
   int nInputs, nHidden, nOutputs, cases, maxIter;
   double randMin, randMax, lambda, error;
   boolean training, save;
   int weightPopulation;

/*
* Declaration of network activations and training-related variables/arrays
* 
* a -- the input activations of the network
* h -- the hidden activations of the network
* f0 - the output value of the network
*
* jTheta - theta values (accumulator) for the hidden activations
* jOmega - omega values of hidden activations; used for training
* jPsi --- psi values of hidden activations; used for training
* 
* iTheta - theta values for the output activations
* iOmega - omega value for output activations
* iPsi --- psi value for output activations
*
* theta -- theta accumulator for the running process
*
* kWeights - stores weight values between the k and j activations
* jWeights - stores weight values between the j and i (output) activations
*
* caseError  - the error for an individual test case of a training iteration
* totalError - the error for all four cases of a training iteration
*
* inputs --- the inputs of the neural network
* eOutputs - the expected outputs of the network
* cOutputs - the calculated outputs after running the network
*/
   double[] a, h, F;
   double[] jTheta;
   double[] iTheta, iPsi;
   double jOmega, jPsi, iOmega, theta, caseError, totalError;
   double[][] kWeights, jWeights;
   double[][] inputs, eOutputs, cOutputs;

/*
* Instance values for the training process; printed for the user
* 
* iter ----- current iteration of the training process
* avgError - average error of the current iteration
*/
   int iter;
   double avgError;

/*
* global constants for the purpose of choosing how to load weights
* RAND - randomized weights
* LOAD - load weights from file
* SET -- hard-code weights with the setWeightshardCode() method
*/
   final int RAND = 0;
   final int LOAD = 1;
   final int SET  = 2;

/*
* DEFAULTCONTROl - the default control file for the network that holds configuration parameters
* weightsFile ---- String to potentially hold the file name for loaded weights
* inputsFile ------ String to hold the file name for the inputs array
*/
   final String DEFAULTCONTROL = "control";
   String controlFile, weightsFile, inputsFile, outputsFile, newWeightsFile;

/*
* setConfigParams sets the configuration parameters to hard-coded values
* See variable declarations for clarification on variable names
*/
   public void setConfigParams()
   {
      nInputs  = 2;
      nHidden  = 5;
      nOutputs = 3;

      weightPopulation = LOAD; // LOAD to load from file, RAND for randomized, "SET" for hardcode
      training = true;
      cases = 4; // test cases

      if (training)
      {
         randMin = 0.1;
         randMax = 1.5;
         maxIter = 100000;
         lambda  = 0.3;
         error   = 2E-4;
      }
   } // public void setConfigParams()

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

      // find the number of lines in the file
      int lines = 0;
      while (br.readLine() != null) { lines++; }
      br.close();

      // convert file to a String array
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
      int line = 1;

      StringTokenizer st = new StringTokenizer(config[line], " ");
      nInputs  = Integer.parseInt(st.nextToken());
      nHidden  = Integer.parseInt(st.nextToken());
      nOutputs = Integer.parseInt(st.nextToken());

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
      System.out.printf("%nInitializing ABC network with node layout %s-%s-%s.%n", nInputs, nHidden, nOutputs);

      if (training)
      {
         System.out.printf("Weights Range: %s to %s%n", randMin, randMax);
         System.out.printf("Test Cases: %d%n", cases);
         System.out.printf("Max Iterations: %,d%n", maxIter);
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
      // Allocation for activations and associated values
      a = new double[nInputs];
      h = new double[nHidden];
      F = new double[nOutputs];

      jTheta = new double[nHidden];
      iTheta = new double[nOutputs];
      iPsi   = new double[nOutputs];

      // Allocation for weights and associated values
      kWeights = new double[nInputs][nHidden];
      jWeights = new double[nHidden][nOutputs];

      inputs   = new double[cases][nInputs];
      eOutputs = new double[cases][nOutputs];
      cOutputs = new double[cases][nOutputs];

      iter = 0;
   } //public void allocateMemoryTrain()

/*
* allocateMemoryRun creates the arrays required for the running process and allocates memory
*/
   public void allocateMemoryRun()
   {
      a = new double[nInputs];
      h = new double[nHidden];
      F = new double[nOutputs];

      kWeights = new double[nInputs][nHidden];
      jWeights = new double[nHidden][nOutputs];

      inputs   = new double[cases][nInputs];
      eOutputs = new double[cases][nOutputs];
      cOutputs = new double[cases][nOutputs];
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
* populateInputs populates the input arrays in the following format
* 0 0
* 0 1
* 1 0
* 1 1
*/
   public void populateInputs()
   {
      inputs[0][0] = 0.0;
      inputs[0][1] = 0.0;
      inputs[1][0] = 0.0;
      inputs[1][1] = 1.0;
      inputs[2][0] = 1.0;
      inputs[2][1] = 0.0;
      inputs[3][0] = 1.0;
      inputs[3][1] = 1.0;
   } // public void populateInputs()

/*
* loadInputs loads the inputs array ƒrom an external file
*/
   public void loadInputs() throws IOException
   {
      String[] file = fileToStrArray(inputsFile);
      
      // confirm the input dimensions match 1. test cases and 2. nInputs
      StringTokenizer st = new StringTokenizer(file[0], " ");
      if (Integer.parseInt(st.nextToken()) != cases ||
          Integer.parseInt(st.nextToken()) != nInputs)
      {
         throw new ArrayIndexOutOfBoundsException("Weights file does not match configuration parameters");
      }

      // start parsing the file (String array representation of the file)
      for (int c = 0; c < cases; c++)
      {
         st = new StringTokenizer(file[c + 1]," "); // first line was param
         int k = 0; // number of inputs
         while (st.hasMoreTokens())
         {
            inputs[c][k] = Double.parseDouble(st.nextToken());
            k++;
         }
      } // for (int k = 0; k < nInputs; k++)
   } // public void loadInputs() throws IOException

/*
* populateOutputs populates the expected outputs array
*/
   public void populateOutputs()
   {
      eOutputs[0][0] = 0.0;
      eOutputs[0][1] = 0.0;
      eOutputs[0][2] = 0.0;
      eOutputs[1][0] = 0.0;
      eOutputs[1][1] = 1.0;
      eOutputs[1][2] = 1.0;
      eOutputs[2][0] = 0.0;
      eOutputs[2][1] = 1.0;
      eOutputs[2][2] = 1.0;
      eOutputs[3][0] = 1.0;
      eOutputs[3][1] = 1.0;
      eOutputs[3][2] = 0.0;
   } // public void populateOutputs()
   
/*
* loadOutputs loads the outputs array ƒrom an external file
*/
   public void loadOutputs() throws IOException
   {
      String[] file = fileToStrArray(outputsFile);
      
      // confirm the input dimensions match 1. test cases and 2. nInputs
      StringTokenizer st = new StringTokenizer(file[0], " ");
      if (Integer.parseInt(st.nextToken()) != cases ||
          Integer.parseInt(st.nextToken()) != nOutputs)
      {
         throw new ArrayIndexOutOfBoundsException("Weights file does not match configuration parameters");
      }

      // start parsing the file (String array representation of the file)
      for (int c = 0; c < cases; c++)
      {
         st = new StringTokenizer(file[c + 1]," "); // first line was param
         int k = 0; // number of inputs
         while (st.hasMoreTokens())
         {
            eOutputs[c][k] = Double.parseDouble(st.nextToken());
            k++;
         }
      } // for (int k = 0; k < nInputs; k++)
   } // public void loadOutputs() throws IOException

/*
* readWeights sets the weight arrays to the weight values in a specified file
*/
public void readWeights() throws IOException
{
   String[] file = fileToStrArray(weightsFile);

   // confirm the weights parameters match with the control
   StringTokenizer st = new StringTokenizer(file[0], " ");
   if (Integer.parseInt(st.nextToken()) != nInputs ||
       Integer.parseInt(st.nextToken()) != nHidden ||
       Integer.parseInt(st.nextToken()) != nOutputs)
   {
      throw new ArrayIndexOutOfBoundsException("Weights file does not match configuration parameters");
   }

   // start parsing the file (String array representation of the file)
   for (int k = 0; k < nInputs; k++)
   {
      st = new StringTokenizer(file[k + 1]," "); // first line was param
      int j = 0;
      while (st.hasMoreTokens())
      {
         kWeights[k][j] = Double.parseDouble(st.nextToken());
         j++;
      }
   } // for (int k = 0; k < nInputs; k++)

   for (int j = 0; j < nHidden; j++)
   {
      st = new StringTokenizer(file[nInputs + j + 1]," ");  // first line was param
      int i = 0;
      while (st.hasMoreTokens())
      {
         jWeights[j][i] = Double.parseDouble(st.nextToken());
         i++;
      }
   } // for (int j = 0; j < nHidden; j++)
} // public void readWeights()

/*
* writeWeights takes in two arrays (probably kWeights and jWeights) and writes them to a new file
*/
public void writeWeights() throws IOException
{
   if (save)
   {
      File f = new File(newWeightsFile);
      BufferedWriter bw = new BufferedWriter (new FileWriter(f, false));

      bw.write(String.valueOf(nInputs  + " "));
      bw.write(String.valueOf(nHidden  + " "));
      bw.write(String.valueOf(nOutputs + " "));
      bw.newLine();

      for (int k = 0; k < nInputs; k++)
      {
         for (int j = 0; j < nHidden; j++)
         {
            bw.write(String.valueOf(kWeights[k][j]) + " ");
         }
         bw.newLine();
      }

      for (int j = 0; j < nHidden; j++)
      {
         for (int i = 0; i < nOutputs; i++)
         {
            bw.write(String.valueOf(jWeights[j][i] + " "));
         }
         bw.newLine();
      }

      bw.close();
   } // if (save)
} // public void writeWeights()

/*
* populateWeightsRandom randomly populates the weights array with random values
* The random values are between the specified range of randMin and randMax
*/
   public void populateWeightsRandom()
   {
      for (int k = 0; k < nInputs; k++)
      {
         for (int j = 0; j < nHidden; j++)
         {
            kWeights[k][j] = rand(randMin, randMax); // weights between input and hidden
         }
      }

      for (int j = 0; j < nHidden; j++)
      {
         for (int i = 0; i < nOutputs; i++)
         {
            jWeights[j][i] = rand(randMin, randMax); // weights between hidden and output
         }
      }
   } // public void populateWeightsRandom()

/*
* setWeightsHardCode will set the weights to pre-determined values, as well as the activations per layer
* This method is for running purposes only
*/
   public void setWeightsHardCode()
   {
      nInputs  = 2;
      nHidden  = 2;
      nOutputs = 1;

      kWeights[0][0] = -3.3;
      kWeights[0][1] = -1.5;
      kWeights[1][0] = 6.8;
      kWeights[1][1] = 0.3;
      jWeights[0][0] = 9.1;
      jWeights[1][0] = -22.5;
   } // public void setWeightsHardCode

/*
* runTrain takes in an array of inputs, then runs the network based on current weights; for training purposes
* 
* @param inputs the activations for the first layer
*/
   public void runTrain(int caseIter, double[] inputs)
   {
      a = inputs; // pointer to the inputs for the specific test case

      for (int j = 0; j < nHidden; j++) // Sets h[j] = ƒ(Θj) , θj = ∑(a[K] * w[K][j])
      {
         jTheta[j] = 0.0;

         for (int k = 0; k < nInputs; k++)
         {
            jTheta[j] += a[k] * kWeights[k][j];
         }

         h[j] = f(jTheta[j]);
      } // for (int j = 0; j < nHidden; j++)

      for (int i = 0; i < nOutputs; i++)
      {
         iTheta[i] = 0.0;

         for (int j = 0; j < nHidden; j++)
         {
            iTheta[i] += h[j] * jWeights[j][i];      // θi = ∑(h[J] * w[J][i])
         }
         F[i] = f(iTheta[i]);                  // F[i] = ƒ(Θi)

         iOmega  = eOutputs[caseIter][i] - F[i];     // ωi = (Ti − Fi)
         iPsi[i] = iOmega * fDeriv(iTheta[i]); // ψ0 = ωi * ƒ'(Θi)
      } // for (int i = 0; i < nOutputs; i++)
   } // public void runTrain(double[] inputs)

/*
* train uses gradient (steepest) descent to train the network
*/
   public void train()
   {
      do // while ((iter < maxIter) && (avgError > error));
      {
         totalError = 0.0;
         for (int caseIter = 0; caseIter < inputs.length; caseIter++)
         {
            caseError = 0.0; // reset the error for the specific case
            
            runTrain(caseIter, inputs[caseIter]); // evaluation of network and accumulation

            for (int j = 0; j < nHidden; j++)
            {
               jOmega = 0.0; // reset jOmega from previous loop

               for (int i = 0; i < nOutputs; i++)
               {
                  jOmega         += iPsi[i] * jWeights[j][i]; // Ωj = ∑(ψI * w[j][I])
                  jWeights[j][i] += lambda * h[j] * iPsi[i];  // w[j][i] += λ * h[j] * ψi
               }

               jPsi = jOmega * fDeriv(jTheta[j]);       // Ψj = Ωj * ƒ'(Θj)

               for (int k = 0; k < nInputs; k++)
               {
                  kWeights[k][j] += lambda * a[k] * jPsi;     // w[k][j] += λ * a[k] * ψ0
               }
            } // for (int j = 0; j < nHidden; j++)

            runRun(inputs[caseIter]); // run again with updated weights for error calculation

            for (int i = 0; i < nOutputs; i++)
            {
               iOmega = eOutputs[caseIter][i] - F[i]; // ωi = (Ti − Fi)
               caseError += (iOmega * iOmega) / 2;    // (∑ ωi * ωi) / 2
            }
            totalError += caseError;
         } // for (int caseIter = 0; caseIter < inputs.length; caseIter++)

         avgError = totalError / cases; // calculate average error
         iter++; // increase the training iteration counter

      } while ((iter < maxIter) && (avgError > error));
   } // public void train()

/*
* runRun takes in an array of inputs, then runs the network based on current weights without a theta array
* 
* @param inputs the activations for the first layer
*/
   public void runRun(double[] inputs)
   {
      a = inputs;

      for (int j = 0; j < nHidden; j++) // Sets h[j] = ƒ(Θj), θj = ∑(a[k] * w[k][j])
      {
         theta = 0.0;

         for (int k = 0; k < nInputs; k++)
         {
            theta += a[k] * kWeights[k][j];
         }

         h[j]= f(theta);
      } // for (int j = 0; j < nHidden; j++)

      for (int i = 0; i < nOutputs; i++) // Sets F[i] = ƒ(Θi), θi = ∑(h[j] * w[j][i])
      {
         theta = 0.0;

         for (int j = 0; j < nHidden; j++)
         {
            theta += h[j] * jWeights[j][i];
         }

         F[i] = f(theta);
      } // for (int i = 0; i < nOutputs; i++)
   } // public void runRun(double[] inputs)

/*
* runCases runs each test case (row) of the inputs[][] array
*/
   public void runCases()
   {
      for (int ind = 0; ind < cases; ind++)
      {
         runRun(inputs[ind]);

         for (int i = 0; i < nOutputs; i++)
         {
            cOutputs[ind][i] = F[i]; // set calculated output array for that specific test case
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
      System.out.printf("Average Error: %.15f%n%n", avgError);

      System.out.printf("Table below in order of: Inputs, Expected Outputs, Calculated Outputs%n");
      for (int c = 0; c < cases; c++)
      {
         for (double input : inputs[c])
         {
            System.out.printf("%s ", input);
         }
         System.out.printf("  ");

         for (double eOut : eOutputs[c])
         {
            System.out.printf("%s ", eOut);
         }
         System.out.printf("  ");

         for (double cOut : cOutputs[c])
         {
            System.out.printf("%.17f ", cOut);
         }
         System.out.printf("%n");
      } // for (int c = 0; c < cases; c++)
   } // public void printTrainResults()

/*
* printRunResults displays the inputs followed by expected and calculated outputs
*/
   public void printRunResults()
   {
      System.out.printf("Table below in order of: Inputs, Expected Outputs, Calculated Outputs%n%n");
      for (int c = 0; c < cases; c++)
      {
         for (double input : inputs[c])
         {
            System.out.printf("%s ", input);
         }
         System.out.printf("  ");

         for (double eOut : eOutputs[c])
         {
            System.out.printf("%s ", eOut);
         }
         System.out.printf("  ");

         for (double cOut : cOutputs[c])
         {
            System.out.printf("%.17f ", cOut);
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
* Main method to run/train based on config params in setConfigParams
* 
* @param args arguments from the command line
*/
   public static void main(String[] args) throws IOException
   {
      ABCback p = new ABCback();

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

} // public class AB1