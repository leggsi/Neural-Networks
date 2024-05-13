package ABC;
/*
* Alex Zhong
* Created 23 February 2024
* 
* This file trains and/or runs an A-B-C Network and adheres to the design document provided by Dr. Eric Nelson.
* Calculation of the error function for training uses gradient descent.
* 
* ==== Methods ====
* setConfigParams()
* echoConfigParams()
* rand(double double)
* readWeights()
* writeWeights(double[][], double[][])
* allocateMemoryTrain()
* allocateMemoryRun()
* populateArrays()
* populateInputs()
* populateOutputs()
* populateWeightsRandom()
* setWeightshardCode()
* runTrain(double[])
* runRun(double[])
* runCases()
* train()
* printTrainResults()
* printRunResults()
* sigmoid(double)
* sigmoidDeriv(double)
* main(String[])
*/

import java.io.*;
import java.util.*;
import java.awt.*;

public class ABC
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
* weightPopulation - how the weights will be populated: "LOAD" for loading from file, "RAND" for random, "SET" for hard-code
*/
   int nInputs, nHidden, nOutputs, cases, maxIter;
   double randMin, randMax, lambda, error;
   boolean training;
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
* kDeltaW -- stores the change in kWeights (∆w) for each training iteration
* kDeriv --- stores the derivatives before application of lambda for kWeights
*
* jWeights - stores weight values between the j and i (output) activations
* jDeltaW -- stores the change in jWeights (∆w) for each training iteration
* jDeriv --- stores the derivatives before application of lambda for jWeights
*
* caseError - the error for an individual test case of a training iteration
* allError -- stores the error of all test cases; changes per training iteration
*
* inputs --- the inputs of the neural network
* eOutputs - the expected outputs of the network
* cOutputs - the calculated outputs after running the network
*/
   double[] a, h, F;
   double[] jTheta, jOmega, jPsi;
   double[] iTheta, iPsi;
   double iOmega, theta, caseError;
   double[][] kWeights, kDeltaW, kDeriv;
   double[][] jWeights, jDeltaW, jDeriv;
   double[] allError;
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


   FileDialog fd; // for file IO

/*
* setConfigParams sets the configuration parameters to hard-coded values
* See variable declarations for clarification on variable names
*/
   public void setConfigParams()
   {
      nInputs  = 2;
      nHidden  = 5;
      nOutputs = 3;

      training = true;
      weightPopulation = RAND; // LOAD to load from file, RAND for randomized, "SET" for hardcode
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
* echoConfigParams prints configuration parameters prior to training/running
*/
   public void echoConfigParams()
   {
      System.out.printf("Initializing AB1 network with node layout %s-%s-%s.%n", nInputs, nHidden, nOutputs);

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
         System.out.printf("Test Cases: %s%n", cases);
         System.out.printf("%nRunning...%n%n");
      }
   } // public void echoConfigparams()

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
* readWeights sets the weight arrays to the weight values in a specified file
*/
   public void readWeights()
   {
      fd = new FileDialog(new Frame(), "Open File", FileDialog.LOAD); // choose which file to read
      fd.setVisible(true);

      File f = null;
      if ((fd.getDirectory() != null) || ( fd.getFile() != null))
      {
         f = new File(fd.getDirectory() + fd.getFile());
      }

      FileReader fr = null;
      try
      {
         fr = new FileReader (f);
      }
      catch (FileNotFoundException fnfe)
      {
         fnfe.printStackTrace();
      }

      BufferedReader br = new BufferedReader(fr);
      int lines = -1;
      String textIn = " ";
      String[] file = null;
      try
      {
         while (textIn != null)
         {
            textIn = br.readLine();
            lines++;
         }
         file = new String[lines];
         fr = new FileReader (f);
         br = new BufferedReader(fr);
         for (int i = 0; i < lines; i++)
         {
            file[i] = br.readLine();
         }
         br.close();
      } // try
      catch (IOException ioe)
      {
         ioe.printStackTrace();
      }

      for (int k = 0; k < nInputs; k++)
      {
         StringTokenizer st = new StringTokenizer(file[k]," ");
         int j = 0;
         while (st.hasMoreTokens())
         {
            kWeights[k][j] = Double.parseDouble(st.nextToken());
            j++;
         }
      } // for (int k = 0; k < nInputs; k++)

      for (int j = 0; j < nHidden; j++)
      {
         StringTokenizer st = new StringTokenizer(file[nInputs + j]," ");
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
   public void writeWeights(double[][] dataIn, double[][]dataIn2)
   {
      fd = new FileDialog(new Frame(), "Save File", FileDialog.SAVE); // choose where to save file
      fd.setVisible(true);

      File f = null;
      if ((fd.getDirectory() != null) || ( fd.getFile() != null))
      {
         f = new File(fd.getDirectory() + fd.getFile());
      }

      FileWriter fw = null;
      try
      {
         fw = new FileWriter (f, true);
      }
      catch (IOException ioe)
      {
         ioe.printStackTrace();
      }

      BufferedWriter bw = new BufferedWriter (fw);
      try
      {
         for (int row = 0; row < dataIn.length; row++)
         {
            for (int col = 0; col < dataIn[row].length; col++)
            {
               bw.write(String.valueOf(dataIn[row][col]) + " ");
            }
            bw.newLine();
         }

         for (int row = 0; row < dataIn2.length; row++)
         {
            for (int col = 0; col < dataIn2[row].length; col++)
            {
               bw.write(String.valueOf(dataIn2[row][col]) + " ");
            }
            bw.newLine();
         }

         bw.close();
      } // try
      catch (IOException ioe)
      {
         ioe.printStackTrace();
      }
   } // public void writeWeights(double[][] dataIn double[][]dataIn2)

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
      jOmega = new double[nHidden];
      jPsi   = new double[nHidden];

      iTheta = new double[nOutputs];
      iPsi   = new double[nOutputs];

      // Allocation for weights and associated values
      kWeights = new double[nInputs][nHidden];
      kDeltaW  = new double[nInputs][nHidden];
      kDeriv   = new double[nInputs][nHidden];

      jWeights = new double[nHidden][nOutputs];
      jDeltaW  = new double[nHidden][nOutputs];
      jDeriv   = new double[nHidden][nOutputs];

      inputs   = new double[cases][nInputs];
      eOutputs = new double[cases][nOutputs];
      cOutputs = new double[cases][nOutputs];

      allError = new double[cases];
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
   public void populateArrays()
   {
      populateInputs();
      populateOutputs();

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
      // inputs[0][2] = 0.0;
      inputs[1][0] = 0.0;
      inputs[1][1] = 1.0;
      // inputs[1][2] = 0.0;
      inputs[2][0] = 1.0;
      inputs[2][1] = 0.0;
      // inputs[2][2] = 0.0;
      inputs[3][0] = 1.0;
      inputs[3][1] = 1.0;
      // inputs[3][2] = 0.0;
   } // public void populateInputs()

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
* @return output value
*/
   public double[] runTrain(double[] inputs)
   {
      a = inputs;

      for (int j = 0; j < nHidden; j++) // Sets h[j] = ƒ(Θj) , θj = ∑(a[K] * w[K][j])
      {
         jTheta[j] = 0.0;

         for (int K = 0; K < nInputs; K++)
         {
            jTheta[j] += a[K] * kWeights[K][j];
         }

         h[j] = sigmoid(jTheta[j]);

      } // for (int j = 0; j < nHidden; j++)

      for (int i = 0; i < nOutputs; i++) // Sets F[i] = ƒ(Θi) , θi = ∑(h[J] * w[J][i])
      {
         iTheta[i] = 0.0;

         for (int J = 0; J < nHidden; J++)
         {
            iTheta[i] += h[J] * jWeights[J][i];
         }

         F[i] = sigmoid(iTheta[i]);

      } // for (int i = 0; i < nOutputs; i++)

      return F;
   } // public double runTrain(double[] inputs)

/*
* runRun takes in an array of inputs, then runs the network based on current weights without a theta array
* 
* @param inputs the activations for the first layer
*/
   public void runRun(double[] inputs)
   {
      a = inputs;

      for (int j = 0; j < nHidden; j++) // Sets h[j] = ƒ(Θj) , θj = ∑(a[k] * w[k][j])
      {
         theta = 0.0;

         for (int k = 0; k < nInputs; k++)
         {
            theta += a[k] * kWeights[k][j];
         }

         h[j]= sigmoid(theta);

      } // for (int j = 0; j < nHidden; j++)

      for (int i = 0; i < nOutputs; i++)// Sets F[i] = ƒ(Θi) , θi = ∑(h[j] * w[j][i])
      {
         theta = 0.0;

         for (int j = 0; j < nHidden; j++)
         {
            theta += h[j] * jWeights[j][i];
         }

         F[i] = sigmoid(theta);

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
* train uses gradient (steepest) descent to train the network
*/
   public void train()
   {
      do // while ((iter < maxIter) && (avgError > error));
      {
         for (int caseIter = 0; caseIter < inputs.length; caseIter++)
         {
            runTrain(inputs[caseIter]);
            
            caseError = 0.0;
            for (int i = 0; i < nOutputs; i++)
            {
               iOmega  = eOutputs[caseIter][i] - F[i];     // ωi = (Ti − Fi)
               iPsi[i] = iOmega * sigmoidDeriv(iTheta[i]); // ψ0 = ωi * ƒ'(Θi)

               caseError += (iOmega * iOmega) / 2; // update caseError after iOmega is set
            }
            allError[caseIter] = caseError; // update error calculation

            for (int j = 0; j < nHidden; j++)
            {
               jOmega[j] = 0.0; // reset omega from the previous iteration

               for (int i = 0; i < nOutputs; i++)
               {
                  jDeriv[j][i]  = -h[j] * iPsi[i];        // ∂E/∂w[j][i] = −h[j] * ψi
                  jDeltaW[j][i] = -lambda * jDeriv[j][i]; // Δw[j][i] = −λ * ∂E/∂w[j][i]

                  jOmega[j] += iPsi[i] * jWeights[j][i];  // Ωj = ∑(ψI * w[j][I])
               }
               

               jPsi[j] = jOmega[j] * sigmoidDeriv(jTheta[j]);   // Ψj = Ωj * ƒ'(Θj)

               for (int k = 0; k < nInputs; k++)
               {
                  kDeriv[k][j]  = -a[k] * jPsi[j];              // ∂E/∂w[k][j] = −a[k] * ψ0
                  kDeltaW[k][j] = -lambda * kDeriv[k][j];       // Δw[k][j] = −λ * ∂E/∂w[k][j]
               }
            } // for (int j = 0; j < nHidden; j++)

            // Update weights
            for (int k = 0; k < nInputs; k++)
            {
               for (int j = 0; j < nHidden; j++)
               {
                  kWeights[k][j] += kDeltaW[k][j];
               }
            } // for (int k = 0; k < nInputs; k++)

            for (int j = 0; j < nHidden; j++)
            {
               for (int i = 0; i < nOutputs; i++)
               {
                  jWeights[j][i] += jDeltaW[j][i];
               }
            } // for (int j = 0; j < nHidden; j++)

         } // for (int caseIter = 0; caseIter < inputs.length; caseIter++)

         // calculate average error of all cases
         avgError = 0.0;
         for (double c : allError) avgError += c; // sum the error across all cases
         avgError /= (double) cases;

         iter++; // increase thte training iteration counter

      } while ((iter < maxIter) && (avgError > error));

   } // public void train()

/*
* printTrainResults displays total iterations, average error, the inputs, and the expected and calculated outputs
*/
   public void printTrainResults()
   {
      if (iter >= maxIter)  { System.out.printf("Maximum iterations reached.%n"); }
      if (avgError < error) { System.out.printf("Error cutoff reached.%n"); }

      System.out.printf("Total Iterations: %,d%n", iter);
      System.out.printf("Average Error: %.15f%n%n", avgError);

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
            System.out.printf("%.5f ", cOut);
         }
         System.out.printf("%n%n");
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
            System.out.printf("%.5f ", cOut);
         }
         System.out.printf("%n%n");
      } // for (int c = 0; c < cases; c++)

   } // public void printRunResults()

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
* Main method to run/train based on config params in setConfigParams
* 
* @param args arguments from the command line
*/
   public static void main(String[] args)
   {
      ABC p = new ABC();

      p.setConfigParams();

      if (p.training)
      {
         p.allocateMemoryTrain();
         p.populateArrays();
         p.echoConfigParams();
         p.train();
         p.runCases(); // run once to set cOutputs array
         p.printTrainResults();
      }
      else
      {
         p.allocateMemoryRun();
         p.populateArrays();
         p.echoConfigParams();
         p.runCases();
         p.printRunResults();
      }
   } // public static void main(String[] args)

} // public class AB1

