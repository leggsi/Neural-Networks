package AB1;
/*
 * Alex Zhong
 * Created 30 January 2024
 * 
 * This file trains and/or runs an A-B-1 Network and adheres to the design document provided by Dr. Eric Nelson.
 * Calculation of the error function for training uses gradient descent.
 * 
 * ==== Methods ====
 * rand: generates a random number between a specified range
 * setConfigParams: sets configuration parameters to hard-coded values
 * echoConfigParams: prints the configuration parameters that were set
 * allocateMemoryTrain: creates the arrays needed to train the neural network
 * allocateMemoryRun: creates the arrays needed to run the network
 * populateArrays: populates the inputs, expected outputs, and weights arrays
 * populateInputs: populates the inputs array to hard-coded values
 * populateXOR: populates the expected output arrays to XOR
 * populateAND: populates the expected output arrays to AND
 * populateOR: populates the expected output arrays to OR
 * populateWeightsRandom: utilizes the rand function to populate weights arrays with randomized values
 * setWeightshardCode: sets the weight arrays to hard-coded values (used for run)
 * runTrain: takes in an input array and runs the network based on current weights (jTheta array)
 * runRun: takes in an input array and runs the network based on current weights (jTheta variable)
 * runCases: takes in a 2D-array of inputs and applies the "run" method to each row of inputs
 * train: trains the network using gradient descent
 * printTrainResults: prints the results of training
 * printRunResults: prints the results of running the network on the hard-coded weights
 * sigmoid: applies the sigmoid function to an input and returns the new value
 * sigmoidDeriv: applies the sigmoid derivative function to an input and returns the new value
 * main: creates an instance of the AB1 class and puts everything together
 */


public class AB1
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
* randWeights - true if populate weights randomly, false to hard-code
*/
   int nInputs, nHidden, nOutputs, cases, maxIter;
   double randMin, randMax, lambda, error;
   boolean training, randWeights;

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
* omega -- omega value for output activation/node
* thetaJ - accumulator for hidden activations for running
* theta0 - theta value for output activation/node
* psi0 --- psi value for output activation/node
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
   double[] a, h, jTheta, jOmega, jPsi;
   double f0, omega, thetaJ, theta0, psi0, caseError;
   double[][] kWeights, kDeltaW, kDeriv;
   double[][] jWeights, jDeltaW, jDeriv;
   double[] allError;
   double[][] inputs;
   double[] eOutputs, cOutputs;

/*
* Instance values for the training process; printed for the user
* 
* iter ----- current iteration of the training process
* avgError - average error of the current iteration
*/
   int iter;
   double avgError;

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
* setConfigParams sets the configuration parameters to hard-coded values
* See variable declarations for clarification on variable names
*/
   public void setConfigParams()
   {
      nInputs  = 2;
      nHidden  = 5;
      nOutputs = 1;

      training = true;
      randWeights = true;
      cases = 4; // test cases

      if (training)
      {
         randMin = -1.5;
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
* allocateMemoryTrain creates the arrays required for the training process and allocates memory
*/
   public void allocateMemoryTrain()
   {
      // Allocation for activations and associated values
      a = new double[nInputs];
      h = new double[nHidden];
      jTheta = new double[nHidden];
      jOmega = new double[nHidden];
      jPsi   = new double[nHidden];

      // Allocation for weights and associated values
      kWeights = new double[nInputs][];
      kDeltaW  = new double[nInputs][];
      kDeriv   = new double[nInputs][];

      for (int k = 0; k < nInputs; k++)
      {
         kWeights[k] = new double[nHidden];
         kDeltaW[k]  = new double[nHidden];
         kDeriv[k]   = new double[nHidden];
      }

      jWeights = new double[nHidden][];
      jDeltaW  = new double[nHidden][];
      jDeriv   = new double[nHidden][];

      for (int j = 0; j < nHidden; j++)
      {
         jWeights[j] = new double[nOutputs];
         jDeltaW[j]  = new double[nOutputs];
         jDeriv[j]   = new double[nOutputs];
      }

      inputs   = new double[cases][nInputs];
      eOutputs = new double[cases];
      cOutputs = new double[cases];

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
      kWeights = new double[nInputs][];
      jWeights = new double[nHidden][];
 
      for (int k = 0; k < nInputs; k++) { kWeights[k] = new double[nHidden]; }

      for (int j = 0; j < nHidden; j++) { jWeights[j] = new double[nOutputs]; }

      inputs   = new double[cases][nInputs];
      eOutputs = new double[cases];
      cOutputs = new double[cases];
   } //public void allocateMemoryRun()


/*
* populateArrays populates the inputs, expected outputs, and weights arrays
*/
   public void populateArrays()
   {
      populateInputs();
      
      populateXOR(); // change accordingly for XOR, AND, OR (expected outputs)

      if (randWeights = true)
      {
         populateWeightsRandom();
      }
      else
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
* populateXOR populates the expected outputs array with XOR
*/
   public void populateXOR()
   {
      eOutputs[0] = 0.0;
      eOutputs[1] = 1.0;
      eOutputs[2] = 1.0;
      eOutputs[3] = 0.0;
   }

/*
* populateAND populates the expected outputs array with AND
*/
   public void populateAND()
   {
      eOutputs[0] = 0.0;
      eOutputs[1] = 0.0;
      eOutputs[2] = 0.0;
      eOutputs[3] = 1.0;
   }

/*
* populateOR populates the expected outputs array with OR
*/
   public void populateOR()
   {
      eOutputs[0] = 0.0;
      eOutputs[1] = 1.0;
      eOutputs[2] = 1.0;
      eOutputs[3] = 1.0;
   }
   
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
   public double runTrain(double[] inputs)
   {
      a = inputs;

      for (int j = 0; j < nHidden; j++) // Sets h[j] = ƒ(Θj) , θj = ∑(a[k] * w[k][j])
      {
         jTheta[j] = 0.0;

         for (int k = 0; k < nInputs; k++)
         {
            jTheta[j] += a[k] * kWeights[k][j];
         }

         h[j]= sigmoid(jTheta[j]);

      } // for (int j = 0; j < nHidden; j++)

      theta0 = 0.0;
      for (int j = 0; j < nHidden; j++) // Calculates θzero = ∑(h[j] * w[j][0])
      {
         theta0 += h[j] * jWeights[j][0];
      }
      f0 = sigmoid(theta0);

      return f0;
   } // public void runTrain(double[] inputs)

/*
* runRun takes in an array of inputs, then runs the network based on current weights without a theta array
* 
* @param inputs the activations for the first layer
* @return output value
*/
   public double runRun(double[] inputs)
   {
      a = inputs;

      for (int j = 0; j < nHidden; j++) // Sets h[j] = ƒ(Θj) , θj = ∑(a[k] * w[k][j])
      {
         thetaJ = 0.0;

         for (int k = 0; k < nInputs; k++)
         {
            thetaJ += a[k] * kWeights[k][j];
         }

         h[j]= sigmoid(thetaJ);

      } // for (int j = 0; j < nHidden; j++)

      theta0 = 0.0;
      for (int j = 0; j < nHidden; j++) // Calculates θzero = ∑(h[j] * w[j][0])
      {
         theta0 += h[j] * jWeights[j][0];
      }
      f0 = sigmoid(theta0);

      return f0;
   } // public void runRun(double[] inputs)

/*
* runCases runs each test case (row) of the inputs[][] array
*/
   public void runCases()
   {
      for (int ind = 0; ind < cases; ind++)
      {
         cOutputs[ind] = runRun(inputs[ind]);
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
            cOutputs[caseIter] = runTrain(inputs[caseIter]); // runs test case of current iteration, also sets theta0 and f0

                                                                // ===== n = 3 (output) layer =====
            omega = eOutputs[caseIter] - f0;                    // ω0 = (T0 − F0)
            psi0 = omega * sigmoidDeriv(theta0);                // ψ0 = ω0 * ƒ'(Θzero)

            for (int j = 0; j < nHidden; j++)
            {
               jDeriv[j][0]  = -h[j] * psi0;                    // ∂E/∂w[j][0] = −h[j] * ψ0
               jDeltaW[j][0] = -lambda * jDeriv[j][0];          // Δw[j][0] = −λ * ∂E/∂w[j][0]

                                                                // ===== n = 2 (hidden) layer =====
               jOmega[j] = psi0 * jWeights[j][0];               // Ωj = ψ0 * w[j][0]
               jPsi[j]   = jOmega[j] * sigmoidDeriv(jTheta[j]); // Ψj = Ωj * ƒ'(Θj)

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
                  jWeights[j][0] += jDeltaW[j][0];
                  kWeights[k][j] += kDeltaW[k][j];
               }
            }

            caseError = (omega * omega) / 2.0; // E = ω0^2 / 2; when error is calculated should not matter

            allError[caseIter] = caseError; // error for the specific test case

         } // for (int caseIter = 0; caseIter < inputs.length; caseIter++)

         // calculate average error of all cases
         avgError = 0.0;
         for (int c = 0; c < cases; c++)
         {
            avgError += allError[c];
         }
         avgError /= cases;

         iter++;

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

      // code below formats inputs, expected, and calculated into a table format
      System.out.printf(" __Inputs___ ___Expected__ __Calculated__%n");
      System.out.printf("|           |             |              |%n");
      for (int ind = 0; ind < cases; ind++)
      {
         System.out.printf("|  ");
         for (double input : inputs[ind])
         {
            System.out.printf("%s ", input);
         }
         System.out.printf(" |     %s     |   %.6f   |", eOutputs[ind], cOutputs[ind]);
         System.out.printf("%n|           |             |              |%n");
      } // for (int ind = 0; ind < cases; ind++)

      System.out.printf(" ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾%n");

   } // public void printTrainResults()

/*
* printRunResults displays the inputs followed by expected and calculated outputs
*/
   public void printRunResults()
   {
      // code below formats inputs, expected, and calculated into a table format
      System.out.printf(" __Inputs___ ___Expected__ __Calculated__%n");
      System.out.printf("|           |             |              |%n");
      for (int ind = 0; ind < cases; ind++)
      {
         System.out.printf("|  ");
         for (double input : inputs[ind])
         {
            System.out.printf("%s ", input);
         }
         System.out.printf(" |     %s     |   %.6f   |", eOutputs[ind], cOutputs[ind]);
         System.out.printf("%n|           |             |              |%n");
      } // for (int ind = 0; ind < cases; ind++)

      System.out.printf(" ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾%n");

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
      AB1 p = new AB1();

      p.setConfigParams();

      if (p.training)
      {
         p.allocateMemoryTrain();
         p.populateArrays();
         p.echoConfigParams();
         p.train();
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

