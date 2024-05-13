/*
* Alex Zhong
* Created 24 April 2024
* 
* This class does the necessary conversions of image files in order to fit as inputs for N-layer.
*
* 1) Take your photos
* 2) Convert them to BMP format using GimpShop or other methods
* 3) Use BMP2OneByte.java to extract single byte gray scale values in the range 0 to 255 from the image
* 4) Write the code needed to read in the one byte values into a 2D array of integers (you need to specify the known size)
* 5) Use the PelArry class to manipulate the array of integer values that represent the image
*    - Use onesComplimentImage() if you have a white background
*    - Find the center of mass of the PelArry
*    - Crop around the center of mass. The cropping is done with hard coded values, so you will need to experiment to get it right.
*    - Save the one byte values in the now cropped PelArray to a file
*    - Convert that binary data file back into a BMP image using BGR2BMP.java
*    - Look at the images, modify the cropping and repeat until they all look good
* 6) Convert the now good PelArry files (scale if 0 to 255 per picture element) into activation files in the range 0 to 1
*/

import java.io.*;

public class ImageEditing
{
   private int[][] intArray;
   private final int ROW = 500;
   private final int COL = 400;

   public void BINtoInt(String binFile) throws IOException
   {
      FileInputStream fstream = new FileInputStream(binFile);

      // Convert our input stream to a DataInputStream
      DataInputStream in = new DataInputStream(fstream);

      intArray = new int[ROW][COL];
      for (int i = 0; i < ROW; i++)
      {
         for (int j = 0; j < COL; j++)
         {
            intArray[i][j] = (int) in.readByte();
         }
      }

      in.close();
   }

   public void printIntArray()
   {
      for (int i = 0; i < ROW; i++)
      {
         for (int j = 0; j < COL; j++)
         {
            System.out.printf("%d ", intArray[i][j]);
         }
         System.out.printf("%n");
      }
   }

   public int[][] getIntArray()
   {
      return intArray;
   }

   public static void main(String args[]) throws IOException
   {
      for (int a = 1; a <= 6; a++)
      {
         for (int b = 1; b <= 5; b++)
         {
            // String rawImage = "Hand Images/Raw BMP/IMG_6535.bmp";
            // String binImage = "Hand2.3.bin";

            // if ((a == 1 && b == 1) || (a == 1 && b == 2) || (a == 1 && b == 3) )
            // {
            //    continue;
            // }
            // String[] arguments = new String[] {rawImage, binImage};
            String c = a + "." + b;
            String input[];

            String ogBinFile = "bin/Hand" + c + ".bin";
            //  String bmpFile = "Hand Images/BMP/" + c + ".bmp";
            // input = new String[2];
            // input[0] = bmpFile;
            // input[1] = ogBinFile;
            // BMP2OneByte.main(input);

            ImageEditing az = new ImageEditing();
            az.BINtoInt(ogBinFile);

            PelArray image = new PelArray(az.getIntArray());

            PelArray offset = image.offsetColors(0, 0, -30);

            PelArray saturate = offset.saturate(5);

            //PelArray offset2 = image.offsetColors(0, 0, -70);

            //PelArray gray = offset.grayScaleImage();

            PelArray blue = saturate.oneColorImage(PelArray.BLUE);

            PelArray crop = blue.crop(130, 180, 389, 389);

            if (a == 6 && b == 5)
            {
               crop = blue.crop(130, 190, 389, 399);
            }

            PelArray centered = crop.offset(crop.getWidth()/2 - crop.getXcom(), crop.getHeight()/2 - crop.getYcom());
            //System.out.printf("COM = %d, %d\n", blue.getXcom(), blue.getYcom());

            int[][] intArray = centered.getPelArray();

            int width = intArray[0].length;
            int height = intArray.length;

            System.out.printf("Image Width %d\n", width);
            System.out.printf("Image Height %d\n\n", height);

            String pelBinFile = "Pray/Pel/Hand" + c + "Pel.bin";
            FileOutputStream fstream = new FileOutputStream(pelBinFile);
            DataOutputStream out = new DataOutputStream(fstream);

            for (int i = 0; i < height; i++)
            {
               for (int j = 0; j < width; j++)
               {
                  if (intArray[i][j] < 0)
                  {
                     out.writeByte(0);
                  }
                  else
                  {
                     out.writeByte(intArray[i][j]);
                  }
               }
            }
            out.close();

            String intBinFile = "Pray/Int/Hand" + c + "Pel.bin";
            fstream = new FileOutputStream(intBinFile);
            out = new DataOutputStream(fstream);

            for (int i = 0; i < height; i++)
            {
               for (int j = 0; j < width; j++)
               {
                  if (intArray[i][j] < 0)
                  {
                     System.out.println("negative encountered");
                     out.writeInt(0);
                  }
                  else
                  {
                     out.writeInt(intArray[i][j]);
                  }
               }
            }
            out.close();

            input = new String[5];
            input[0] = Integer.toString(0);
            input[1] = Integer.toString(width);
            input[2] = Integer.toString(height);
            input[3] = intBinFile;
            input[4] = "Pray/Processed/Hand" + c + "Processed.bmp";

            BGR2BMP.main(input);
         }
      }
   } // public static void main(String args[])
} // public class ImageEditing