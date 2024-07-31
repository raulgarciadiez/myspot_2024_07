#!/usr/bin/env python
# coding: utf-8

# In[43]:


import glob, sys, os, subprocess
import statistics
import pandas as pd 


## Input File structure expected:
# Big Header
# %Kinetic Energy %Energy counts
# secondary header
# %Kinetic Energy (repeated) % Ring current (mA)

######################################################################
##                     CHECK THIS:                                  ##
######################################################################

# Path to inputs folder
PATH = os.getcwd() 
# Extension of data file:
extension_dat = "*.dat"
extension = "*.txt"
# Full path to calibration file name:
# Same calibration file for all files to be processed in PATH.
calibration_path = os.path.join(PATH, "Cal_Pt_Red_xps.csv")     #Here you have to put the name of the calibration archive
# Output folder:
output_folder = os.path.join(PATH, "Outputs_Marianne") 

# Delimiters in data file:
# Column delimiter for Ke counts/s
delimiter1 = "  "
# Column delimiter for Ke Ring Current
delimiter2 = " "

# Debug mode:
debug = False

# Normalization criteria:
# Plot counts vs Eb, take nPoints atEnd (else at beginning), take average
# and divide everything by average.
nPoints = 5
atEnd = False

######################################################################
##                          FUNCTION:                               ##
######################################################################

def open_and_save(archives):
    
    for f in archives:

        ####### Loop over files to process:
            bigHeader = []
            isHeader = True
            foundSecondaryHeader = False
            Region = {}

            kineticEnergy = []  # [eV]
            counts  = []
            ringCurrent   = []  # [mA]
            bindingEnergy = []  # [eV]
            excitationEnergy = {}

            # kinetic Energy is repeated, once with counts,
            # and again with ringCurrent, so boolean to indicate if we need to read it.
            readKineticEnergy = True  

            #---------------   LOADING DATA --------------------------# 
            print("Opening file: "+ f)
            with open(f, 'r') as readFile:
                for line in readFile:
                    if line == "\n":   # empty lines
                        continue
                    if "#" in line and isHeader:
                        bigHeader.append(line)
                        if "Region" in line:
                            aux = line.replace("\n","").split(" ")
                            Region['element'] = aux
                        if "Excitation Energy" in line:    
                            aux1 = line.replace("\n","").split(" ")
                            excitationEnergy[0]= float(aux1[-1])   # [eV], convert str->float
                    else:
                        # '#' might still show up in lines because there's a secondary 
                        # header separating energy counts from ring current
                        isHeader = False
                        if "#" in line:
                            foundSecondaryHeader = True
                            continue
                        else:
                            if not foundSecondaryHeader:
                                formatLine = line.replace("\n","").split(delimiter1)
                                kineticEnergy.append(float(formatLine[0]))
                                counts.append(float(formatLine[1]))
                                # calculate binding energy as we read the file:
                                # binding energy = excitation energy - kinetic energy
                                bindingEnergy.append(excitationEnergy[0] - kineticEnergy[-1] )
                            else:
                                formatLine = line.replace("\n","").split(delimiter2)
                                # file split by secondaryHeader, empty line in between.
                                ringCurrent.append(float(formatLine[1]))        

            if (len(kineticEnergy) != len(bindingEnergy) ) or (len(kineticEnergy) != len(counts)) or (len(kineticEnergy) != len(ringCurrent)):
                print("Warning column size mismatch, aborting...")
                sys.exit()

            if debug:
                print("Loaded the following data: ")
                print("KE\t BE\t counts/s\t Ring Current [mA]\t")
                for i in range(0,len(kineticEnergy)):
                    print("{0}\t{1}\t{2}\t{3}".format(kineticEnergy[i], bindingEnergy[i], counts[i], ringCurrent[i]))
            print("Input file closed.")
            print("--------------")

            #---------------   NORMALIZATION --------------------------#

            # compute mean:
            if atEnd:
                normFactor = statistics.mean(counts[-nPoints:])
            else: 
                normFactor = statistics.mean(counts[:nPoints])
            # Norm_int "Normalized intensity":divide everything by mean: 
            Norm_int = [ eC/normFactor for eC in counts ] 

            #--------------- COMPUTE ENERGY OFFSET --------------------#

            # open calibration file:
            calPanda = pd.read_csv(calibration_path)
            # locate energy excitation value in column 'Eex'
            idx =calPanda['Eex'].tolist().index( int(excitationEnergy[0]) ) 
            # look up Eoffset in row idx
            eOffset = calPanda['Eoffset (eV)'].tolist()[idx]

            # Save eOffset:
            BE_offset = [ eOffset for i in range(0,len(bindingEnergy))]
            # Correct by offset:
            BE_cal = [ Eb - eOffset for Eb in bindingEnergy ]
       
    
            #--------------- SAVE TO OUTPUT FILE --------------------# 
       
            filename = os.path.basename(f) #here I get the name of the file I am using 
            outputFileName = filename.split("/")[-1].split(".") #extaction of the file extension
            outputFileName = os.path.join(output_folder, outputFileName[0]+"_PM.txt")
            print("The output File Path is:",outputFileName)


            with open(outputFileName, 'w') as writeFile:
                for line in bigHeader[:-2]:
                    writeFile.write(line)
                # add one extra line to header specifying normalization criteria:
                if atEnd: nStr = "the end"
                else: nStr = "the beginning "
                line = "# Normalization: average of {0} points at {1}\n\n".format(nPoints, nStr)
                writeFile.write(line)
                tableHeader = "# KE\tBE_calc\tNorm_int\n"
                writeFile.write(tableHeader)
                for i in range(0,len(kineticEnergy)):
                    writeFile.write("{0}\t{1}\t{2}\n".format(kineticEnergy[i],  BE_cal[i], Norm_int[i]))


# In[44]:


#print(os.getcwd())


# In[45]:


#print(calibration_path)
#print(output_folder)
#print(extension_dat)
#print(extension)


# In[46]:


######################################################################
##                       DON'T MODIFY THIS:                         ##
######################################################################

# Check if output folder exists, else create it:
if not os.path.isdir(output_folder):
    subprocess.call("mkdir " + output_folder, shell = True)

# collect inputs file names from folder:     
if os.path.join(PATH,extension):
    collectFiles = glob.glob(os.path.join(PATH,extension))   

if os.path.join(PATH,extension_dat):
    collectFiles_dat = glob.glob(os.path.join(PATH,extension_dat))

print("Looking for files inside: " + PATH)
#print("The output folder name is:",output_folder)

if len(collectFiles) != 0:
    print("The extension of the files are:",extension)
    print("Found {0} files".format(len(collectFiles)))
    print("--------------")
    open_and_save(collectFiles)
        
else:
    print("The extension of the files are:",extension_dat)
    print("Found {0} files".format(len(collectFiles_dat)))
    print("--------------")
    open_and_save(collectFiles_dat)   
    


# In[ ]:



   


# In[ ]:




