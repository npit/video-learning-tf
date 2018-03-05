#! /usr/bin/env python3

import os
import sys
import math
import tqdm
import numpy
from PIL import Image
import argparse
import subprocess
import scipy.misc
import matplotlib
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as aF
import librosa
import matplotlib.pyplot as plt
import logging

audio_extensions = ["wav", "mp3"]

def parseArguments():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-i' , '--input_folder', nargs=1, required=True, help="input folder with audio files")
    parser.add_argument('-t' , '--time_slice'  , nargs=1, default=[60], help="generate spectograms of at least <time_slice> seconds per video, always from the center", type=float)
    parser.add_argument('-o','--output_folder',nargs=1, default='.',help="output folder")
    parser.add_argument('-w','--walk_folders',action="store_true",help ="iterate over folders in the input folder")
    args = parser.parse_args()
    links = args.input_folder
    if not os.path.isdir(links[0]):
        parser.error("argument -i/--input--folder must be directory")
    return args

def produceSpectoGrams_Aug(input_folder, output_folder, time_slice, fold_prefix, logger):


    generated_shapes = {}
    file_paths = [ os.path.join(input_folder,f) for f in os.listdir(input_folder)]
    file_paths = [f for f in file_paths if f.split(".")[-1].lower() in audio_extensions]

    if not file_paths:
        logger.info("No suitable files in the folder.")
        return
    t = tqdm.trange(len(file_paths))
    totalFrames = 0
    errorLog = []
    #Create spectogram of each file and write it to the same folder
    for idx,i in enumerate(t):
        file_name = os.path.splitext(file_paths[i])[0]
        [Fs, signal]    = audioBasicIO.readAudioFile(file_paths[i])            
        if isinstance(signal, int):
            continue            
        signal = audioBasicIO.stereo2mono(signal)            
        N = signal.shape[0]
        Win = int(time_slice * Fs)
        curPos = 0    
        countFrames = 0
        if not (curPos + Win - 1 < N):
            msg = "Invalid parameters for file %d/%d : %s, N:%d, pos:%d, Win:%d" % (idx+1,len(t),file_name,N, curPos, Win)
            errorLog.append(msg)
        outfolder = os.path.join(output_folder, fold_prefix,os.path.basename(file_name))
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)

        while (curPos + Win - 1 < N):                        # for each window
            countFrames += 1
            x = signal[curPos:curPos+Win]                    # get current window                        
            curPos += Win
            # SPECTGRAM GENERATION (100x19)
            specgram, _, _  = aF.stSpectogram(x, Fs, round(Fs * 0.020), round(Fs * 0.010), False) 
            specgram = specgram[:, 0:100]   

            # MELGRAM GENERATION (128 x 21)                                
            #specgram = numpy.log10(librosa.feature.melspectrogram(x, Fs, None, int(0.040*Fs), int(0.010*Fs), power=2.0))                
            #specgram = ((specgram + 1) * 22)/255     
            #specgram = specgram.T                                

            ndarr = numpy.uint8(matplotlib.cm.jet(specgram)*255)
            if not ndarr.shape in generated_shapes:
                generated_shapes[ndarr.shape] = 0
            generated_shapes[ndarr.shape] += 1

            im1 = Image.fromarray(ndarr)
            #outfile = file_name.replace(input_folder, spectrRootFolder + os.sep) + "_segment{0:d}.png".format(countFrames)
            outfile = os.path.join(outfolder,"_segment%d.png" % countFrames)

            #print "saving to :",outfile
            scipy.misc.imsave(outfile, im1)
            msg =  "%d frames file %d/%d :  %s" % (countFrames,idx+1,len(t),os.path.basename(file_name))
        totalFrames += countFrames
        t.set_description(msg)


    if errorLog:
        print "\nFailed to produce spects for %d/%d files:" % (len(errorLog), len(t))
    for msg in errorLog:

        logger.error(msg)

        #except Exception as e:
        #    print e
    logger.info("Produced a total of %d frames from %d source videos."%(totalFrames,len(t)))
    logger.info("Image shapes produced:")
    for shp in generated_shapes:
        print(shp, generated_shapes[shp])



# configure logging settings
def configure_logging( logfile, logging_level):
    print("Initializing logging to logfile: %s" % logfile)
    # clear existing logging config from pyaudioanalysis
    root = logging.getLogger()
    map(root.removeHandler, root.handlers[:])
    map(root.removeFilter, root.filters[:])

    logging_level = logging_level
    logger = logging.getLogger('audio-logger')
    logger.setLevel(logging_level)

    formatter = logging.Formatter('%(asctime)s| %(levelname)7s - %(filename)15s - line %(lineno)4d - %(message)s')

    # file handler
    if not logger.handlers:
        handler = logging.FileHandler(logfile)
        handler.setLevel(logging_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # console handler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging_level)
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)
    return logger


if __name__ == '__main__':

    print("in main:",sys.argv)
    args = parseArguments()
    print args
    time_slice = args.time_slice[0]
    input_folder = args.input_folder[0]
    output_folder = args.output_folder[0]
    # append with the time slice value
    output_folder = output_folder[:-1] if output_folder[-1] == os.sep else output_folder
    walk_folders = args.walk_folders


    logfile = "log_spectrograms_ts_%f.log" % time_slice
    logger = configure_logging(logfile,logging.INFO)

    print("Running")
    if not walk_folders:
        logger.info("Extracting from a single folder.")
        fold_prefix = os.path.basename(input_folder)
        logger.info("Extracting spectrograms from %s with a time slice of %4.4f" %  (input_folder, time_slice))
        produceSpectoGrams_Aug(input_folder, output_folder, time_slice, fold_prefix, logger)
    else:
        logger.info("Extracting spectrograms from %s with a time slice of %4.4f" %  (input_folder, time_slice))
        folders = [ fold for fold in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder,fold)) ]
        logger.info("Walking %d %s %s" % (len(folders) ," folders in", input_folder))
        for i,fold in enumerate(folders):
            curr_input_folder = os.path.join(input_folder, fold)
            fold_prefix = os.path.basename(fold)
            logger.info( "Folder %d/%d %s %s %s" % (i+1,len(folders),curr_input_folder,"fold.prefix",fold_prefix))
            produceSpectoGrams_Aug(curr_input_folder, output_folder, time_slice, fold_prefix, logger)



