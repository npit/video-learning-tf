import os, sys
import subprocess

fps=25
extension = "avi"
endtime="23:59:59"
vidfile, shotfile = tuple(sys.argv[1:])
vidname = os.path.basename(vidfile)
outfolder = "video_parts"
outvidfolder = os.path.join(outfolder, vidname)

if not os.path.exists(outfolder):
    os.makedirs(outvidfolder)

def sec_from_nframes(nframes,nfps):
    return nframes / nfps

def sec_to_hms(duration_sec):
    m, s = divmod(duration_sec, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

# read frame numbers
with open(shotfile,"r") as f:
    for line in f:
        framenums = [ int(n) for n in line.strip().split() ]


# every chunk has to be defined as a tuple: startTIME, durationSECOND

# calc time pairs
timepairs = []
curr_start = 0
print("start-sec duration start_hms duration_hms")
for idx,frn in enumerate(framenums):
    start_hms = sec_to_hms(curr_start)

    duration = sec_from_nframes(frn-curr_start, fps)
    duration_hms = sec_to_hms(duration)
    timepairs.append((start_hms, duration_hms))
    print(" %f %f " % (curr_start, duration))
    curr_start += duration

print()
print("resulting frame nums - timetags")
timepairs.append((sec_to_hms(curr_start),endtime  ))
framenums = [0] + framenums + [-1]
for tp, frm in zip(timepairs, framenums):
    print(frm, tp)
# use -c copy for ffmpeg

cmds = []
for idx, tp in enumerate(timepairs):
    outfile = os.path.join(outvidfolder, "video_" + str(idx) + "." + extension)
    cmd=[]
    cmd.extend('ffmpeg -loglevel warning -y -i'.split() + [vidfile] + "-c copy -ss".split() + [tp[0],"-t",tp[1],"-sn",outfile])
    print(" ".join(cmd))
    process=subprocess.run(cmd)

