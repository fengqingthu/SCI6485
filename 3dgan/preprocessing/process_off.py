# ---------- Copyright (C) 2022 Qing Feng ----------
# 3d dataset preprocessing for 3dgan dataset
# voxelize and convert obj files to mat files
# For MacOs 10.10 above, please install XQuatz

# First, command: chmod 755 meshconv
# Example: python3 process_off.py 64 ../dataset/A_NormalClothes

import os
import sys
import binvox_rw
import scipy.io as io

def main():
    args = sys.argv[1:]
    cube_len = args[0]
    dir = args[1]
    odir = dir + "_obj"
    if not os.path.exists(odir):
        os.system("mkdir " + odir)

    try:
        # loop over .off files in the dir
        for off_file in os.listdir(dir):
            # skip non-off files
            if not off_file.endswith(".off"):
                continue
            # convert to obj file
            ipath = dir + "/" + off_file
            opath = odir + "/" + off_file[:-3] + "obj"
            if not os.path.exists(opath):
                os.system("./meshconv " + ipath + " -c obj -o " + opath)
    finally:
        # process obj
        # os.system("python3 " + "process_obj,py " + cube_len + " " + odir)
        # clean any intermediate files
        # os.system("rm -r " + odir)
        pass

if __name__ == "__main__":
    main()