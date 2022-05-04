import os
import sys
import scipy.io as io
import numpy as np

def main():
    args = sys.argv[1:]
    cube_len = int(args[0])
    vsize = 1/cube_len*1000

    dir = args[1]
    odir = dir + "_csv"
    if not os.path.exists(odir):
        os.system("mkdir " + odir)
    
    try:
        # loop over .ply files in the dir
        for mat_file in os.listdir(dir):
            # skip non-obj files
            if not mat_file.endswith(".mat"):
                continue
            path = dir + "/" + mat_file
            voxels = io.loadmat(path)['instance']
            head = []
            xarr = []
            yarr = []
            zarr = []
            index = 0
            for x in range(cube_len):
                for y in range(cube_len):
                    for z in range(cube_len):
                        if voxels[x,y,z]:
                            head.append(index)
                            xarr.append(x*vsize)
                            yarr.append(y*vsize)
                            zarr.append(z*vsize)
                            index += 1
            
            head = np.array(head)
            xarr = np.array(xarr)
            yarr = np.array(yarr)
            zarr = np.array(zarr)
            np.savetxt(odir + "/" + mat_file[:-3] + "csv", np.column_stack((head, xarr, yarr, zarr)), fmt = '%.15f', delimiter=',')
            print(f"---------- Processed {mat_file} ----------")
    
    except:
        print("!!! Processing failed !!!")
    finally:
        print("---------- Preprocessing finished ----------\n")
            
if __name__ == "__main__":
    main()