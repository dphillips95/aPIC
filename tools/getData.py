import os
import sys
import math
import numpy as np
import functools as ftools
import h5py
import subprocess
import xarray as xr
import multiprocessing as mp
import re
import argparse
import scipy.constants as const
from time import sleep
from contextlib import nullcontext

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--Ncores", help = "Number of CPU threads",
                    type = int, default = 1)
parser.add_argument("--step", help = "Time step selection cadence",
                    type = int, default = 1)
args = parser.parse_args()

def natural_sort_key(s, _nsre=re.compile(r'(\d+)')):
   return [int(text) if text.isdigit() else text.lower()
           for text in _nsre.split(s)]

def parseParameter(paramGetCmd, paramUnit = 1, indices = 0, return_func = float, emptyErr = False):
   # Read parameter via shell command.
   try:
      res = subprocess.run(paramGetCmd, shell = True, capture_output = True,
                           text = True, check = True)
   except subprocess.CalledProcessError:
      print("ERROR: shell command failed: " + paramGetCmd)
      sys.exit()
   res_array = np.array(res.stdout.split())
   if len(res_array) == 0 and emptyErr is True:
      raise RuntimeError("Process returned empty")
   if return_func == str:
      if indices is not None:
         return res_array[indices]
      return res_array
   if isinstance(indices, int):
      return return_func(res_array[indices])*paramUnit
   if indices is not None:
      return [return_func(x)*paramUnit for x in res_array[indices]]
   return [return_func(x)*paramUnit for x in res_array]

def getData(proc,nx,ny,nz):
   proc_name = proc.split('.')[0]
   print("Reading Data from " + proc_name)
   with h5py.File(proc, 'r') as f:
      field_data = np.array(f['level_0']['data:datatype=0']).reshape(-1,nx,ny,nz)
      data = dict(zip(field_list,np.array(field_data)))
   
   return data

proc_data = [x for x in sorted(os.listdir("./"), key = natural_sort_key) if x.startswith("plt") and x.endswith(".h5")]

with h5py.File(proc_data[-1], 'r') as f:
   field_count = f.attrs["num_components"].item()
   field_list = []
   for ii in range(field_count):
      field_list.append(f.attrs["component_" + str(ii)].decode("UTF-8"))

print("Reading parameters and settings")

params = dict()
qom = []
with open('input.cfg', 'r') as f:
   for line in f:
      if line.startswith("#") or line == "\n":
         continue
      field,value = line.split("=")
      field,subfield = field[:-1].split(".")
      value = value[1:-1]
      match field:
         case "simulation":
            match subfield:
               case "steps":
                  params["Ncycles"] = int(value)
               case "dt":
                  params["dt"] = float(value)
               case _:
                  continue
         case "domain":
            match subfield:
               case "x_min":
                  params["x_min"] = float(value)
               case "x_max":
                  params["x_max"] = float(value)
               case "y_min":
                  params["y_min"] = float(value)
               case "y_max":
                  params["y_max"] = float(value)
               case "z_min":
                  params["z_min"] = float(value)
               case "z_max":
                  params["z_max"] = float(value)
               case "dx":
                  params["dx"] = float(value)
               case "dy":
                  params["dy"] = float(value)
               case "dz":
                  params["dz"] = float(value)
               case "x_size":
                  params["nx"] = int(value)
               case "y_size":
                  params["ny"] = int(value)
               case "z_size":
                  params["nz"] = int(value)

   if "dx" in params.keys():
      params["x_min"] = -params["dx"]*params["nx"]/2
      params["x_max"] = params["dx"]*params["nx"]/2
   if "dy" in params.keys():
      params["y_min"] = -params["dy"]*params["ny"]/2
      params["y_max"] = params["dy"]*params["ny"]/2
   if "dz" in params.keys():
      params["z_min"] = -params["dz"]*params["nz"]/2
      params["z_max"] = params["dz"]*params["nz"]/2
   
   params["Lx"] = params["x_max"] - params["x_min"]
   params["Ly"] = params["y_max"] - params["y_min"]
   params["Lz"] = params["z_max"] - params["z_min"]

cycle_nos = np.array(sorted(int(x[3:-3]) for x in proc_data if int(x[3:-3]) % args.step == 0))
params["nt"] = int(proc_data[-1][3:-3])

print("Reading Data")

if __name__ == '__main__':
   with mp.Pool(args.Ncores) if args.Ncores != 1 else nullcontext() as pool:
      getData_loc = ftools.partial(getData, nx = params["nx"], ny = params["ny"], nz = params["nz"])
      
      if pool is None:
         res = list(map(getData_loc, proc_data))
      else:
         res = list(pool.map(getData_loc, proc_data))

data = {key:np.array([x[key].copy() for x in res]) for key in field_list}

del res

var_list = field_list.copy()

d_i = np.mean((params["Lx"]/params["nx"],params["Ly"]/params["ny"],params["Lz"]/params["nz"])).item()

B = np.sqrt(data["Bx"]**2 + data["By"]**2 + data["Bz"]**2).mean().item()

# rho = (data["rho"][1].mean().item()*(4*math.pi))/d_i**3

# rho_i = []
# for var in sorted(var_list, key = natural_sort_key):
#    if var.startswith("rho_"):
#       rho_i.append((data[var][1].mean().item()*(4*math.pi))/d_i**3)
# rho_i = np.array(rho_i)

# dens = np.abs(rho_i)/const.e

combined_data = {key:(("t","x","y","z"),val) for key,val in data.items()}
del data

xlocs = np.linspace(0, params["Lx"], params["nx"])
ylocs = np.linspace(0, params["Ly"], params["ny"])
zlocs = np.linspace(0, params["Lz"], params["nz"])
ticlocs = np.array(cycle_nos)
tlocs = params["dt"]*ticlocs

combined_data["timestep"] = (("t",),ticlocs)

coords = {"t":tlocs, "x":xlocs, "y":ylocs, "z":zlocs}

print("Writing dataset to netcdf file")

dataset = xr.Dataset(combined_data, coords)

dataset.to_netcdf("data.nc", engine = 'h5netcdf')
