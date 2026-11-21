import os
from ansys.mapdl.core import launch_mapdl

#YOU DON'T NEED TO RUN THIS SCRIPT TO RUN ANSYS WITH THE INPUT FILES, BUT THIS WILL RUN THE ENTIRE ROUTINE HEADLESS

input_dir = r"C:\Users\denki\Documents\GitHub\hummingbird-compliant-mech-g19\ansys-scripts" #has to be global (change to yours)
run_dir = r"C:\Users\denki\Documents\ansys-apdl\compliant\hummingbird-v2" #has to be global as well (change to yours)
os.makedirs(run_dir, exist_ok=True)

mapdl = launch_mapdl(run_location=run_dir, override=True)
mapdl.finish()
mapdl.clear()

for file in os.listdir(input_dir):
    target_file = os.path.join(input_dir, file)
    output = mapdl.input(target_file)
    mapdl.run('/SHOW')
    print(output)
mapdl.exit()


