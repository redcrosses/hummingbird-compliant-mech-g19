import os
import shutil
from ansys.mapdl.core import launch_mapdl

#YOU DON'T NEED TO RUN THIS SCRIPT TO RUN ANSYS WITH THE INPUT FILES, BUT THIS WILL RUN THE ENTIRE ROUTINE HEADLESS
def run_ansys(ident_str, input_dir, show=False):
    run_dir = r"C:\Users\denki\Documents\ansys-apdl\compliant\hummingbird"+ f"-{ident_str}" #has to be global as well (will be created if it doesn't exist)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)

    mapdl = launch_mapdl(run_location=run_dir, override=True)
    mapdl.finish()
    mapdl.clear()

    for file in os.listdir(input_dir):
        print(f'Running {file}...')
        target_file = os.path.join(input_dir, file)
        output = mapdl.input(target_file)
        mapdl.run('/SHOW')
        print(output)
    
    #put the results into working directory
    copy_tar = os.path.join(os.getcwd(), f"results_post26_{ident_str}.txt")
    print(copy_tar)
    shutil.copyfile(os.path.join(run_dir, r'results_post26.txt'), copy_tar)
    
    if show:
        mapdl.open_gui()
        mapdl.exit()
    else: 
        mapdl.exit()


