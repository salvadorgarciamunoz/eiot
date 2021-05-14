# Modules
## eiot
Perform Extended Iterative Optimization Technology (EIOT), Supervised EIOT, and PLS-based EIOT. Developed by Sal Garcia (sal.garcia@lilly.com / sgarciam@ic.ac.uk / salvadorgarciamunoz@gmail.com)

More details available in the paper:
Garcia-Munoz, S. and Hernandez Torres, E., 2020. Supervised Extended Iterative Optimization Technology for the estimation of powder compositions in pharmaceutical applications: Method and lifecycle management. Industrial & Engineering Chemistry Research.

https://pubs.acs.org/doi/pdf/10.1021/acs.iecr.0c01385

## eiot_extras
Helper functions for plotting, data processing, and analysis.

# Getting Started with Python (pyEIOT)

## Dependencies

EIOT requires the following python packages: pyphimva (https://github.com/salvadorgarciamunoz/pyphi), numpy, matplotlib, scipy, pyomo, pandas, bokeh. pyphimva (pyphi) is not yet on pip or conda, so it must be downloaded and installed first. The rest can be installed via setup.py below or manually using pip/conda and the ```requirements.txt``` file.

## Required External Dependencies

- IPOPT as an executable in your system path.
  - Windows: ```conda install -c conda-forge IPOPT=3.11.1``` or download from [IPOPT releases page](https://github.com/coin-or/Ipopt/releases), extract and add the IPOPT\bin folder to your system path or add all files to your working directory.
  - Mac/Linux: ```conda install -c conda-forge IPOPT```, download from [IPOPT releases page](https://github.com/coin-or/Ipopt/releases), or [Compile using coinbrew](https://coin-or.github.io/Ipopt/INSTALL.html#COINBREW).
- libhsl with ma57 within library loading path or in the same directory as IPOPT executable.
   - Speeds up IPOPT for large problems but requires a free academic or paid industrial license and a local IPOPT installation.
   - Must request in advance and building the source code is nontrivial. Expert use only.
Note:
-  EIOT has been configured to use either an installed GAMS python module or GAMS executable in your system path; however, an apparent bug in the pyomo-gams interface results in occasional constraint violation infeasibilities and very slow load times.
- Unlike pyphimva, EIOT solves many, MANY Pyomo NLPs. Using the NEOS server has too much overhead to be useful.

Adding a folder to your system path:
 - Windows: temporary ```set PATH=C:\Path\To\ipopt\bin;%PATH%``` or persistent ```setx PATH=C:\Path\To\ipopt\bin;%PATH%```.
 - Mac/Linux: ```export PATH=/path/to/ipopt:$PATH```, add to .profile/.*rc file to make persistent.
 - Both via Conda: after activating your environment, use ```conda env config vars set``` and your OS-specific set or export command.

## Installation
1) Ensure you have Python 3 installed and accessible via your terminal ("python" command).
   - It's strongly encouraged you to create a virtual environment using anaconda (```conda create -n your_eiotenv python```) or venv (```pip -m venv your_eiotenv```). You can then activate your environment ```conda activate yourenv``` or venv Windows ```yourenv\Scripts\activate.bat``` or venv Linux/mac ```source yourenv/bin/activate``` ) and then install into a sandboxed environment.
1*) Open https://github.com/salvadorgarciamunoz/pyphi and follow the (similar) installation instructions to install the pyphi module from pyphimva.
2) Download this repository via ```git clone``` or manually using the download zip button at the top of the page.
3) Open a command terminal and navigate to your repository's "pyEIOT" directory.
4) Run ```python setup.py install``` to install the ```eiot```, ```eiot_extras``` and any remaining dependencies.

To confirm you have a working installation, navigate to the ```examples``` folder and copy the ```Sample_Script_EIOTwPLS.py``` file to the directory of your choice. Run it using ```python Sample_Script_EIOTwPLS.py``` and verify that there are no errors logged to the console. This code may take more than 6 minutes and requires you to "X" out of the diagnostic graphs that show up to move forward.

## Getting Started with MATLAB (mEIOT)
1) Download this repository via ```git clone``` or manually using the download zip button at the top of the page.
2) Open Matlab and in the command Window enter ```addpath("C:/path/to/your/eiot/mEIOT/EIOT_Toolbox");```. This will make EIOT available to you for the session. To add it automatically, on the top ribbon select "Set Path". Move the EIOT_Toolbox folder into one of the listed folders or add its location to your default path using the pop-up.
