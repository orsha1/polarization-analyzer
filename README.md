Perovskite Oxides Analysis Tool
===============================

Introduction
------------

This tool is designed for analyzing perovskite oxide structures. It calculates various properties such as polarization, displacement, and alpha angles for both A and B-site cations based on provided configurations.

Configuration
-------------

To use this tool, provide the configuration in a JSON file named `config.json`. An example of the configuration file is provided below:

{  "type":  "vasp",  "path":  "path/to/POSCAR.vasp",  "name":  "perovskite",  "save_pbc_vasp":  true  }

-   `"type"`: Specifies the type of input file. Currently supports only `"vasp"` POSCAR format, `"QE"` input file, and `"QE"` output file.
-   `"path"`: Path to the input file.
-   `"name"`: Name to be used for the output files.
-   `"save_pbc_vasp"`: Whether to save the supercell with periodic boundary conditions (PBC).
-   `"read_all_steps"`: Whether to read all steps of QE output. Will create a folder containing an output for each step in the relaxation.

Required Packages
-----------------

Ensure you have the following Python packages installed:

-   `numpy`
-   `scipy`
-   `ase`
-   `pandas`
-   `tabulate`

Install them using pip:
`pip install numpy scipy ase pandas tabulate`

Usage
-----

1.  Ensure you have the configuration file `config.json` ready with the correct parameters.
2.  Run the provided Python script or the jupyter notbook
python perovskite_analysis.py

Output
------

The tool generates various output files including:

-   Polarization statistics (grep with "!" the single line output)
-   Displacement statistics
-   Alpha angle statistics
-   Detailed summary log
