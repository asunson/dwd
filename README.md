# Distance Weighted Discrimination (DWD) Implementation for Python

These functions are adapted from the R functions written by jcrudy, available at https://github.com/jcrudy/CONOR. (the socp solver was also adapted from his work)

Those functions were previously adapted from the MATLAB functions written by J.S. Marron, available at https://genome.unc.edu/pubsup/dwd

The main function call is `dwd(platform 1, platform2)`. Where each platform is structured as a pandas dataframe with genes on the rows and samples on the columns. The index of each dataframe should be gene names. Column headers are optional. 

Dependencies:

* sklearn
* numpy
* pandas
* clsocp (included)


