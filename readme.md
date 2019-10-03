[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/molinaris/nerlab.git/master)

# NEUROMUSCULAR MODEL NOTEBOOK
The neuromuscular model notebook is an interactive tool for studying the underlying mechanisms of electromyogram (EMG) and force generation. The model, which was based on previous studies (see references below), can be used for research purposes as well as a teaching platform.

The notebook was created by [Ricardo G. Molinari](https://github.com/molinaris) under the supervision of [Dr. Leonardo A. Elias](https://github.com/leoelias-unicamp) at the [Neural Engineering Research Laboratory](http://www.fee.unicamp.br/deb/leoelias/ner-lab?language=en), [University of Campinas](http://www.unicamp.br/unicamp/english) (Brazil).

## Installation Instructions
### Dependencies
- [Anaconda](https://www.anaconda.com/) (a Python Data Science Plataform)
  - If you already have an Anaconda distribution installed in your computer, it is important to update it:

    `conda update conda`

### Creating the Environment
After installing Anaconda, you have to create a Python 3.7 environment with the following packages: NumPy, Matplotlib, SciPy, Pandas, Ipywidgets, and Plotly. Use the following command in Anaconda terminal:

`conda create -n nb_env python=3.7.0 numpy=1.17.0 matplotlib=3.1.1 scipy=1.1.0 pandas=0.23.4 ipywidgets=7.4.1`

### Activating the environment
- Windows users: `activate nnm_env`
- Linux/MacOS users: `source activate nnm_env`

### Running the neuromuscular model notebook
- Use the command `jupyter notebook` to start the Jupyter notebook application.
- In Jupyter notebook, open the file 'NeuromuscularModel.ipynb'.
- It is highly recommended to execute each cell in the order of appearance, since there are dependences between the cells.

## References
1. [Cisi, R., Kohn, A., 2008.](https://dx.doi.org/10.1007/s10827-008-0092-8)Simulation system of spinal cord motor nuclei and associated nerves and muscles, in a Web-based architecture. Journal of Computational Neuroscience, 25(3), 520-542.
2. [Challis and Kitney, 1990](https://doi.org/10.1007/BF02442601). Biomedical signal processing (in four parts) - Part 1 Time-domain methods. Medical & Biological Engineering & Computning, 28(6), 509-524.
3. [Challis and Kitney, 1991](https://doi.org/10.1007/BF02446704). Biomedical signal processing (in four parts) - Part Part 3 The power spectrum and coherence function. Medical & Biological Engineering & Computning, 29(3), 225-241.
4. [Enoka and Fuglevand, 2001](https://doi.org/10.1002/1097-4598(200101)24:1<4::AID-MUS13>3.0.CO;2-F). Motor unit physiology: Some unresolved issues. Muscle and Nerve, 24(1), 4-17.
5. [Fuglevand, A. J., Winter, D. A., & Patla, A. E., 1993.](https://doi.org/10.1152/jn.1993.70.6.2470) Models of recruitment and rate coding organization in motor-unit pools. Journal of Neurophysiology, 70(6), 2470–88.
6. [Johnson et al., 1973](https://doi.org/10.1016/0022-510X(73)90023-3). Data on the distribution of fibre types in thirty-six human muscles. Journal of the Neurological Sciences, 18(1), 111-129.
7. [Kernell D., 2006](https://doi.org/10.1093/acprof:oso/9780198526551.001.0001). The Motoneurone and its Muscle Fibres. Published to Oxford Scholarship Online, September, 2009.
8. [Lo Conte et al., 1994](http://doi.org/10.1109/10.335863).  Hermite Expansions of Compact Support Waveforms: Applications to Myoelectric Signals. IEEE Transactions on Biomedical Engineering, 41(12), 1147-1159.
9. [Yao, W., Fuglevand, R. J., Enoka, R. M., 2000.](https://doi.org/10.1152/jn.2000.83.1.441) Motor-unit synchronization increases EMG amplitude and decreases force steadiness of simulated contractions. Journal of Neurophysiology, 83(1), 441-452.
10. [Zhou, P., Rymer, W. Z.,2004.]( https://doi.org/10.1152/jn.00367.2004) Factors governing the form of the relation between muscle force and the EMG: a simulation study. Journal of Neurophysiology, 92(5), 2878-2886.


## Funding
This project was funded by [CNPq](http://www.cnpq.br/) (Brazilian NSF): proc. #409302/2016-3 and #312442/2017-3.