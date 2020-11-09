
# NEUROMUSCULAR MODEL NOTEBOOK
The neuromuscular model notebook is an interactive tool for studying the underlying mechanisms of muscle force and electromyogram (EMG) generation. The model, which was based on previous studies (see references below), can be used for research purposes as well as a teaching/learning platform.

The notebook was created by [Ricardo G. Molinari](https://github.com/molinaris) under the supervision of [Dr. Leonardo A. Elias](https://github.com/leoelias-unicamp) at the [Neural Engineering Research Laboratory](http://www.fee.unicamp.br/deb/leoelias/ner-lab?language=en), [University of Campinas](http://www.unicamp.br/unicamp/english) (Brazil).

Open this notebook with myBinder [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/molinaris/neuromuscular_notebook/version0.4?filepath=neuromuscular_note.ipynb) or sign in to your Google account to open it on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molinaris/neuromuscular_notebook/blob/version0.4/neuromuscular_note.ipynb)

To run the notebook on your local machine, please follow the installation instructions provided below.

## Installation Instructions
### Dependencies
- [Anaconda](https://www.anaconda.com/) (a Python Data Science Plataform)
  - If you already have an Anaconda distribution installed in your computer, it is important to update it:

    `conda update conda`

### Creating the Environment
After installing Anaconda, you have to create a Python 3.7 environment with the following packages: NumPy, Matplotlib, SciPy, Pandas, Ipywidgets, and Plotly. Use the following command in Anaconda terminal:

`conda create -n nemu_env python=3.7.0 numpy=1.17.0 matplotlib=3.1.1 scipy=1.1.0 pandas=0.23.4 ipywidgets=7.4.1`

### Activating the environment
- Windows users: `activate nemu_env`
- Linux/MacOS users: `source activate nemu_env`

### Running the neuromuscular model notebook
- Use the command `jupyter notebook` to start the Jupyter notebook application.
- In Jupyter notebook, open the file 'neuromuscular_note.ipynb'.
- It is highly recommended to execute each cell in the order of appearance, since there are dependencies between the cells.

## Publications
- [Molinari, R.G., Elias, L.A.](Poster_SfN_2019_RGM_LAE.pdf) (2019) An interactive Python notebook as an educational tool for neuromuscular control. In: Proceedings of the 49th Annual Meeting of the Society for Neuroscience, Chicago.

## References
1. [Bigland-Ritche et al.](http:doi.org/10.1177/107385849800400413) (1998) Contractile properties of human motor units: Is man a cat? Neuroscientist, 4(4), 240–249.
2. [Cisi, R.R.L., Kohn, A.F](https://dx.doi.org/10.1007/s10827-008-0092-8) (2008) Simulation system of spinal cord motor nuclei and associated nerves and muscles, in a Web-based architecture. Journal of Computational Neuroscience, 25(3), 520-542.
3. [Challis, R.E., Kitney, R.I.](https://doi.org/10.1007/BF02442601) (1990) Biomedical signal processing (in four parts) - Part 1 Time-domain methods. Medical & Biological Engineering & Computning, 28(6), 509-524.
4. [Challis, R.E., Kitney, R.I.](https://doi.org/10.1007/BF02446704) (1991) Biomedical signal processing (in four parts) - Part Part 3 The power spectrum and coherence function. Medical & Biological Engineering & Computning, 29(3), 225-241.
5. [Enoka, R.M., Fuglevand, A.J.](https://doi.org/10.1002/1097-4598(200101)24:1<4::AID-MUS13>3.0.CO;2-F) (2001) Motor unit physiology: Some unresolved issues. Muscle and Nerve, 24(1), 4-17.
6. [Fuglevand, A.J., Winter, D.A., Patla, A.E.](https://doi.org/10.1152/jn.1993.70.6.2470) (1993) Models of recruitment and rate coding organization in motor-unit pools. Journal of Neurophysiology, 70(6), 2470–88.
7. [Johnson et al.](https://doi.org/10.1016/0022-510X(73)90023-3) (1973) Data on the distribution of fibre types in thirty-six human muscles. Journal of the Neurological Sciences, 18(1), 111-129.
8. [Kernell D.](https://doi.org/10.1093/acprof:oso/9780198526551.001.0001) (2006) The Motoneurone and its Muscle Fibres. Oxford University Press.
9. [Lo Conte et al.](http://doi.org/10.1109/10.335863) (1994) Hermite Expansions of Compact Support Waveforms: Applications to Myoelectric Signals. IEEE Transactions on Biomedical Engineering, 41(12), 1147-1159.
10. [Milner-brown, et al.](https://doi.org/10.1113/jphysiol.1973.sp010087) (1973). The contractile properties of human motor units during voluntary isometric contractions. Journal of Physiology, 228, 285–306.
11. [Shin, H., Suresh, N.L., Rymer, W.Z., Xiaogang, H.](https://doi.org/10.1088/1741-2552/aa925d) (2018) Relative contribution of different altered motor unit control to muscle weakness in stroke: a simulation study. Journal of Neural Engineering, 15(1).
12. [Watanabe et al.](https://doi.org/10.1152/jn.00073.2013) (2013) Influences of premotoneuronal command statistics on the scaling of motor output variability during isometric plantar flexion. Journal of Neurophysiology, 110(11), 2592–2606.
13. [Yao, W., Fuglevand, A.J., Enoka, R.M.](https://doi.org/10.1152/jn.2000.83.1.441) (2000) Motor-unit synchronization increases EMG amplitude and decreases force steadiness of simulated contractions. Journal of Neurophysiology, 83(1), 441-452.
14. [Zhou, P., Rymer, W.Z.]( https://doi.org/10.1152/jn.00367.2004) (2004) Factors governing the form of the relation between muscle force and the EMG: a simulation study. Journal of Neurophysiology, 92(5), 2878-2886.

## Funding
This project was funded by [CNPq](http://www.cnpq.br/) (Brazilian NSF): proc. #409302/2016-3.
