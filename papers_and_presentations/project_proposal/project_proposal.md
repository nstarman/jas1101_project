---
title: JAS 1101 Final Project Proposal
---

# Project Proposal

*Qing Liu, Vivian Ngo, Nathaniel Starkman*

Choose a project and develop a research question

A 1-page , single-spaced "proposal" describing the group's research project. See the Final Project Description document for more details (in Modules/ Final Project).

You may hand in a hard-copy at the beginning of class on Monday Jan 27, or send a PDF to gwen.eadie@utoronto.ca with the subject "JAS1101: project choice" before class on Monday Jan. 27

1. Choose a project and develop a research question (5%)
Each group must choose a project idea from the list provided and expand upon it, or come up with a project idea together. Once you’ve decided on a topic, together write a 1 page, single-spaced “proposal” describing:
what astronomical system, object, etc. you have chosen to work on
what research question you are asking about this system, object, etc
what data you will use to try to answer this question
what kind(s) of statistical analysis/analyses you will apply to these data to perform inference and/or answer your scientific question
any challenges you suspect might arise


Main Source:
https://arxiv.org/pdf/1807.09775.pdf
https://arxiv.org/abs/1304.7156

Supplementary:
https://arxiv.org/abs/1212.0859

Additional:
Survival analysis: https://astrostatistics.psu.edu/su10/lectures/upperlimits.pdf 

Population constraints on the existence of Intermediate Mass Black Holes in Milky Way Globular Clusters: specifically the M-sigma relation and King’s Profile model selection

Do populations of globular clusters prefer models with intermediate mass black holes? 

Identify stars that are cluster members (isochrone, space, kinematics)
Data truncation: 1.Malmquist (magnitude limit) 2. Unresolved center
Heteroscedatic data (further objects have more error)

Model fitting, Model selection (survival analysis?)
Hypothesis test - whether there is a blackhole 
IMBH = Intermediate Mass Black Holes 

All starts within some spatial region 
Combine data into one dataset after extracting and regularizing globular clusters
One dataset
For each globular cluster, have a list of the stars with their positions, velocities, and colors - UGRIZ system..? 
150 globular clusters 
Github: vivianngo97 ,   NGC4676


http://simbad.u-strasbg.fr/simbad/sim-ref?querymethod=bib&simbo=on&submit=submit+bibcode&bibcode=2019MNRAS.484.2832V 


To date, the favored method of constraining intermediate mass black holes (IMBHs) in globular clusters (GCs) is on a case-by-case basis, necessitating a  detailed and precise analysis of each GC. Instead, we will attempt to consolidate the full (known) GC catalogue and perform a population-level inference on the favorability of IMBH-inclusive GC models over IMBH-free models. The question we are asking is: for globular clusters as a population, whether the presence of IMBH in the center is statistically significant. And furthermore, from an integral perspective, whether a black hole mass - velocity dispersion ($M_{BH}-\sigma$) correlation holds for IMBH. The answer would shed light on the role that IMBHs plays in globular clusters, and help us better understand the co-evolution of black holes with astronomical systems on different scales.

We propose to limit our investigation to Milky Way globular clusters (GCs). Our data will be sourced primarily from the Gaia catalogue, which is publicly available. As needed, we will cross-match with precision photometry datasets, such as Pan-STARRS and 2Mass, to which we also have access. These datasets offer improvements upon the Gaia photometry and augment the Gaia kinematic information. Prior studies, such as Vasiliev 2018, have identified 150 GCs and provided details on how to reproduce a GC catalogue.

There are two primary lines of evidence we will pursue. The first is through the spatial and kinematic distribution functions (DFs). Kızıltan et. al. (2017) describes how an IMBH affects both the spatial distribution of stars and their kinematics in its host GC: such as quenching mass segregation and heating the cluster by increased scattering. These effects should extend beyond the black hole’s radius of direct influence and should be dynamical signatures detectable by analyzing the DF of the GC. The second independent line of evidence serves as a consistency check for the distribution function modeling -- the famous $M_{BH}-\sigma$ relation (though extended to GC mass scales). Observations using multiple probes have revealed that the existence of super massive black holes at galactic centers (e.g. Balick & Brown 1974) and a tight $M_{BH}-\sigma$ correlation (e.g. Kormendy & Ho 2013). The tightness of the correlation reveals a strong connection between the growth of the central black hole and the kinematics of the host galaxy, indicating the self-regulation of the system due to the presence of the black hole (e.g. Hopkins et al. 2007). It has been inferred from formation theory that globular clusters are likely to host IMBHs at their centers. Thus it is of great interest to test whether such correlation holds down to the low mass regime, for the reason that globular clusters have very different formation and evolution conditions with galaxies in their structures, compositions, environments, etc. By checking the mass inferred from the $M_{BH}-\sigma$ against the inference from the DF investigation, we hope to lend credence  to both analyses.

The statistical analyses that we plan to apply are model fitting, hypothesis testing, survival analysis, and simulations. One of our goals is to quantitatively describe properties of globular clusters by fitting models to determine differences due to the presence of IMBH. We also plan to conduct hypothesis tests to determine the significance of correlations. Moreover, sometimes only upper limits of $M_{BH}$ can be derived, as a case of censored data. To deal with this, we will leverage survival analysis in our research. Finally, we will also use Markov Chains and Monte Carlo techniques to perform simulations. 

Potential challenges we might meet in our analysis are as follows: 1) Missing data. First, in a portion of globular clusters, individual stars are unresolved at cluster centers due to the high concentration there. Second, the catalog has a faint end limit, which means stars fainter than that limit are not included in the sample (the Malmquist bias). 2) Data contamination. Foreground and background stars are mixed with cluster members, which are required to be cleaned using isochrone fitting. 3) Data heteroscedasticity. Further globular clusters have larger uncertainties in their measurements and each globular cluster has different mass, mass-to-light ratio, and metallicity. It is important to appropriately scale the measurements based on the properties of globular clusters before combining them into one sample. 4) As mentioned before, sometimes only upper limits of measurements are available. It is challenging to simultaneously include upper limits and uncertainties because common statistical analysis only deal with either of them. Therefore, further methods need to be explored to account for both.



Feedback from Gwen:

A bit ambitious 
Citations required for proposals  
Lower mass clusters - no IMBH? 
Upper limits are very uncertain
-> focus on one globular cluster first and then do it for the others