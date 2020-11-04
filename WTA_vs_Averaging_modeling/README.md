# Can we use modeling to identify the strategies used by zebrafish adopted in response to competing stimuli?

When confronted with a crowded visual scene, animals often choose a single object
for a behavioral response from multiple competing stimuli.

Elementary forms of spatial
attention exist in many species, including flies, fish and mice.


Winner-take-all (WTA) computations, in which an animal
responds to a single stimulus out of many, are considered to be crucial during bottom-up, stimulus-driven attention (Itti and Koch, 2001). In addition to WTA mechanisms, evidence suggests that presentation of multiple visual stimuli in primates can also lead to gaze shifts toward the mean location of the stimuli. 
See for example the following publication:
https://www.jneurosci.org/content/17/19/7490.


To estimate the relative contributions of each strategy, we fit the data with
a model that mixed predictions from both behavioral strategies.

All models are based on repeated random sampling, where one stimulus response from an S1 trial and one stimulus response from an S2 trial are combined. The repetition of this sampling procedure generates a distribution of combined responses. The averaging model combines the pair of responses by taking the vector average of the response angle. In agreement with the reduced amount of backward responses, we implemented a mechanism to reduce the prevalence of such
escapes in our model by redistributing backward swims to other headings. The winner-take-all (WTA) model chooses randomly between the S1 response and the S2 response (effectively adding the S1 and S2 response distribution). The mixture model implements a random assortment between the winner-take-all model with probability p) and the averaging model (with probability p-1). Distributions are plotted using a kernel density estimate (KDE) plot, with a von Mises (circularized) distribution. To compare the similarity of distributions, a circularized version of the energy distance metric was used.

# Answer: YES! Modeling reveals that fish larvae use a combination of WTA and averaging strategies.

For more details see the following publication: https://www.biorxiv.org/content/10.1101/598383v1

The modeling approach was performed with great help of my great friend Joe Donovan (https://github.com/joe311).
