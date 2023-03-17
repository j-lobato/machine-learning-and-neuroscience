# Is Machine Learning being misused in Neuroscience?


João Avelar Lobato
Research Director at Cooper and Sacks
 
A number of articles report over 70% accuracy in the identification of psychopathologies by applying machine learning on different features, such as heart rate variation (Byun et al., 2019), electrodermal activity (Kim et al., 2018), brain imaging (Gao et al., 2018, from a review of MRI), brain connectivity imaging (Liu et al., 2022), speech analysis (Lin et al., 2022), and traditional surveys (Haque et al., 2021). 


While not criticizing any specific study, we could ask how valid is a 70% or even 80%+ accuracy rate in a small sample size, which is common in neuroscience. A recent study showed, for example, that the 1038 most highly cited neuroimaging studies in the past three decades had a sample size of only 12. While studies in 2017 and 2018 had on average a higher sample size, it was between 23 and 24 (Szucs and Ioannidis, 2020). In a review of studies that used machine learning to predict autistic and non-autistic individuals, Vabalas et al. (2019) indicated that about half of the studies had a sample size below 80. The authors also said that small sample sizes are associated with higher accuracy.


To explore this question, I trained seven algorithms with no tuning on a small dataset that can be found at https://osf.io/yhk2p/. The dataset consists of data from 26 individuals, one of which was excluded for not having a label, of parents of children with autism who had completed a web training. The dataset has four features: household income, most advanced degree of the parent, the child’s social functioning, and the baseline scores of parental use of behavioral interventions at home. The label is a binary field, with 0 indicating no improvement of the children’s behavior and 1 improvement. The dataset was designed to assess "the effects of an interactive web training to teach parents behavior analytic procedures to reduce challenging behaviors in children with autism spectrum disorder" (Turgeon & Lanovaz, 2020) 


The dataset is highly skewed, with 17 (or 68%) of the 25 labels belonging to one class 1, which means that an algorithm that simply outputs 1 would get a 68% accuracy rate, as mentioned by the authors of the article. In the article (preprint version) the accuracy rate of a random forest model was 77%. 


I run seven models (logistic regression, linear discriminant analysis, K neighbors classifier, gaussian naive bayes, SVC, random forest, and XGBoost) in the dataset. Only one of them did not have an accuracy above 70% (linear discriminant analysis). XGBoost had an accuracy of 88%. 


Does that mean the features of the dataset are good indicators of which families will benefit from the autism behavior training? After all, how relevant is an accuracy rate in such a small, skewed dataset? Can the prediction really tell us who is more likely to benefit from the autism training?


To answer this, I ran the same algorithms above but this time using a random set of labels in the test, where about half of them belonged to one 0. I did it 39 times, creating 39 random label sets. In 26 (67%) of the cases at least one algorithm had an accuracy of more than 70%. 


Perhaps that shouldn't be a surprise as some of the random test labels could happen to correlate with one of the features given the small sample size. It was, nonetheless, concerning that in so many cases at least one algorithm had an accuracy above 70% when verified against random test labels.  


I then used random labels for both training and testing. That is, the algorithms were trained on purely random data. I created 40 random labels. In 18 (45%) of them at least one algorithm had an accuracy of more than 70%. When the training and test labels were stratified, in 23 (58%) of times at least one algorithm had an accuracy over 70%. There was no parameter tuning at any point, which could have further increased the performance of the algorithms. 


It was probably again the small sample size of the datasets, in which random variables could just happen to match the random labels. The key point, however, is that machine learning algorithms are so powerful that they can make sense of spurious datasets. This is not a new assertion, but may lead to misleading research findings. Studies might treat features as reliable predictors of the outcome of an intervention or to classify pathologies when in reality they are not. 
 
In a further step, I created 40 random datasets with six random features varying from 0 to 9 and binary labels [0, 0], with a sample size of 30. Despite being fed with purely random data, at least one model had an accuracy of 70% or more in 26 (65%) of the cases. When the labels were not stratified it was 17 (43%). That means you could create a dataset with ridiculous features such as the number of letters in people’s names, length of their forearm, or ask them to choose a number between 0 and 9, and use those features to predict with high accuracy if they are depressed or not, for example. You could then correctly (but misleadingly) say that an algorithm had 70% or even 80% of accuracy and that some spurious features were correlated with depression. 


Even when with a sample size of 80, the models had an accuracy above 70% in 10% to 15% of the cases. By creating uneven labels, with 60% of them representing one class, which is common in real experiments, and decreasing the accuracy threshold to 65%, the algorithms made sense of 17 (43%) of the datasets. Increasing the number of features to 64 and changing the range of some of them did not significantly change the proportion of datasets with accuracy above 70%. 


The assertion that a high accuracy in a small sample size is not necessarily meaningful is not new. However, it has been ignored in a large number of studies. It is concerning that meaningless features might influence the choice of interventions and clinical practices. The simple experiment above showed that high accuracy rates can be found even in the most extreme cases of algorithms being trained on random labels or datasets. Obviously this is linked to the stochastic nature of the data generated but this does not seem to be taken into account in many studies. 


Aleem et al. (2022) discussed a study that had 97.54% accuracy in the detection of people with depression using a sample size of only 66 (58% of whom were depressed, which means the labels were not balanced). Any practitioner would be concerned with such a high accuracy in a small sample size, particularly because 58% of the labels represented one group. 


While machine learning is a tool that can (and is already) supporting neuroscience research, protocols need to be developed to better understand its potentials and limitations. 



* Due to the stochastic nature of the models and the fact that the code will generate new slightly different results 


References

Aleem, S., Huda, N. U., Amin, R., Khalid, S., Alshamrani, S. S., & Alshehri, A. M. (2022). Machine Learning Algorithms for Depression: Diagnosis, Insights, and Research Directions. Electronics, 11(7), 1111. https://doi.org/10.3390/electronics11071111

Byun S, Kim AY, Jang EH, Kim S, Choi KW, Yu HY, Jeon HJ (2019). Detection of major depressive disorder from linear and nonlinear heart rate variability features during mental task protocol. Computers in biology and medicine. Retrieved March 17, 2023, from https://pubmed.ncbi.nlm.nih.gov/31404718/ 

Gao, S., Calhoun, V. D., &amp; Sui, J. (2018, November). Machine learning in major depression: From classification to treatment outcome prediction. CNS neuroscience &amp; therapeutics. Retrieved March 17, 2023, from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6324186/ 

Haque, U. M., Kabir, E., & Khanam, R. (2021). Detection of child depression using machine learning methods. PLOS ONE, 16(12), e0261131. https://doi.org/10.1371/journal.pone.0261131

Lin, Y., Liyanage, B. N., Sun, Y., Lu, T., Zhu, Z., Liao, Y., Wang, Q., Shi, C., & Yue, W. (2022). A deep learning-based model for detecting depression in senior population. Frontiers in Psychiatry, 13. https://doi.org/10.3389/fpsyt.2022.1016676

Liu, Z., Wong, N. M., Shao, R., Lee, S., Huang, C., Liu, H. A., Lin, C. P., & Lee, T. M. (2022). Classification of Major Depressive Disorder using Machine Learning on brain structure and functional connectivity. Journal of Affective Disorders Reports, 10, 100428. https://doi.org/10.1016/j.jadr.2022.100428

Kim, A. Y., Jang, E. H., Kim, S., Choi, K. W., Jeon, H. J., Yu, H. Y., &amp; Byun, S. (2018, November 19). Automatic detection of major depressive disorder using electrodermal activity. Nature News. Retrieved March 17, 2023, from https://www.nature.com/articles/s41598-018-35147-3 
Szucs, D.; Ioannidis, J. (2020, July 15). Sample size evolution in neuroimaging research: An evaluation of highly-cited studies (1990–2012) and of latest practices (2017–2018) in high-impact journals. NeuroImage. Retrieved March 13, 2023, from https://www.sciencedirect.com/science/article/pii/S1053811920306509 
Turgeon, S., & Lanovaz, M. J. (2020). Tutorial: Applying Machine Learning in Behavioral Research. Perspectives on Behavior Science, 43(4), 697–723. https://doi.org/10.1007/s40614-020-00270-y





