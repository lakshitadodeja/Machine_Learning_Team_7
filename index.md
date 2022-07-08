## Predicting Effective Arguments


### Introduction
Academic writing is a crucial part of life and the need to ensure students develop writing confidence and proficiency is pivotal.  Findings show that human tutoring is effective at improving students’ writing performance, but it is time consuming and labor intensive<sup>[1]</sup> . Over the years, Automated essay scoring (AES), and feedback has been gaining attention due to technological advances in educational assessment. 
AES systems such as Accessor, e-rater and Project Essay Grade<sup>[2]</sup>, use linear regression and hand-crafted features such as proposition counts and length of essays. Other sophisticated AES are limited by cost, and they often fail to evaluate the quality of argumentative elements, such as organization, evidence, and idea development. Deep Learning-based models and word embeddings<sup>[3]</sup> are currently being explored to address these limitations.

### Problem Statement
Due to resource constraints and limitations especially in underrepresented communities, teachers’ ability to issue writing tasks and feedback to students are limited. We will train a model to classify argumentative elements in student writing as "effective," "adequate," or "ineffective”. Our [dataset](https://www.kaggle.com/competitions/feedback-prize-effectiveness)<sup>[4]</sup> contains about 36k argumentative essays written by U.S students in grades 6-12 and contains features corresponding to the seven discourse elements - lead, position, claim, counterclaim, rebuttal, evidence and concluding statement. This will enable students get automated guidance and feedback on writing tasks and help them improve their writing skills.  

### Methods
In this project, we attempt to predict these ratings through Supervised and Unsupervised learning methods. As this classification problem is based on textual inputs, we will use Natural Language Processing techniques to approach this problem.

**Supervised Learning Methods**  


1.	BERT<sup>[5]</sup> : We will use pretrained BERT embeddings with trainable hidden layers to obtain accurate classification of each essays 
2.	T5<sup>[6]</sup> : T5 is a 220 million parameter model pre-trained on a multi-task mixture of unsupervised and supervised tasks. We will finetune this model for our classification task. 
3.	Bidirectional GRU<sup>[7]</sup> :  We will use bidirectional GRUs to model the contexts from both directions, enabling the model to make a more informed decision for our task 
4.	We will also be using the Pegasus paraphrasing software to augment our training data.<sup>[8]</sup>

For BERT and T5, we will perform both multitask and single-task learning.

**Unsupervised Learning Methods**

This will be employed to identify patterns in the data and attribute labels a posteriori, to account for datasets without labels to guide the training process. This will be accomplished through One-hot encoding and BERT encodings.

* One-Hot encodings: For this analysis, we use Tf-idf (Term frequency - inverse document frequency) representation for entry essay snippet. Considering each word in the dataset to be a dimension results in around 29k features for each data point. However, this large dimensional space includes due to mispellings and various forms of the same word, thus leading to redundant features. To reduce these reduntant features, the vocabulary (consequently the dimesions of the dataset) is limited only the words that are present in at least two different essays. Further, these words are lemmatized to their root word. Through these steps, the vector-space of the data is limited to 7795 dimensions. This vector-space is further reduced through dimensionality through PCA<sup>[10]</sup> and t-SNE<sup>[11]</sup> to select the dominant features of the dataset. Clustering in then performed in this reduced dimensional space using k-Means clustering <sup>[9]</sup> and Gaussian Mixture models (GMM).

* BERT encodings: Despite many dimensinality reductions and feature enginnering, one-hot encoding treats each word indivudually and cannot capture the semantic representation of the text. Therefore, BERT encoding are used to address this deficiency of one-hot encodings. Similar to the analysis using one-hot vectors, we extract dominant features using PCA and t-SNE, and reduce the dimensionility of the data. Then, clustering is performed using k-Means and GMM.

<!---
We also compare the effect of dimensionality reduction techniques before clustering through k-Means algorithm to understand the effect of word embeddings on the essay rating predictions, using :
    1. PCA<sup>[10]</sup>
    2. T-SNE<sup>[11]</sup>
--->

### Data Collection 
Our [dataset](https://www.kaggle.com/competitions/feedback-prize-effectiveness)<sup>[4]</sup> contains about 36k argumentative essays written by U.S students in grades 6-12 and contains features corresponding to the seven discourse elements - lead, position, claim, counterclaim, rebuttal, evidence and concluding statement. In the following sections, we will explore these datasets in detail, and develop a model to classify arguments as effective, inadequate and ineffective. 











### Potential Results and Discussion
All approaches will be objectively compared through metrics such as accuracy, F1 score, precision, and recall. 
We will also qualitatively compare approaches using transcripts of court proceeding (Trial of Johnny Depp vs Amber Heard). The output of each argument will described  "effective," "adequate," or "ineffective” classification goal of our project. 

**1. Supervised learning metrics discussion**

As discussed above, Single task learning (STL) and multitask learning (MTL) were implemented for supervised learning. 
In the multi-class classification for the three class arguments – Effective, adequate and ineffective, we compared precision, recall and f1 scores for STL and MTL models over balanced and imbalanced data sets. 

As expected, overall performance improvement <sup>[12]</sup> is achieved in the MTL model compared to the STL model over both balanced and imbalanced data set (Summary is shown in table 1 below). 

|Learning model | Balanced  | Imbalanced| 
| ------------- | :-------: | :-------: | 
| MTL           | 0.62      | 0.68      | 
| STL           | 0.58      | 0.65      | 

**Table 1: Summary of accuracy results for STL and MTL over balanced and unbalanced data sets.**

For the balanced data set, the performance scores were mostly consistent for both multi-task and single task. Noticeable performance variations were in the adequate and ineffective labels where the MTL performed better for recall and F1 scores except the precision scores of the adequate label where the STL showed better performance. 

Interestingly for the imbalanced data set, the recall and f1-score improved with STL compared to the MTL, but performed lower for precision. (Report summary is shown in table 2(a-d)).


**MTL Imbalanced data set**
              
|Labels         | Precision | Recall   | F1-score  | 
| ------------- | :-------: | :-------:| :-------: | 
| Effective     | 0.71      | 0.08     | 0.15      | 
| Adequate      | 0.67      | 0.90     | 0.77      | 
| Ineffective   | 0.72      | 0.58     | 0.64      |   
| Macro avg.  | 0.70      | 0.52     | 0.52      | 
| Weighted  avg.| 0.69      | 0.68     | 0.63      | 

**MTL Balanced dataset**
|Labels         | Precision | Recall    | F1-score |  
| ------------- | :-------: | :-------: | :-------:| 
| Effective     | 0.39      | 0.50      |0.44      | 
| Adequate      | 0.68      | 0.68      |0.68      | 
| Ineffective   | 0.70      | 0.57      |0.63      | 
| Macro avg.    | 0.59      | 0.58      |0.58      |
| Weighted  avg.| 0.64      | 0.62      |0.62      | 


**STL Imbalanced data set**
              
|Labels         | Precision | Recall   | F1-score  | 
| ------------- | :-------: | :-------:| :-------: | 
| Effective     | 0.46      | 0.36     | 0.33      | 
| Adequate      | 0.66      | 0.83     | 0.74      | 
| Ineffective   | 0.72      | 0.53     | 0.61      | 
| Macro avg.    | 0.61      | 0.54     | 0.56      | 
| Weighted  avg.| 0.64      | 0.65     | 0.63      | 

**STL balanced data set**
              
|Labels         | Precision | Recall   | F1-score  | 
| ------------- | :-------: | :-------:| :-------: | 
| Effective     | 0.38      | 0.52     | 0.44      | 
| Adequate      | 0.73      | 0.50     | 0.60      | 
| Ineffective   | 0.55      | 0.79     | 0.65      | 
| Macro avg.  | 0.55      | 0.61     | 0.56      | 
| Weighted  avg.| 0.62      | 0.58     | 0.58      | 

**Table 2(a-d): Summary of precision, recall and F1-scores for STL and MTL over balanced and unbalanced data sets.**

**Note:**
1. Precision: TP/(TP+FP) -  indicates  what fraction of predictions as a positive class were actually positive.	
2. Recall: TP/(TP+FN) - indicates the fraction of all positive samples were correctly predicted as positive by the classifier.
3. f1_score: 2*(Precision.Recall)/(Precision + Recall) -is the harmonic mean of Precision and Recall. Research has shown that the F1 score is a better performance metrics than accuracy score for highly imbalanced datasets. 


### Gantt Chart 

The Gantt chart for our project can be found [here](https://gtvault-my.sharepoint.com/:x:/g/personal/sjain443_gatech_edu/EVY1kVoq6ixHlA6FHNfmD4wBotXA6n20QsYxsModKdRhPA?e=2YC7zb&isSPOFile=1)

### References

[1]	C. Lu and M. Cutumisu, “Integrating deep learning into an automated feedback generation system for automated essay scoring,” Eric.ed.gov. [Online]. Available: https://files.eric.ed.gov/fulltext/ED615567.pdf. [Accessed: 11-Jun-2022].\
[2]	E. B. Page, “Computer grading of student prose, using modern concepts and software,” The Journal of Experimental Education, vol. 62, no. 2, pp. 127–142, 1994.\
[3]	K. Taghipour and H. T. Ng, “A neural approach to automated essay scoring,” in Proceedings of the 2016 Conference on EMNLP, 2016, pp. 1882–1891.\
[4]	Feedback Prize - Predicting Effective Arguments, link : https://www.kaggle.com/competitions/feedback-prize-effectiveness\
[5]	Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).\
[6]	Raffel, Colin, et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." arXiv preprint arXiv:1910.10683 (2019).\
[7]	Cho, Kyunghyun, et al. "On the properties of neural machine translation: Encoder-decoder approaches." arXiv preprint arXiv:1409.1259 (2014).\
[8]	Zhang, Jingqing, et al. "Pegasus: Pre-training with extracted gap-sentences for abstractive summarization." International Conference on Machine Learning. PMLR, 2020.\
[9]	Forgy, Edward W. "Cluster analysis of multivariate data: efficiency versus interpretability of classifications." biometrics 21 (1965): 768-769.\
[10]	Pearson, Karl. "LIII. On lines and planes of closest fit to systems of points in space." The London, Edinburgh, and Dublin philosophical magazine and journal of science 2.11 (1901): 559-572.\
[11]	Van Der Maaten, Laurens. "Accelerating t-SNE using tree-based algorithms." The journal of machine learning research 15.1 (2014): 3221-3245.\
[12] Y. Zhang and Q. Yang, “A Survey on Multi-Task Learning,” arXiv [cs.LG], 2017. \



