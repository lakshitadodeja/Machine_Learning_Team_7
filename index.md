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

This will be employed to identify patterns in the data and attribute labels a posteriori, to account for datasets without labels to guide the training process.
1.	We will perform K-means clustering<sup>[9]</sup> on vector space based on
    1. One-Hot encodings 
    2. 	BERT encodings. 
2.	We also compare the effect of dimensionality reduction techniques before clustering through k-Means algorithm to understand the effect of word embeddings on the essay rating predictions, using :
    1. PCA<sup>[10]</sup>
    2. T-SNE<sup>[11]</sup>


### Potential Results and Discussion
All approaches will be objectively compared through metrics such as accuracy, F1 score, precision, and recall. 
We will also qualitatively compare approaches using transcripts of court proceeding (Trial of Johnny Depp vs Amber Heard). The output of each argument will described  "effective," "adequate," or "ineffective” classification goal of our project. 

### Supervised learning metrics discussion 
As discussed above, Single task learning (STL) and multitask learning (MTL) were implemented for supervised learning. 
In the multi-class classification for the three class arguments – Effective, adequate and ineffective, we compared precision, recall and f1 scores for STL and MTL models over balanced and imbalanced data sets. 

As expected, overall performance improvement <sup>[12]</sup> is achieved in the MTL model compared to the STL model over both balanced and imbalanced data set. 

For the balanced data set, the performance scores were mostly consistent for both multi-task and single task. Noticeable performance variations were in the adequate and ineffective labels where the MTL performed better for recall and F1 scores except the precision scores of the adequate label where the STL showed better performance. 

Interestingly for the imbalanced data set, the recall and f1-score improved with STL compared to the MTL, but performed lower for precision. 
Summary of performance is shown in the table below;


| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |


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



