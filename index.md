## Predicting Effective Arguments


### Introduction
Academic writing is a crucial part of life. Therefore, the need to ensure students develop writing confidence and proficiency is pivotal.  Findings show that human tutoring is effective at improving students’ writing performance, but it is time consuming and labor intensive [1] 
Over the years, Automated essay scoring (AES), and feedback has been gaining attention due to technological advances in educational assessment. 
Several AES systems exist such as the Accessor, e-rater and Project Essay Grade [2], which used simple linear regression techniques and text classification on hand-crafted features such as proposition counts and length of essays. Other automated writing feedback which exists are limited by costs and they often fail to evaluate the quality of argumentative elements, such as organization, evidence, and idea development. Deep learning models, feature engineering and word embeddings [2] are some of the techniques being explored today. 

Our approach is to train a model to classify argumentative elements in student writing as "effective," "adequate," or "ineffective”. The dataset contains argumentative essays written by U.S students in grades 6-12 (About 36,000 essays) and contains features corresponding to the seven discourse elements -lead, position, claim, counterclaim, rebuttal, evidence and concluding statement.


### [Dataset](https://www.kaggle.com/competitions/feedback-prize-effectiveness)


### Problem Statement
Due to resource constraints and limitations especially in underrepresented communities, teachers’ ability to issue writing tasks and feedback to students are limited. Our goal is to train a model to classify argumentative elements in student writing as "effective," "adequate," or "ineffective”. This will enable students get automated guidance and feedback on writing tasks and help them improve their writing skills.   

### Methods

**Supervised Learning Methods**  
1. BERT embeddings, adding 2-3 trainable hidden layers to BERT to obtain accurate classification of the essays through text classification and rating the writing quality of each essay.
2. 	T5  (To be completed)
3. 	GRU  (To be completed)
4. 	If necessary, we may augment the training data through paraphrasing software such as Pegasus. 

**Unsupervised Learning Methods**.  
This will be employed to identify patterns in the data and attribute labels a posteriori, to account for datasets without labels to guide the training process. We will perform :  

1.	K-means on vector space based term frequency of one-hot encodings as well as BERT encodings. 
2.	We also compare the effect of dimensionality reduction techniques such as PCA and t-SME before clustering through k-Means algorithm to understand the utility of dimensionality reduction and the effect of word embeddings on the essay rating predictions. 


### Potential Results and Discussion
All these approaches will be objectively compared through metrics such as accuracy, F1 score, precision, and recall. 
We will also qualitatively compare approaches using transcripts of court proceeding (Trial of Johnny Depp vs Amber Heard). The output of each argument will be classified using a positive, neutral or negative emoji, in line with the "effective," "adequate," or "ineffective” classification goal of our project. 

### References

1. C. Lu and M. Cutumisu, “Integrating deep learning into an automated feedback generation system for automated essay scoring,” Eric.ed.gov. [Online]. Available: https://files.eric.ed.gov/fulltext/ED615567.pdf. [Accessed: 11-Jun-2022].
2. E. B. Page, “Computer grading of student prose, using modern concepts and software,” The Journal of Experimental Education, vol. 62, no. 2, pp. 127–142, 1994.
3. K. Taghipour and H. T. Ng, “A neural approach to automated essay scoring,” in Proceedings of the 2016 Conference on EMNLP, 2016, pp. 1882–1891.
4. X. Liu, P. He, W. Chen, and J. Gao, “Multi-task deep neural networks for natural language understanding,” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019, pp. 4487–4496.


