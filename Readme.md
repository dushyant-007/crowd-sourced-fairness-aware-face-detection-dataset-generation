# Crowd-Sourced Fairness-aware Face detection dataset generation

## Project Objective

The objective of this project is to create a face detection dataset that is fair and unbiased in terms of age, gender, and race. The aim is to develop a face detection model that can accurately detect and differentiate between faces from different demographic groups with equal accuracy.

### Definition of Fairness

By fairness, we mean that the face detection model should exhibit consistent accuracy across various demographic groups. In other words, the accuracy for detecting faces of individuals with black skin should be comparable to the accuracy for individuals with brown skin, Asian skin, or white skin.

## Inspiration and Background

This project is motivated by discussions surrounding the responsible use of AI. Experts in the field of Artificial Intelligence and Machine Learning acknowledge that the fairness of machine learning models depends on the quality and bias within the training data. The data collection and labeling process play a crucial role in determining the biases present in the resulting algorithms.

Several approaches have been proposed to address the fairness problem, including the use of generative examples to mitigate race-related features, augmenting the dataset with underrepresented group images, and modifying the loss function to consider accuracies across different demographic groups. However, these methods only partially solve the issue, as the underlying training data remains inherently biased.

## Proposed Solution: Creating an Equitable Dataset

To achieve true fairness and overcome biased training data, the project aims to create a dataset that accurately represents all sections of society in equal proportions. This solution requires a comprehensive collection of data that is representative of diverse age groups, genders, and races.

### Leveraging Crowdsourcing Platforms

To accomplish this task, we are exploring the use of crowdsourcing platforms as a means of collecting highly accurate data. Specifically, we plan to utilize Amazon Mechanical Turk to generate accurate labels for our dataset. By leveraging the power of crowdsourcing, we can involve a larger pool of contributors and obtain a more comprehensive and diverse dataset.

## Future Considerations

Through this project, we will evaluate the effectiveness of crowdsourcing platforms, such as Amazon Mechanical Turk, in gathering data with a high level of accuracy. Additionally, we will analyze the implications of utilizing such platforms for building fair and responsible AI systems. This research will contribute to advancing the state of responsible AI and shed light on how progress can be made using crowd-sourcing platforms.

## File Structures and details 

1. Readme.md 
   - details the project, it's motivation, methods and results, details the file descriptions and the usecases of the files. 
2. CompoFair-main
    - This file contains all the work that Akash has done on the project. This is his codebase. The information about indivisual components will be available inside.
    - The high level overview is that there are datasets that were generatated from amazon mechanical turk , trained models and there training and testing scripts, data from the celebA dataset, a webpage that was designed to collect data from the AMT, files containing the identity of each image from the celebA dataset. 
3. Dushyant't investigations 
   - This file contains all the work that Dushyant has done on the datasets and scripts , this is final deliverable of the independent study. 
4. Literature
   - This comprises of a list of the papers that I have read and studied for this work, it will be useful during final documentation. 
5. Type Segmentation discussion 
   - This is a file that will mostly probably be deleted. 
   - It is a discussion regarding a specific paper that I believe will play an important role for the classes.
6. .gitignore 
   - a file that is used to keep track of files that are not tracked. lol 
7. .gitattributes 
   - Not important, shows some repo parameters. 
8. File transfer code.txt
   - This looks like a script to transfer the data from one place to another. 
     - Which data ? - it looks like it the data related to images in celebA dataset, the csv file contains the image names and all those particular images are transferred to the new location called destination. 
     - what is the source ? - name of the folder is 'img_align_celeba', we will see later what modification has akash done to these images and whether they are even different from celebA dataset. 
     - Destination ? - it is /project_folder/crowdsourcing_batch.
9. IS589 Presentation.pptx - Presentation by Akash
10. IS589 independent Study.pdf - final report by Akash. 
11. MTurk Specifications
    - It contains details about the MTurk Configuration that Akash used for data collection . 
    - IT contains info - rewards per assignment, number of assignment per task, time alloted per assignment, task expired in , auto-approve and pay workers in __ days, criterion for worker selection - HIT approval rate, minimus number of HITs , whether a worker is MTurk masters or not. 
12. project discussion points - 
    - to be deleted later, details the discussion points between Dushyant and Yang. 
13. Project_proposal_IS589 - Dushyant's draft of independent study proposal. 
14. Proposal draft1.pdf - dushyant's proposal draft. 
15. ptype.html
    - This is the html file related to framework2 (refer the report) of data collection. 
    - Contains three different groups of images and how we can see if we can gather systematic differences between race, gender and age.
    - Segmentation is done on the basis of race and skin color. 
16. ptype_old_young.html
    - same thing as ptype.html but the segmentation is done on the basis of age. 

