# Testing the various Face detection algorithms for fairness with respect to various face features. 

## Face features - Race, age , gender, depending upon eyes size, color, eyebrows colour, the size of ears , cheeks, lips, etc. 

## Motivation 

We want the models to be equally responsive to people of all the demographics , gender , race and ethnicity and etc.etc. 

But It is almost impossible to do so simply by training a data on large datasets, becuase of different levels of availability of data wrt different features. 

For now , we try our best to reduce bias by reducing it for Race, age, and gender. 

How can we do that ? - create a dataset which have the following labels on each face, characteristics - of eyebrow, cheeks, lips, eyes, ears, forehead, nose, tongue, chin and mouth etc. 

## Objective 

We want to demonstrate that creation of this data isd healful in countering bias in the models
we are going to do it by passing different types of data into the face detection systems and seeing if the 
face detection model has equal accuracy for different demographics.

## Dataset

We have labelled celebrity dataset with all the required features.

Pros - 

1. Simple data, High resolution and professionally shot for clarity. 
2. Easy to label by people of different backgrounds as lesser nuance is required.

Cons or Possible challenges.  -

1. Face detection systems need to work in real world application and we would want to make sure that 
faces do get recognised in real world. The images in the real world aren't as organized and clear and professionally shot as the compared to the images in the celebrity dataset. 
2. The celebrity dataset, doesn't have much occlusions. 
3. The celeb dataset doesn't have many orientations. 
4. The celeb dataset doesn't have backgrouond detailing, which doesn't allow any candidate to be more professional. 

