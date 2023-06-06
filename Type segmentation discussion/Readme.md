# Discussion of Type Segmentation as a technique for face bias mitigation. 

## Motivation 
Most of the other face detection algorithm lead to reduction in model overall
performance. We want a algorithm that will not lead to overall performance reduction , while also 
addressing the fairness issue. 

## Principle 
The principle is that the data is not all same and the decision threshhold used by face detection systems for 
face detection is different for each class , mainly becuase the contribution of every particulalr segment to the 
whole dataset and therefore in the stored model is not the same. 

By dividing the data into clusters , we can assign different threshold to each cluster and achieve optimal 
performance. 

## Performance 

According to "Terhörst, P., & Kolf, J. N. (2020). Post-comparison mitigation of demographic bias in face recognition using fair score normalization. Pattern Recognition Letters, 140, 332-338. https://doi.org/10.1016/j.patrec.2020.02.006"
, the achieved performance is better than non-normalized version. They call this process  normalization. 

## Extra advantage 

This process can be implemented on top of most face detection algorithms. 


