# Toward High Quality Facial Representation Learning (ACM MM 2023)


## Abstract
> Face analysis tasks have a wide range of applications, but the universal facial representation has only been explored in a few works. In this paper, we explore high-performance pre-training methods to boost the face analysis tasks such as face alignment and face parsing. We propose a self-supervised pre-training framework, called **Mask Contrastive Face (MCF)**, with mask image modeling and a contrastive strategy specially adjusted for face domain tasks. To improve the facial representation quality, we use feature map of a pre-trained visual backbone as a supervision item and use a partially pre-trained decoder for mask image modeling. To handle the face identity during the pre-training stage, we further use random masks to build contrastive learning pairs. We conduct the pre-training on the LAION-FACE-cropped dataset, a variants of LAION-FACE 20M, which contains more than 20 million face images from Internet websites. For efficiency pre-training, we explore our framework pre-training performance on a small part of LAION-FACE-cropped and verify the superiority with different pre-training settings. Our model pre-trained with the full pre-training dataset outperforms the state-of-the-art methods on multiple downstream tasks. Our model achieves 0.932 NME$_{diag}$ for AFLW-19 face alignment and 93.96 F1 score for LaPa face parsing.


### [Paper](https://arxiv.org/abs/2309.03504)
