# pythonalg
In this Github, there are two projects:
1. Influencer Categorization : NLP  in Python for influencers categorization on Instagram
2. Next Basket Prediction : Deep learning in Python for Recommender Systems

# Influencer Categorization
1. keyword extraction algorithm which includes: gensim_LDA, gensim_tf-idf, sklearn_tf-idf, Rake, Text_Rank 
2. keyword list is: Keyword = polling of keyword_extraction_algorithms + top n words +top m hashtags
3. Keyword list maps to a predefined category list by word2vec

# Next Basket Prediction
1. This problem modeled as sequence prediction problem. Therefore, the RNN deep learning network was deployed
2. To overcome sparsity, auto-encoder was used. The dataset and its feature vector was large 
3. The performance was measured by recall, precision and F1 score <br />
Please read the paper: Temporal Behavioral Modeling for Recommendation A Deep Learning Approach_v2.pdf 

