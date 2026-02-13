## Table of Contents
- [1. What is the LDA Model](#1-what-is-the-lda-model)
- [2. How to Choose the Number of Topics](#2-how-to-choose-the-number-of-topics)
- [3. Using Topic Models for Cold Start in Recommendation](#3-using-topic-models-for-cold-start)
- [4. References](#4-references)
- [5. Code Implementation](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/5.3%20Topic%20Model/HillaryEmail.ipynb)

## 1. What is the LDA Model

**LDA** (Latent Dirichlet Allocation) is a topic model proposed by Blei, Ng, Jordan (2003). It represents each document as a probability distribution over topics and each topic as a distribution over words. Documents are modeled as bags of words; a document can have multiple topics.

**Document generation:** Sample topic distribution θ from Dirichlet(α); for each word, sample topic z from θ; sample word from topic z's word distribution φ. LDA adds Dirichlet priors (α, β) to pLSA, making it Bayesian.

**Key concepts:** Gamma function; Binomial, Multinomial, Beta, Dirichlet distributions; conjugate priors; pLSA; Gibbs sampling.

## 2. How to Choose the Number of Topics

1. Empirical tuning (most common)
2. Perplexity comparison
3. Log marginal likelihood
4. Nonparametric: HDP (Dirichlet process)
5. Topic similarity: cosine, KL divergence

## 3. Using Topic Models for Cold Start in Recommendation

Cold start: recommend with little or no user/item data. Types: user, item, system.

**User cold start:** Use registration info, search terms, or external data to infer interest topics. Find users with similar topics; use their history.

**Item cold start:** Use item metadata (e.g., director, actors, keywords) to infer topics. Recommend to users who like similar topics.

**Topic model:** Treat each user/item as a document; features as words. Learn topic distributions; use for matching and preference estimation.

## 4. References

[Understanding LDA Topic Models](https://blog.csdn.net/v_july_v/article/details/41209515)

## 5. Code Implementation

[LDA: Analyzing Hillary's Emails](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/5.3%20Topic%20Model/HillaryEmail.ipynb)

------

> Author: [@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> Welcome to join the discussion! <a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP Interview Study Group" title="NLP Interview Study Group"></a>
