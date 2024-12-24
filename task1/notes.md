## Analysis to be performed 

- Normal masked language modelling through BERT and other models (this is the base case)

-  Masked language modelling with contextual information additional contextual information. This is basically through narrative contexts whcih can be either positive, negative or neutral 

- Masked language modelling with identity terms along multiple axis - does one axis dominate in those cases? 

- Masked language modelling with small adversarial examples to detect stereotypes (using words like some, few, a lot, etc)

- Bias reversal via reverse social rules (sentiment prediction in sentences which have reverse of stereotypes listed)

- See if the model disproportionately attends to biased keywords, use methods like LIME or SHAP 

- Giving the model a biased sentence or a not biased sentence and tell it to predict if the sentence is biased or not? Sort of self evaluation of bias?

## Things to do if I have time

- Analysis with Caste and Gender as Identity terms

- Currently, I've worked with religion and region as they seemed the most relevant to me. But I can also work with other identity terms.

- Analysis of religion based bias in models trained in India and outside India