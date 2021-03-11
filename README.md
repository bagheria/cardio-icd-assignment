## Cardio ICD assignment
Version 0.7.0

## Automatic multilabel assignment of ICD10 codes in discharge letters from cardiology

In this study, we propose a pipeline for the automatic classification of ICD-10 codes from free-text clinical discharge letters from cardiology. We investigated the usage of solely the summary paragraph of discharge letters (conclusion), adding clinical variables (age/sex) and multilabel classification as is the case in clinical practice. We focussed well-defined and frequently used three-digit ICD-10 codes related to cardiology with sufficient granularity to be clinically relevant such as atrial fibrillation (I48) or acute myocardial infarction (I21).


#### Researchers:
  * Arjan Sammani
  * Ayoub Bagheri
  * Daniel Oberski
  * Folkert W. Asselbergs
  * Peter G.M. van der Heijden
  * Anneline SJM te Riele
  * Annette F Baas
  * CAJ Oosters


### ICD10 structure

![alt text](https://github.com/bagheria/cardio-icd-assignment/blob/master/Documents/icd_structure.PNG)

### Pipeline
We presented a deep learning pipeline for automatic multilabel ICD-10 classification in free medical text: Dutch discharge letters from cardiology. Given the sensitive nature of these data, we added an anonymization step.

![alt text](https://github.com/bagheria/cardio-icd-assignment/blob/master/Documents/pipeline.png)


## Papers ##
1- Bagheri, A., Sammani, A., Van der Heijden, P.G.M., Asselbergs, F.W., Oberski, D.L. (2020). Automatic
ICD-10 classification of diseases from Dutch discharge letters, In Proceeding of the 13th International
Joint Conference on Biomedical Engineering Systems and Technologies, Malta, 281-289. [Link](https://discovery.ucl.ac.uk/id/eprint/10098370/1/C2C_2020_2.pdf)

2- Sammani, A., Bagheri, A., Van der Heijden, P.G.M., Te Riele, A.S.J.M., Baas, A.F., Oosters, C.A.J., Oberski, D.L., Asselbergs, F.W. (2021). Automatic multilabel detection of ICD10 codes in cardiology discharge letters using neural networks, npj Digit. Med. 4, 37 (2021). https://doi.org/10.1038/s41746-021-00404-9.

## License ##
This project is licensed under the terms of the [MIT License](LICENSE)

## Citation ##
Please cite the aforementioned papers. 
