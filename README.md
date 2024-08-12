# Confounders in Instance Variation for the Analysis of Data Contamination

This is the repository for our paper *"Confounders in Instance Variation for the Analysis of Data Contamination"* (authored by Behzad Mehrbakhsh, Darío Garigliotti, Fernando Martínez-Plumed, and Jose Hernández-Orallo), accepted in the [1st Workshop on Data Contamination (CONDA)](https://conda-workshop.github.io/) co-located at the [ACL 2024](https://2024.aclweb.org/) conference. You can access to the paper in PDF format [here](https://aclanthology.org/2024.conda-1.2/).


## How to cite it?
```
@inproceedings{mehrbakhsh-etal-2024-confounders,
    title = "Confounders in Instance Variation for the Analysis of Data Contamination",
    author = "Mehrbakhsh, Behzad  and
      Garigliotti, Dario  and
      Mart{\'\i}nez-Plumed, Fernando  and
      Hernandez-Orallo, Jose",
    editor = "Sainz, Oscar  and
      Garc{\'\i}a Ferrero, Iker  and
      Agirre, Eneko  and
      Ander Campos, Jon  and
      Jacovi, Alon  and
      Elazar, Yanai  and
      Goldberg, Yoav",
    booktitle = "Proceedings of the 1st Workshop on Data Contamination (CONDA)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.conda-1.2",
    pages = "13--21",
    abstract = "Test contamination is a serious problem for the evaluation of large language models (LLMs) because it leads to the overestimation of their performance and a quick saturation of benchmarks, even before the actual capability is achieved. One strategy to address this issue is the (adversarial) generation of variations, by including different exemplars and different rephrasings of the questions. However, these two interventions can lead to instances that can be more difficult (accumulating on the expected loss of performance by partly removing the contamination) but also to instances that can be less difficult (cancelling the expected loss of performance), which would make contamination undetectable. Understanding these two phenomena in terms of instance difficulty is critical to determine and measure contamination. In this paper we conduct a comprehensive analysis of these two interventions on an addition task with fine-tuned LLAMA-2 models.",
}
```
