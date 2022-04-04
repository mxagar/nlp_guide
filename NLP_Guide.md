# NLP Guide

A guide on Natural Language Processing (NLP) structured after following the Udemy course [NLP - Natural Language Processing with Python](https://www.udemy.com/course/nlp-natural-language-processing-with-python/) by José Marcial Portilla.

This file 

`NLP_Guide.md`

provides a general guide of the course and points out to the different notebooks of each section:

1. Setup
2. Python Text Basics
3. NLP Basics
4. Part of Speech Tagging & Named Entity Recognition
5. Text Classification
6. Semantics and Sentiment Analysis
7. Topic Modeling
8. Deep Learning for NLP

Mikel Sagardia, 2022.
No warranties.

## 0. Setup

I installed the following packages/libraries in the environments `ds` and `tf`, which already contained many of the other requirements:

```bash
conda activate ds
conda install spacy keras nltk
conda activate tf
conda install spacy keras nltk
```

However, a YAML with the requirements to create a new conda environment is provided:

`./utils/nlp_course_env.yml`

This is its content:

```yml
name: nlp_course
channels:
  - defaults
dependencies:
  - pip=18.1
  - spacy=2.0.16
  - numpy=1.15.4
  - keras
  - matplotlib=3.0.1
  - pandas=0.23.4
  - nltk=3.3.0
  - scikit-learn=0.20.1
  - jupyter=1.0.0
```

## 1. Python Text Basics