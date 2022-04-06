# NLP Guide

A guide on Natural Language Processing (NLP) structured after following the Udemy course [NLP - Natural Language Processing with Python](https://www.udemy.com/course/nlp-natural-language-processing-with-python/) by José Marcial Portilla.

Note that I would have forked the original repository to add notes, but the material is provided with a download link.

This notes file of mine

`NLP_Guide.md`

provides a general guide of the course and points out to the different notebooks of each section:

0. Setup
1. Python Text Basics: `./01_Python_Text_Basics`
2. NLP Basics
3. Part of Speech Tagging & Named Entity Recognition
4. Text Classification
5. Semantics and Sentiment Analysis
6. Topic Modeling
7. Deep Learning for NLP

Mikel Sagardia, 2022.
No guarantees.

## 0. Setup

I installed the following packages/libraries in the environments `ds` and `tf`, which already contained many of the other requirements:

```bash
conda activate ds
conda install keras nltk
conda install -c conda-forge spacy
# Download dictionaries/models
python -m spacy download en # spacy.load('en_core_news_sm')
python -m spacy download es # spacy.load('es_core_news_sm')
python -m spacy download de # spacy.load('de_core_news_sm')
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

## 1. Python Text Basics: `./01_Python_Text_Basics`

### `01_Text_PDF_Files.ipynb`

This notebook presents the basic python commands to open and handle files and the text in them.

Overview of contents:

1. Working with Text Strings
    - 1.1 f-Strings
    - 1.2 Minimum Widths, Alignment and Padding
    - 1.3 Date Formatting
2. Working with Text Files
    - 2.1 Create a File with Magic Commands
    - 2.2 Opening and Handling Text Files
    - 2.3 Writing to Files
    - 2.4 Appending to a File
    - 2.5 Context Managers
3. Working with PDF Files
    - 3.1 Opening PDFs
    - 3.2 Adding to PDFs
    - 3.3 Example: Extracting Text from PDFs

### `02_Regular_Expressions.ipynb`

This notebook introduces the basics of regular expression searching; functions, identifiers and examples are presented.

Overview of contents:

1. Basic Search Functions
2. Patterns
    - 2.1 Identifiers & Quantifiers
    - 2.2 Groups
    - 2.3 OR Statements: `|`
    - 2.4 Wildcards: `.`
    - 2.5 Starts with and Ends with: `^,$`
    - 2.6 Exclusion: `[^]`
    - 2.7 Brackets for Grouping (Words): `[]+`
    - 2.8 Parenthesis for Multiple Options
    - 2.9 Example: Find Emails

## 2. Natural Language Processing Basics: `02_Natural_Language_Processing_Basics/`

