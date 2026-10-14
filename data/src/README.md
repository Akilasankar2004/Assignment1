# Banking Intent Classification

This project uses the BANKING77 dataset to classify customer intents using a fine-tuned BERT model.

## Folder Structure
- `data/`: Contains the dataset.
- `src/`: Contains the code for dataset preparation, training, and evaluation.
- `model/`: Saved fine-tuned model and tokenizer.

## How to Run
From the project root, install dependencies and run training/evaluation:

```powershell
python -m pip install -r data/src/requirements.txt
Set-Location data
python -m src.train
python -m src.evaluate
```

## Streamlit UI
From the project root, install the dependencies and launch the interactive classifier:

```powershell
python -m pip install -r data/src/requirements.txt
python -m streamlit run app.py
```

Enter a customer message to view the most likely BANKING77 intent in customer-friendly
language and other possible topics. Intent names are aligned to the dataset's numeric
class IDs, which are ordered alphabetically by the original BANKING77 category.

## Dataset
`data/banking77.csv` contains all 13,083 examples from BANKING77 in `text,label` format. Labels are integer IDs from 0 to 76, assigned alphabetically by intent name.

BANKING77 is provided by PolyAI under the Creative Commons Attribution 4.0 International license. Source: [PolyAI task-specific datasets](https://github.com/PolyAI-LDN/task-specific-datasets). Dataset paper: Casanueva et al., “Efficient Intent Detection with Dual Sentence Encoders” (2020), [arXiv:2003.04807](https://arxiv.org/abs/2003.04807).