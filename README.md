# Named Entity Recognition with BERT

**A model that reads a sentence and picks out the people, organisations, places and other named things in it, using the surrounding words to decide what each one is.**

Context is the hard part. "Apple" can be a company or a fruit, "Jordan" a person, a country or a pair of trainers, and "Kent" a county or someone's surname. A simple rule like "capitalised words are names" falls apart quickly. This project fine-tunes BERT, a model that reads every word in light of the words on both sides of it, and compares it with exactly that kind of simple rule.

```
Tim Cook flew to London to announce that Apple will buy 10,000 apples from a farm in Kent.

Tim Cook  → PER (person)        0.999
London    → LOC (location)      0.999
Apple     → ORG (organisation)  0.984
Kent      → LOC (location)      0.999
```

Note that "apples" (the fruit) is correctly left alone, while "Apple" (the company) is tagged as an organisation.

## Results

Evaluated on the CoNLL-2003 test set (3,453 sentences the model never saw during training):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Rule-based baseline (capitalised words) | 0.05 | 0.08 | 0.06 |
| **Fine-tuned BERT (`bert-base-cased`)** | **0.912** | **0.924** | **0.918** |

The baseline tags every capitalised word (apart from the first word of a sentence) as an entity. It barely works because capitalisation alone can't tell a person from a company from a place, so it has to give everything the same label. BERT gets about 92% F1 by learning from context instead.

It also holds up on text it wasn't trained on, like this news paragraph:

| Entity | Label | Confidence |
|---|---|---|
| Andrew Bailey | PER | 0.999 |
| Bank of England | ORG | 0.999 |
| Middle East | LOC | 0.998 |
| European Central Bank | ORG | 0.999 |
| Whitbread | ORG | 0.966 |

## How it works

1. **Load the data.** CoNLL-2003 is a standard benchmark of news sentences where every word is labelled as person, organisation, location, miscellaneous or none. It's loaded with Hugging Face `datasets`.
2. **Tokenise and align labels.** BERT splits words into sub-word pieces (for example "Inzamamul" becomes several tokens), so the word-level labels have to be realigned to the new tokens. Extra pieces and special tokens are masked out so they don't affect training.
3. **Baseline.** A rule-based tagger that marks capitalised words (other than the first word of a sentence) as entities gives a reference score to beat.
4. **Fine-tune BERT.** `bert-base-cased` gets a token-classification head with 9 output labels (IOB2 tags: `B-PER`, `I-PER`, `B-ORG`, and so on). It's trained for 3 epochs with a learning rate of 2e-5 using the Hugging Face `Trainer`.
5. **Evaluate.** Entity-level precision, recall and F1 are computed with `seqeval`, which only counts an entity as correct if its whole span and type match.
6. **Test on real text and analyse errors.** The notebook runs custom sentences and a news paragraph through the model, then pulls out test examples where it got things wrong.

**Tech:** Python · PyTorch · Hugging Face Transformers · Datasets · seqeval

## Running it

The easiest way is Google Colab with a GPU runtime (Runtime → Change runtime type → GPU).

1. Open `ner_pipeline.ipynb` in Colab or Jupyter.
2. Install the dependencies (the first code cell does this for you):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the cells in order. The dataset downloads automatically.

**Training from scratch** (Section 8) takes around 15 minutes on a Colab GPU.

**To skip training**, put the fine-tuned weights in a folder called `saved_ner_model/` next to the notebook, skip Section 8 and run Section 8.1 instead. The weights are about 400 MB, too large for GitHub, so they're not included in this repository.

## What it gets wrong

Section 12 pulls out test sentences to show where the model struggles:

- **Unusual names get split into sub-word pieces.** In the cricket scorecard line "Inzamamul Haq st Germon b Astle 2", BERT breaks "Inzamamul" into "In" + "##zamamul", and the predicted entity boundaries end up in the wrong place even though it recognises a person.
- **Shorthand text with little context**, like sports scorecards (which CoNLL-2003 has plenty of), gives the model very few surrounding words to work from.

## Project structure

```
.
├── ner_pipeline.ipynb   # the full pipeline: data, baseline, training, evaluation, error analysis
├── requirements.txt
└── README.md
```

## Author

**Yaseen Mneimneih** · [GitHub](https://github.com/Yassman04) · [LinkedIn](https://www.linkedin.com/in/yaseen-mneimneih-8a9305233/)
