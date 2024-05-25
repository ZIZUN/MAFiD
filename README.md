# MAFiD
Code for "MAFiD: Moving Average Equipped Fusion-in-Decoder for Question Answering over Tabular and Textual Data", EACL2023 Findings

## Model test

```python
import src.model
import torch

model_class = src.model.FiDT5
model = model_class.from_pretrained('t5-base')
model = model.to('cpu')

bsz = 2
row_num = 20
txt_len = 5

input_ids = torch.ones((bsz, row_num, txt_len), dtype=torch.long)
attention_mask = torch.ones((bsz, row_num, txt_len), dtype=torch.long)
decoder_input_ids = torch.zeros((bsz, row_num * txt_len), dtype=torch.long)

question_ids = torch.ones((bsz, row_num, txt_len), dtype=torch.long)
question_attention_mask = torch.ones((bsz, row_num, txt_len), dtype=torch.long)
psg_ids=torch.ones((bsz, row_num,  txt_len), dtype=torch.long)
psg_attention_mask=torch.ones((bsz, row_num, txt_len), dtype=torch.long)

model.forward(input_ids=input_ids, attention_mask=attention_mask,
              question_ids=question_ids, question_attention_mask=question_attention_mask,
              psg_ids=psg_ids, psg_attention_mask=psg_attention_mask,
              decoder_input_ids=decoder_input_ids, 
              )

outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_mask,
    question_ids=question_ids, question_attention_mask=question_attention_mask,
    psg_ids=psg_ids, psg_attention_mask=psg_attention_mask,    
    max_length=50,
)

```


## References
- [Fusion-in-Decoder](https://arxiv.org/pdf/2007.01282) 
- [Facebook-Mega](https://arxiv.org/pdf/2209.10655)
- [MLA](https://arxiv.org/pdf/2106.00950)


## Citation
```bibtex


@inproceedings{lee-etal-2023-mafid,
    title = "{MAF}i{D}: Moving Average Equipped Fusion-in-Decoder for Question Answering over Tabular and Textual Data",
    author = "Lee, Sung-Min  and
      Park, Eunhwan  and
      Seo, Daeryong  and
      Jeon, Donghyeon  and
      Kang, Inho  and
      Na, Seung-Hoon",
    editor = "Vlachos, Andreas  and
      Augenstein, Isabelle",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-eacl.177",
    doi = "10.18653/v1/2023.findings-eacl.177",
    pages = "2337--2344",
    abstract = "Transformer-based models for question answering (QA) over tables and texts confront a {``}long{''} hybrid sequence over tabular and textual elements, causing long-range reasoning problems. To handle long-range reasoning, we extensively employ a fusion-in-decoder (FiD) and exponential moving average (EMA), proposing a Moving Average Equipped Fusion-in-Decoder (\textbf{MAFiD}). With FiD as the backbone architecture, MAFiD combines various levels of reasoning: \textit{independent encoding} of homogeneous data and \textit{single-row} and \textit{multi-row heterogeneous reasoning}, using a \textit{gated cross attention layer} to effectively aggregate the three types of representations resulting from various reasonings. Experimental results on HybridQA indicate that MAFiD achieves state-of-the-art performance by increasing exact matching (EM) and F1 by 1.1 and 1.7, respectively, on the blind test set.",
}
```
