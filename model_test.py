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
