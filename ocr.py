from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('stepfun-ai/GOT-OCR2_0', 
                                          trust_remote_code=True, 
                                          cache_dir='/root/autodl-fs/ocr',
                                          )
model = AutoModel.from_pretrained('stepfun-ai/GOT-OCR2_0', 
                                  trust_remote_code=True,
                                  low_cpu_mem_usage=True, 
                                  device_map='cuda', 
                                  use_safetensors=True, 
                                  pad_token_id=tokenizer.eos_token_id,
                                  cache_dir='/root/autodl-fs/ocr',
                                  )
model = model.eval().cuda()
# print(model)

# input your test image
image_file = '/root/autodl-fs/ocr/000a3eb88193b0e076d87f86cb6b5fb6.jpg'

# plain texts OCR
# res = model.chat(tokenizer, image_file, ocr_type='ocr')

# format texts OCR:
# res = model.chat(tokenizer, image_file, ocr_type='format')

# fine-grained OCR:
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_box='')
res = model.chat(tokenizer, image_file, ocr_type='format', ocr_box='[5, 8, 182, 24]')
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_color='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_color='')

# multi-crop OCR:
# res = model.chat_crop(tokenizer, image_file, ocr_type='ocr')
# res = model.chat_crop(tokenizer, image_file, ocr_type='format')

# render the formatted OCR results:
# res = model.chat(tokenizer, image_file, ocr_type='format', render=True, save_render_file = './demo.html')

print(res)
