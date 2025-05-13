from transformers import BertTokenizerFast, AutoTokenizer, RobertaTokenizerFast, XLMRobertaTokenizerFast

def initiate_tokenizer(settings):
    if settings['tokenizer'] == 'globertise':
        tokenizer = RobertaTokenizerFast.from_pretrained('globalise/GloBERTise')
    if settings['tokenizer'] == 'globertisererun':
        tokenizer = RobertaTokenizerFast.from_pretrained('globalise/GloBERTise')
    if settings['tokenizer'] == 'globertisev02':
        tokenizer = RobertaTokenizerFast.from_pretrained('globalise/GloBERTise')
    if settings['tokenizer'] == 'globertisev02rerun':
        tokenizer = RobertaTokenizerFast.from_pretrained('globalise/GloBERTise')
    return(tokenizer)


