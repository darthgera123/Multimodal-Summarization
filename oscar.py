### TODO:
# Try adding our 4 pos features
# Try Add TOKTYPE IDS
# Try Set Use_img_layernormtotrue
# Try using transformers version as in theoriginal repo

# Reextract features using the open-sourced bottom up attention to get 2054? 

import os

import torch
import torch.nn as nn

from param import args

from src.vilio.modeling_bertX import BertLayerNorm, GeLU, BertLayer

from src.vilio.modeling_bertO import BertO
from src.vilio.transformers.tokenization_auto import AutoTokenizer
from transformers import EncoderDecoderModel


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids,
        decoder_input_ids=None,decoder_input_mask=None,decoder_segment_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_mask = decoder_input_mask
        self.decoder_segment_ids = decoder_segment_ids

def preprocess_bert(sents,max_seq_len,tokenizer,title=None,max_seq_len_title=None):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for i in range(len(sents)):
        sent = sents[i]
        sent = " ".join(str(sent).split())
        tokens = tokenizer.tokenize(sent)

        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:(max_seq_len - 2)]
            print("Too long: ", tokens)

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        segment_ids = [0] * len(input_ids)

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        if title != None:
            tit = title[i]
            tit = " ".join(str(tit).split())
            tokens_tit = tokenizer.tokenize(tit)

            if len(tokens_tit) > max_seq_len - 2:
                tokens_tit = tokens_tit[:(max_seq_len - 2)]
                print("Too long: ", tokens)

            tokens_tit = ["[CLS]"] + tokens_tit + ["[SEP]"]
            decoder_input_ids = tokenizer.convert_tokens_to_ids(tokens_tit)

            decoder_segment_ids = [0] * len(decoder_input_ids)

            decoder_input_mask = [1] * len(decoder_input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_len_title - len(decoder_input_ids))
            decoder_input_ids += padding
            decoder_input_mask += padding
            decoder_segment_ids += padding
            assert len(decoder_input_ids) == max_seq_len_title
            assert len(decoder_input_mask) == max_seq_len_title
            assert len(decoder_segment_ids) == max_seq_len_title
        else:
            decoder_input_ids = None
            decoder_input_mask = None
            decoder_segment_ids= None
                
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              decoder_input_ids=decoder_input_ids,
                              decoder_input_mask=decoder_input_mask,
                              decoder_segment_ids=decoder_segment_ids))


    # for sent in sents:
    #     # Remove double whitespaces
    #     sent = " ".join(str(sent).split())
    #     tokens = tokenizer.tokenize(sent)

    #     if len(tokens) > max_seq_len - 2:
    #         tokens = tokens[:(max_seq_len - 2)]
    #         print("Too long: ", tokens)

    #     tokens = ["[CLS]"] + tokens + ["[SEP]"]
    #     input_ids = tokenizer.convert_tokens_to_ids(tokens)

    #     segment_ids = [0] * len(input_ids)

    #     input_mask = [1] * len(input_ids)

    #     # Zero-pad up to the sequence length.
    #     padding = [0] * (max_seq_len - len(input_ids))
    #     input_ids += padding
    #     input_mask += padding
    #     segment_ids += padding

    return features

def preprocess_roberta(sents, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for sent in sents:
        # Remove double whitespaces & append whitespace for Roberta
        sent = " " + " ".join(str(sent).split())
        tokens = tokenizer.tokenize(sent)

        # EXP --- 2 </s> as in Roberta
        if len(tokens) > max_seq_len - 3:
            tokens = tokens[:(max_seq_len - 2)]
            print("Too long: ", tokens)


        # Pair of sequences: <s> A </s></s> B </s>
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [0] + input_ids + [2] + [2]

        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        # Pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([1] * padding_length)
            input_mask = input_mask + ([0] * padding_length)
            segment_ids = segment_ids + ([0] * padding_length)

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features


class ModelO(nn.Module):
    """
    Oscar Model with varying Bert Encoders.
    """
    def __init__(self, args=args, max_seq_len=64,max_seq_len_title=32, max_img_seq_len=args.num_features, tr_name=args.tr):
        """
        max_seq_len: Or Repo - VQA: 128
        max_img_seq_len: Or Repo - NLVR2: 40 // GQA: 45 // VQA: 50 --- Set to args.num_features, as we dont have padding implemented
        tr_name: transformer model
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_seq_len_title = max_seq_len_title
        self.tr_name = tr_name
        self.max_img_seq_len = max_img_seq_len

        ### BUILD TOKENIZER ###
        self.tokenizer = AutoTokenizer.from_pretrained(tr_name)

        ### BUILD MODEL ###
        if tr_name.startswith("bert"):
            self.model, loading_info = BertO.from_pretrained(tr_name, output_loading_info=True, 
                                                            img_feature_dim=2048 + args.num_pos)

        print("UNEXPECTED: ", loading_info["unexpected_keys"])
        print("MISSING: ", loading_info["missing_keys"])
        print("ERRORS: ", loading_info["error_msgs"])


        ### CLASSIFICATION HEADS ###
        # LXRT Default classifier tends to perform best; For Albert gelu_new outperforms gelu
        # Make sure to only have used stuff below as it seems to have an effect on random initilization!

        self.encoder_decoder = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'gpt2')

        self.decoder = self.encoder_decoder.decoder
        self.decoder = self.decoder.cuda(0)
        
        self.decoder.config.max_length = 128
        self.decoder.config.min_length = 8
        self.decoder.config.no_repeat_ngram_size = 3
        self.decoder.config.early_stopping = True
        self.decoder.config.length_penalty = 2.0
        self.decoder.config.num_beams = 4

        
        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_weights)
        
    @property
    def dim(self):
        return self.model.config.hidden_size

    def forward(self, sent,title, visual_feats, visual_attention_mask=None):
        
        # if self.tr_name.startswith("bert") or self.tr_name.startswith("albert"):
        #     train_features = preprocess_bert(sents,self.max_seq_len,self.tokenizer,title,self.max_seq_len_title)

        # input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        # input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        # segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).cuda()

        # decoder_input_ids = torch.tensor([f.decoder_input_ids for f in features], dtype=torch.long).cuda()
        # decoder_input_mask = torch.tensor([f.decoder_input_mask for f in features], dtype=torch.long).cuda()
        # decoder_segment_ids = torch.tensor([f.decoder_segment_ids for f in features], dtype=torch.long).cuda()
        input_ids = torch.tensor(sent[:,0,:].clone().detach(),dtype=torch.long).cuda()
        input_mask = torch.tensor(sent[:,1,:].clone().detach(),dtype=torch.long).cuda()
        segment_ids = torch.tensor(sent[:,2,:].clone().detach(),dtype=torch.long).cuda()

        decoder_input_ids = torch.tensor(title[:,0,:].clone().detach(),dtype=torch.long).cuda()
        decoder_input_mask = torch.tensor(title[:,1,:].clone().detach(),dtype=torch.long).cuda()
        decoder_segment_ids = torch.tensor(title[:,2,:].clone().detach(),dtype=torch.long).cuda()

        img_feat, img_pos_feat = visual_feats  

        # Cat Pos feats into img feats
        img_feat = torch.cat((img_feat, img_pos_feat), dim = -1).cuda()
        # They only use 50 feats in or repo
        img_feat = img_feat[:, :self.max_img_seq_len]

        image_mask = torch.ones((input_ids.shape[0], self.max_img_seq_len), dtype=torch.long).cuda()
        input_mask = torch.cat((input_mask, image_mask), dim = -1).cuda()

        seq_out, output,encoder_output = self.model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, img_feats=img_feat)
        
        hidden_state = seq_out
        decoder_output = self.decoder(input_ids= decoder_input_ids,attention_mask=decoder_input_mask,
                                        encoder_hidden_states=hidden_state,labels=decoder_input_ids.clone().detach())
        # decoder outputs are loss,logits (the size will be batchsize*128(max_seq_length)*vocabulary size),past_key_vectors(length 12 not important)


        return decoder_output 

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_O.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load pre-trained model from %s" % path)
        state_dict = torch.load("%s" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            
            if key.startswith("bert.img_embedding.weight"):
                if args.num_pos == 6:
                    new_state_dict[key[5:]] = value
                else:
                    value = value[:, :2052].clone()
                    new_state_dict[key[5:]] = value
                    print("MODIFYING:", key)

            # Masked pretrained model
            elif key.startswith("bert."):
                print("SAVING {} as {}.".format(key, key[5:]))
                new_state_dict[key[5:]] = value
            
            elif key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
          
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)

    def init_weights(self, module):
        """ Initialize the weights """
        print("REINITING: ", module)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def reinit_weights(self, module):
        """ Re-init final bert weights for a better model """

        # This refers to the LXRTEncoder from modeling
        if isinstance(module, nn.ModuleList):
            if isinstance(module[-1], BertLayer):
                print("Reiniting :", module[-1])
                # Reinit that layer: 
                module[-2:].apply(self.init_weights)
        # Alternatively for child in module.children() can be used

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, encoder_outputs=None, **kwargs):

        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
        }

        # Ideally all models should have a :obj:`use_cache`
        # leave following to ifs until all have it implemented
        if "use_cache" in decoder_inputs:
            input_dict["decoder_use_cache"] = decoder_inputs["use_cache"]

        if "past_key_values" in decoder_inputs:
            input_dict["past_key_values"] = decoder_inputs["past_key_values"]

        return input_dict

    def generate(self,features,visual_feats, visual_attention_mask=None):
        # Right now only focussing on text inputs. Need to fix ASAP
        # train_features = preprocess_bert(sents,self.max_seq_len,self.tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).cuda()


        # img_feat, img_pos_feat = visual_feats

        # # Cat Pos feats into img feats
        # img_feat = torch.cat((img_feat, img_pos_feat), dim = -1).cuda()
        # # They only use 50 feats in or repo
        # img_feat = img_feat[:, :self.max_img_seq_len]

        # image_mask = torch.ones((input_ids.shape[0], self.max_img_seq_len), dtype=torch.long)
        # input_mask = torch.cat((input_mask, image_mask), dim = -1).cuda()

        # seq_out, output,encoder_output = self.model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, img_feats=img_feat)
        outputs = self.decoder.generate(input_ids=input_ids,attention_mask=input_mask)
        outputs_str = self.tokenizer.batch_decode(outputs,skip_special_tokens=True)

        return outputs_str


