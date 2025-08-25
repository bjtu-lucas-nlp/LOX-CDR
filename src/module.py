from transformers import GPT2LMHeadModel, LlamaForCausalLM, LlamaPreTrainedModel, LlamaConfig, PreTrainedModel, AutoModelForCausalLM, MixtralForCausalLM, BitsAndBytesConfig, MixtralPreTrainedModel, MixtralConfig, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch
import copy
import evaluate
from typing import Union, List, Tuple
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict
)
from typing import Optional
from torch.autograd import Function
from bitsandbytes.nn import SwitchBackLinear, Linear8bitLt
import numpy as np
from sentence_transformers import SentenceTransformer, util
# from lit_llama.adapter import LLaMA

class UIPrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, freezeLM=True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # freeze pretrained model parameters
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        model.init_prompt(nuser, nitem)
        return model

    def init_prompt(self, nuser, nitem):
        self.src_len = 2
        emsize = self.transformer.wte.weight.size(1)  # 768
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, user, item, text, mask, ignore_index=-100):
        device = user.device
        batch_size = user.size(0)

        # embeddings
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)  # (batch_size, emsize)
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)
        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), w_src], 1)  # (batch_size, total_len, emsize)

        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)


class ContinuousPromptLearning(UIPrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)



class EncoderLayer(nn.Module):
    def __init__(self, num_input, num_dim):
        super(EncoderLayer, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(num_input, num_dim), nn.ReLU())

    def forward(self, input_data):
        coding = self.encoder(input_data)
        return coding


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        # lucas其中GRL中的forward加上p、就是在训练次数变大的时候，逐渐降低radient reversal layer的权重
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # lucas: GRL所做的就是，就是将传到本层的误差乘以一个负数（-a）系数，这样就会使得GRL前后的网络其训练目标相反，以实现对抗的效果。
        output = grad_output.neg() * ctx.p
        return output, None

# lucas 240326 added for GAN-grl classifier
class Classifier(nn.Module):
    def __init__(self, num_dim):
        super(Classifier, self).__init__()
        # 40维度-10维度
        # self.encoder = EncoderLayer(num_dim_s_2, num_dim_hidden)
        self.classifier = nn.Sequential(
            # lucas 20210607 全连接层，第一个是输入二维张量，第二个是输出二维张量维度
            # lucas 20210607 构建2分类的分类器
            # 10维度-2维度
            nn.Linear(num_dim, 2),
            nn.Sigmoid()
        )

    def forward(self, input_data, p):
        # 40维度-10维度
        # embeds = self.encoder(input_data)
        # embeds_revsers = ReverseLayerF.apply(input_data, p) #lucas疑问？gradient reverse layer?
        embeds_revsers = GradientReversal.apply(input_data, p)
        label = self.classifier(embeds_revsers)
        return label

# Lucas 240708 added pura GRL without linear layer
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GRL, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversal.apply(x, self.lambda_)


# lucas 240311 Prefix tuning + GAN ----> on Mixtral2
class PrefixTune_GAN_Mixtral2(MixtralPreTrainedModel):#LlamaPreTrainedModel, MixtralForCausalLM, MixtralPreTrainedModel
    def __init__(self, config, *args, device_map=None, llm_model_path=None, nuser=None, nitem=None, quanti_config=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super().__init__(config, *args, **kwargs)
        self.device_map = device_map
        self.llm_model_path = llm_model_path
        self.nuser = nuser
        self.nitem = nitem
        self.p = 0.6

        # self.peft_config = args[0]
        self.model_config = config
        self.emb_size = self.model_config.hidden_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        print(f"Lucas module nuser: {self.nuser}; nitem: {self.nitem}; lambda1: {self.lambda1}; lambda2: {self.lambda2}; lambda3: {self.lambda3}")
        print(f"---Lucas--PeftPromptLlama2_V5--config: {self.model_config}--args: {args}")

        # pretrained_llama2_model = LlamaForCausalLM.from_pretrained(config.llm_model_path, config=config,
        #                                                            load_in_8bit=True, torch_dtype=torch.float16,
        #                                                            device_map=config.device_map)  # load_in_8bit=True, torch_dtype=torch.float16, device_map=device_map)
        # pretrained_llama2_model = LlamaForCausalLM.from_pretrained(config.llm_model_path, config=config, load_in_8bit=True, torch_dtype=torch.float16,
        #                                                            device_map=self.device_map)  # load_in_8bit=True, torch_dtype=torch.float16, device_map=device_map)
        # pretrained_llama2_model = LlamaForCausalLM.from_pretrained(config.llm_model_path, config=config,
        #                                                            device_map=self.device_map)
        # lucas 240311 added for 4bit with Mixtral
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        # )
        # pretrained_llm_model = AutoModelForCausalLM.from_pretrained(self.llm_model_path, load_in_4bit=True, torch_dtype=torch.float16,
        #                                                            device_map=self.device_map)  # lucas 230101 带config就报错
        # pretrained_llm_model = AutoModelForCausalLM.from_pretrained(self.llm_model_path, quantization_config=bnb_config,
        #                                                             device_map=self.device_map)  # lucas 230101 带config就报错
        # self.llama2_model = MixtralForCausalLM.from_pretrained(self.llm_model_path, quantization_config=bnb_config,
        #                                                             device_map=self.device_map)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map=device_map, quantization_config=quanti_config)

        print(f"Lucas cur mistral model: {self.llm_model}")
        # pretrained_llm_model = AutoModelForCausalLM.from_pretrained(self.llm_model_path,
        #                                                                device_map=self.device_map)  # lucas 230101 带config就报错
        # lucas 240101 上面成功运行，下面尝试不用from_prtrained的方式
        # pretrained_llama2_model = LlamaForCausalLM.from_pretrained(self.llm_model_path, load_in_8bit=True,
        #                                                            torch_dtype=torch.float16,
        #                                                            device_map=self.device_map)  # lucas 230101 带config就报错

        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.1,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules=["q_proj", "k_proj", "v_proj", 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            target_modules= ["q_proj", "k_proj", "v_proj", 'o_proj', 'gate_proj']
        )
        self.llm_model = prepare_model_for_kbit_training(self.llm_model, use_gradient_checkpointing=True)
        self.llm_model = get_peft_model(self.llm_model, peft_config)  # lucas 231224 added try to train lora
        print(f"Lucas cur mistral model after peft: {self.llm_model}")
        self.cur_device = next(self.llm_model.parameters()).device
        print(f"Lucas--lucas_module--PeftPromptLlama2_V4--load_pretrained--device_map: {self.device_map}--llm_model cur_device: {self.cur_device}")
        # self.llm_model = pretrained_llama2_model.to(self.cur_device)
        # self.llm_model = pretrained_llm_model

        self.concatenate_len = 2  # added user and item embedding
        self.user_embeddings = nn.Embedding(self.nuser, self.emb_size)#.to(device_map[''])
        self.item_embeddings = nn.Embedding(self.nitem, self.emb_size)#.to(device_map[''])

        # Lucas 240326 added grl for user/item embeddings
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
        )
        self.user_semantic_classifier = Classifier(self.emb_size)  # 768, 10

        # Lucas 240326 added 最后预测层
        self.rating_loss = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss()


        # Lucas 240416 change the predict layer
        # self.predict_layer = nn.Linear(self.emb_size * 2, 1)
        dropout = 0.2
        self.predict_emb = self.emb_size * 2
        self.activation = nn.Tanh()
        self.predict_layer = nn.Sequential(nn.Linear(self.predict_emb, self.predict_emb),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(self.predict_emb, self.predict_emb / 2),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(self.predict_emb / 2, 1))

        # lucas 240326 initialize the parameters of modules
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        # Lucas 240416 change the prediction layer initialization
        # nn.init.kaiming_uniform_(self.predict_layer.weight,
        #                          a=1, nonlinearity='sigmoid')
        for layer in self.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

        if not isinstance(self.llm_model, LlamaForCausalLM):
            print(f"Lucas --Error-- PeftPromptLlama2_V5 llm_model is not LlamaForCausalLM")
        else:
            print(f"Lucas -- PeftPromptLlama2_V5 llm_model is LlamaForCausalLM!!")

        if not isinstance(self.llm_model, LlamaPreTrainedModel):
            print(f"Lucas --Error-- PeftPromptLlama2_V5 llm_model is not LlamaPreTrainedModel")
        else:
            print(f"Lucas -- PeftPromptLlama2_V5 llm_model is LlamaPreTrainedModel!!")

        if not isinstance(self, LlamaForCausalLM):
            print(f"Lucas --Error-- PeftPromptLlama2_V5 llm_model is not LlamaForCausalLM")
        else:
            print(f"Lucas -- PeftPromptLlama2_V5 llm_model is LlamaForCausalLM!!")

        if not isinstance(self, LlamaPreTrainedModel):
            print(f"Lucas --Error-- PeftPromptLlama2_V5 is not LlamaPreTrainedModel")
        else:
            print(f"Lucas -- PeftPromptLlama2_V5 is LlamaPreTrainedModel!!")

        if not isinstance(self, PreTrainedModel):
            print(f"Lucas --Error-- PeftPromptLlama2_V5 is not PreTrainedModel")
        else:
            print(f"Lucas -- PeftPromptLlama2_V5 is PreTrainedModel!!")

        print(f"Lucas finish initialize model---PeftPromptLlama2_V5")

    def get_input_embeddings(self):
        # return self.llm_model.model.embed_tokens
        return self.llm_model.base_model.model.model.embed_tokens  # Lucas 240314 after peft added base_model

    def set_input_embeddings(self, value):
        # self.llm_model.model.embed_tokens = value
        self.llm_model.base_model.model.model.embed_tokens = value    # Lucas 240314 after peft added base_model

    def get_output_embeddings(self):
        # return self.llm_model.lm_head
        return self.llm_model.base_model.model.lm_head    # Lucas 240314 after peft added base_model

    def set_output_embeddings(self, new_embeddings):
        # self.llm_model.lm_head = new_embeddings
        self.llm_model.base_model.model.lm_head = new_embeddings  # Lucas 240314 after peft added base_model

    def set_decoder(self, decoder):
        # self.llm_model.model = decoder
        self.llm_model.base_model.model = decoder   # Lucas 240314 after peft added base_model

    def get_decoder(self):
        # return self.llm_model.model
        return self.llm_model.base_model.model  # Lucas 240314 after peft added base_model

    @classmethod
    def load_pretrained(cls):
        cls.llm_model = MixtralForCausalLM.from_pretrained(cls.model_config, cls.device_map)

    def forward(
            self,
            user: torch.LongTensor = None,
            item: torch.LongTensor = None,
            rating: torch.LongTensor = None,
            domain: torch.LongTensor = None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            ignore_index: Optional[int] = -100,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # embeddings
        # user = user.to(torch.float32)
        # item = item.to(torch.float32)
        if rating is not None:
            rating = rating.to(torch.float32)
        if domain is not None:
            domain = domain.to(torch.long)
        # input_ids = input_ids.to(torch.float32)
        # attention_mask = attention_mask.to(torch.float32)
        # labels = labels.to(torch.float32)
        print(f"Lucas model forward-----user: {user};----item:{item};----raing: {rating};----domain: {domain}")
        print(f"Lucas model forward---; input_ids: {input_ids}; attention_mask: {attention_mask}")
        print(f"Lucas model forward---input_ids 0: {input_ids[0]}; ")
        print(f"Lucas model forward--labels: {labels}")
        # print(f"Lucas model forward--labels[0]: {labels[0]}")

        # # lucas 231123 change the forward function
        # tokens_emb = self.llm_model.model.embed_tokens(input_ids)
        # return self.llm_model.forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user_embeddings device: {next(self.user_embeddings.parameters()).device}")

        user_emb = self.user_embeddings(user)#.to(self.device_map[''])  # (batch_size, emsize)
        item_emb = self.item_embeddings(item)#.to(self.device_map[''])  # (batch_size, emsize)

        # lucas 240326 added for tune user/item embedding
        user_emb = self.hidden_layers(user_emb)
        item_emb = self.hidden_layers(item_emb)
        # 230326 通过GRLayer GAN生成器判断source和target类别
        # p = float(batch_int + (epoch - dann_epoch) * num_batch / (args.epochs - dann_epoch) / num_batch)
        # p = 2. / (1. + np.exp(-10 * p)) - 1
        predict_domain = self.user_semantic_classifier(user_emb, self.p)  # classifier，输入40维度，输出10维度,经过GRL再经过线性层+sigmoid输出2维分类

        cur_device = user_emb.device
        # print(f"Lucas model get_input_embeddings device: {next(self.llm_model.base_model.model.get_input_embeddings().parameters()).device}")
        tokens_emb = self.llm_model.base_model.model.get_input_embeddings()(input_ids).to(cur_device)  # (batch_size, tgt_len, emsize)
        print(f"Lucas--lucas_module--user_emb size: {user_emb.size()}--item_emb size: {item_emb.size()}--tokens_emb size: {tokens_emb.size()}")
        # print(f"Lucas user_emb: {user_emb};")
        # print(f"Lucas  item_emb: {item_emb};")
        # print(f"Lucas tokens_emb: {tokens_emb}")
        concatenate_emb = torch.cat([user_emb.unsqueeze(1), item_emb.unsqueeze(1), tokens_emb], 1)  # (batch_size, total_len, emsize)
        concatenate_emb = concatenate_emb.to(torch.float32)

        # Lucas 240327 changed the prefix padding values
        # pad_left = torch.ones((concatenate_emb.size()[0], self.concatenate_len), dtype=torch.int64, device=cur_device)
        pad_left = torch.zeros((concatenate_emb.size()[0], self.concatenate_len), dtype=torch.int64, device=cur_device)

        if attention_mask is None:
            # auto-regressive generation
            return self.llm_model.forward(inputs_embeds=concatenate_emb)
        else:
            # training
            # input padding
            print(f"Lucas--lucas_module--pad_left size: {pad_left.size()}---attention_mask size: {attention_mask.size()}")
            print(f"Lucas pad_left: {pad_left}; attention_mask: {attention_mask}")
            pad_input = torch.cat([pad_left, attention_mask], 1)  # (batch_size, total_len)

            # prediction for training
            # self.pred_left = torch.full((self.batch_size, self.concatenate_len), ignore_index, dtype=torch.int64)  # (batch_size, src_len)
            print(f"Lucas--lucas_module---concatenate_len: {self.concatenate_len}--concatenate_emb.size()[0]: {concatenate_emb.size()[0]}")
            self.pred_left = torch.full((concatenate_emb.size()[0], self.concatenate_len), ignore_index, device=self.cur_device)
            if attention_mask is not None:
                if labels is not None:
                    print(f"Lucas--lucas_module.py--attention_mask size: {attention_mask.size()}--labels size: {labels.size()}")
                else:
                    print(f"Lucas--lucas_module.py--labels is None")
            else:
                print(f"Lucas--lucas_module.py--attention_mask is None")
            # self.pred_right = torch.where(attention_mask == 1, labels, torch.tensor(ignore_index).to(self.cur_device)).to(self.cur_device)  # replace <pad> with ignore_index
            self.pred_right = torch.where(attention_mask == 1, labels, ignore_index).to(self.cur_device)  # replace <pad> with ignore_index
            prediction = torch.cat([self.pred_left, self.pred_right], 1)  # (batch_size, total_len)
            print(f"Lucas--lucas_module.py--pad_input: {pad_input}--concatenate_emb: {concatenate_emb}--prediction: {prediction}")
            print(f"Lucas--lucas_module.py--pad_input size: {pad_input.size()}--prediction size: {prediction.size()}--concatenate_emb size: {concatenate_emb.size()}")
            print(f"Lucas--lucas_module.py--pad_input device: {pad_input.device}--prediction device: {prediction.device}--concatenate_emb device: {concatenate_emb.device}")
            print(f"Lucas--lucas_module.pu--model device: {self.llm_model.device}")

            # Lucas 240327 added for rating prediction
            concat_emb = torch.cat((user_emb, item_emb), -1)
            predict_ratings = self.predict_layer(concat_emb)
            print(f"Lucas--predict_ratings: {predict_ratings}; original ratings: {rating}")
            rating_loss = self.rating_loss(predict_ratings.view(-1), rating)
            print(f"Lucas--rating_loss: {rating_loss}")
            # return_loss_dict["scores"] = prediction.view(-1)

            # GRL classification loss
            print(f"Lucas--predict_domains: {predict_domain}; original domains: {domain}")
            predict_domains_loss = self.exp_loss_fn(predict_domain, domain)
            print(f"Lucas--domain prediction loss: {predict_domains_loss}")

            # lucas231122 change the following code to return llm_model directly
            # print(f"Lucas--lucas_module.py--model-forward--\npad_input: {pad_input}")
            # print(f"concatenate_emb: {concatenate_emb}")
            # print(f"prediction: {prediction}")
            # output = self.llm_model.forward(attention_mask=pad_input, inputs_embeds=concatenate_emb, labels=prediction)
            print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
            output = self.llm_model.forward(inputs_embeds=concatenate_emb, attention_mask=pad_input,
                                               labels=prediction)   # {'loss':, 'logits':, 'past_key_values', 'hidden_states', 'attentions'}
            print(f"Lucas--llm loss: {output}")
            print(f"Lucas--llm loss device: {output['loss'].device}; rating loss device: {rating_loss.device}; domain loss: {predict_domains_loss.device}")
            print(f"Lucas--llm loss dtype: {output['loss'].dtype};--rating loss dtype: {rating_loss.dtype};--domain loss dtype: {predict_domains_loss.dtype}")
            output['loss'] = self.lambda3*output['loss'].to(torch.float32) + self.lambda1*rating_loss.to(output['loss'].device) + self.lambda2*predict_domains_loss.to(output['loss'].device)
            # output = self.llm_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # print(f"Lucas--lucas_module V7 forward result: {output}")
            output['rating_prediction'] = predict_ratings.view(-1)
            output['domain_prediction'] = predict_domain
            print(f"Lucas--final loss: {output}")
            return output

    # lucas 240117 add inputs interface for generation, otherwise generate func do not take user/item/labels as input
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                # "attention_mask": attention_mask,
                "user": kwargs.get("user"),
                "item": kwargs.get("item"),
                # "labels": kwargs.get("labels")
            }
        )
        return model_inputs


# lucas 240402 NCF + Prefix tuning + GAN -----> on Mixtral2
class PrefixTune_NCF_GAN_Mixtral2(MixtralPreTrainedModel):#LlamaPreTrainedModel, MixtralForCausalLM, MixtralPreTrainedModel
    def __init__(self, config, *args, device_map=None, llm_model_path=None, nuser=None, nitem=None, quanti_config=None, factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super().__init__(config, *args, **kwargs)
        self.device_map = device_map
        self.llm_model_path = llm_model_path
        self.nuser = nuser
        self.nitem = nitem
        self.factor_num = factor_num
        self.ncf_layer_num = ncf_layer_num
        self.drop_out = drop_out
        self.p = 0.6

        # self.peft_config = args[0]
        self.model_config = config
        self.emb_size = self.model_config.hidden_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        print(f"Lucas module nuser: {self.nuser}; nitem: {self.nitem}; lambda1: {self.lambda1}; lambda2: {self.lambda2}; lambda3: {self.lambda3}")
        print(f"Lucas module factor_num: {self.factor_num}")
        print(f"---Lucas--PeftPromptLlama2_V5--config: {self.model_config}--args: {args}")

        # self.llama2_model = MixtralForCausalLM.from_pretrained(self.llm_model_path, quantization_config=bnb_config,
        #                                                             device_map=self.device_map)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map=device_map, quantization_config=quanti_config)

        print(f"Lucas cur mistral model: {self.llm_model}")
        # pretrained_llm_model = AutoModelForCausalLM.from_pretrained(self.llm_model_path,
        #                                                                device_map=self.device_map)  # lucas 230101 带config就报错
        # lucas 240101 上面成功运行，下面尝试不用from_prtrained的方式
        # pretrained_llama2_model = LlamaForCausalLM.from_pretrained(self.llm_model_path, load_in_8bit=True,
        #                                                            torch_dtype=torch.float16,
        #                                                            device_map=self.device_map)  # lucas 230101 带config就报错
        # Lucas 240506
        # r：更新矩阵的秩，以 int 表示。较低的秩会导致较小的更新矩阵和较少的可训练参数。
        # target_modules：模型中使用LoRA更新矩阵的模块，模型中常见的是，更新注意力模块
        # alpha ：LoRA 缩放因子
        # bias ：指定是否应训练 bias 参数。可以是 'none' 、 'all' 或 'lora_only'
        # lora_dropout：dropout的比例
        # task_type：模型任务类型，这里我们使用CAUSAL_LM任务
        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.1,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", 'o_proj', 'gate_proj']
        )
        self.llm_model = prepare_model_for_kbit_training(self.llm_model, use_gradient_checkpointing=True)
        self.llm_model = get_peft_model(self.llm_model, peft_config)  # lucas 231224 added try to train lora
        print(f"Lucas cur mistral model after peft: {self.llm_model}")
        self.cur_device = next(self.llm_model.parameters()).device
        print(f"Lucas--lucas_module--PeftPromptLlama2_V4--load_pretrained--device_map: {self.device_map}--llm_model cur_device: {self.cur_device}")
        # self.llm_model = pretrained_llama2_model.to(self.cur_device)
        # self.llm_model = pretrained_llm_model

        # lucas 240402 NCF module
        self.embed_user_GMF = nn.Embedding(self.nuser, self.factor_num)
        self.embed_item_GMF = nn.Embedding(self.nitem, self.factor_num)
        # lucas 230327 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        self.embed_user_MLP = nn.Embedding(
            self.nuser, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.nitem, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # layer1：256-128；layer2: 128-64; layer3: 64-32
        MLP_modules = []
        for i in range(self.ncf_layer_num):
            input_size = self.factor_num * (2 ** (self.ncf_layer_num - i))
            MLP_modules.append(nn.Dropout(p=self.drop_out))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)


        self.concatenate_len = 2  # added user and item embedding
        self.user_embeddings = nn.Embedding(self.nuser, self.emb_size)#.to(device_map[''])
        self.item_embeddings = nn.Embedding(self.nitem, self.emb_size)#.to(device_map[''])

        # Lucas 240326 added grl for user/item embeddings
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
        )
        self.user_semantic_classifier = Classifier(self.emb_size)  # 768, 10

        # Lucas 240326 added 最后预测层
        self.rating_loss = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss()

        self.predict_emb = self.factor_num * 2 + self.emb_size * 2
        # Lucas 240416 change the predict layer
        # self.predict_layer = nn.Linear(self.predict_emb, 1)
        dropout = 0.2
        self.activation = nn.Tanh()
        self.predict_layer = nn.Sequential(nn.Linear(self.predict_emb, self.predict_emb),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(self.predict_emb, 512),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(512, 1))

        # lucas 240326 initialize the parameters of modules
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        # nn.init.xavier_uniform_(self.transform_matrix.weight)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        # Lucas 240416 change the prediction layer initialization
        # nn.init.kaiming_uniform_(self.predict_layer.weight,
        #                          a=1, nonlinearity='sigmoid')
        for layer in self.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

        if not isinstance(self.llm_model, MixtralPreTrainedModel):
            print(f"Lucas --Error-- PrefixTune_NCF_GAN_Mixtral2 llm_model is not MixtralPreTrainedModel")
        else:
            print(f"Lucas -- PrefixTune_NCF_GAN_Mixtral2 llm_model is MixtralPreTrainedModel!!")

        print(f"Lucas finish initialize model---PrefixTune_NCF_GAN_Mixtral2")

    def get_input_embeddings(self):
        # return self.llm_model.model.embed_tokens
        return self.llm_model.base_model.model.model.embed_tokens  # Lucas 240314 after peft added base_model

    def set_input_embeddings(self, value):
        # self.llm_model.model.embed_tokens = value
        self.llm_model.base_model.model.model.embed_tokens = value    # Lucas 240314 after peft added base_model

    def get_output_embeddings(self):
        # return self.llm_model.lm_head
        return self.llm_model.base_model.model.lm_head    # Lucas 240314 after peft added base_model

    def set_output_embeddings(self, new_embeddings):
        # self.llm_model.lm_head = new_embeddings
        self.llm_model.base_model.model.lm_head = new_embeddings  # Lucas 240314 after peft added base_model

    def set_decoder(self, decoder):
        # self.llm_model.model = decoder
        self.llm_model.base_model.model = decoder   # Lucas 240314 after peft added base_model

    def get_decoder(self):
        # return self.llm_model.model
        return self.llm_model.base_model.model  # Lucas 240314 after peft added base_model

    @classmethod
    def load_pretrained(cls):
        cls.llm_model = MixtralForCausalLM.from_pretrained(cls.model_config, cls.device_map)

    def forward(
            self,
            user: torch.LongTensor = None,
            item: torch.LongTensor = None,
            rating: torch.LongTensor = None,
            domain: torch.LongTensor = None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            ignore_index: Optional[int] = -100,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # embeddings
        # user = user.to(torch.float32)
        # item = item.to(torch.float32)
        if rating is not None:
            rating = rating.to(torch.float32)
        if domain is not None:
            domain = domain.to(torch.long)
        # input_ids = input_ids.to(torch.float32)
        # attention_mask = attention_mask.to(torch.float32)
        # labels = labels.to(torch.float32)
        print(f"Lucas model forward-----user: {user};----item:{item};----raing: {rating};----domain: {domain}")
        print(f"Lucas model forward---; input_ids: {input_ids}; attention_mask: {attention_mask}")
        print(f"Lucas model forward---input_ids 0: {input_ids[0]}; ")
        print(f"Lucas model forward--labels: {labels}")
        # print(f"Lucas model forward--labels[0]: {labels[0]}")

        # # lucas 231123 change the forward function
        # tokens_emb = self.llm_model.model.embed_tokens(input_ids)
        # return self.llm_model.forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user_embeddings device: {next(self.user_embeddings.parameters()).device}")

        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        # lucas 240402 added GRL layer for user embeddings in NCF
        emb_user_NCF = torch.cat((embed_user_GMF, embed_user_MLP), -1)
        user_NCF_dom_label = self.user_NCF_classifier(emb_user_NCF, self.p)

        user_emb = self.user_embeddings(user)#.to(self.device_map[''])  # (batch_size, emsize)
        item_emb = self.item_embeddings(item)#.to(self.device_map[''])  # (batch_size, emsize)

        # lucas 240326 added for tune user/item embedding
        user_emb = self.hidden_layers(user_emb)
        item_emb = self.hidden_layers(item_emb)
        # 230326 通过GRLayer GAN生成器判断source和target类别
        # p = float(batch_int + (epoch - dann_epoch) * num_batch / (args.epochs - dann_epoch) / num_batch)
        # p = 2. / (1. + np.exp(-10 * p)) - 1
        predict_domain = self.user_semantic_classifier(user_emb, self.p)  # classifier，输入40维度，输出10维度,经过GRL再经过线性层+sigmoid输出2维分类

        cur_device = user_emb.device
        # print(f"Lucas model get_input_embeddings device: {next(self.llm_model.base_model.model.get_input_embeddings().parameters()).device}")
        tokens_emb = self.llm_model.base_model.model.get_input_embeddings()(input_ids).to(cur_device)  # (batch_size, tgt_len, emsize)
        print(f"Lucas--lucas_module--user_emb size: {user_emb.size()}--item_emb size: {item_emb.size()}--tokens_emb size: {tokens_emb.size()}")
        # print(f"Lucas user_emb: {user_emb};")
        # print(f"Lucas  item_emb: {item_emb};")
        # print(f"Lucas tokens_emb: {tokens_emb}")
        concatenate_emb = torch.cat([user_emb.unsqueeze(1), item_emb.unsqueeze(1), tokens_emb], 1)  # (batch_size, total_len, emsize)
        concatenate_emb = concatenate_emb.to(torch.float32)

        # Lucas 240327 changed the prefix padding values
        # pad_left = torch.ones((concatenate_emb.size()[0], self.concatenate_len), dtype=torch.int64, device=cur_device)
        pad_left = torch.zeros((concatenate_emb.size()[0], self.concatenate_len), dtype=torch.int64, device=cur_device)

        if attention_mask is None:
            # auto-regressive generation
            return self.llm_model.forward(inputs_embeds=concatenate_emb)
        else:
            # training
            # input padding
            print(f"Lucas--lucas_module--pad_left size: {pad_left.size()}---attention_mask size: {attention_mask.size()}")
            print(f"Lucas pad_left: {pad_left}; attention_mask: {attention_mask}")
            pad_input = torch.cat([pad_left, attention_mask], 1)  # (batch_size, total_len)

            # prediction for training
            # self.pred_left = torch.full((self.batch_size, self.concatenate_len), ignore_index, dtype=torch.int64)  # (batch_size, src_len)
            print(f"Lucas--lucas_module---concatenate_len: {self.concatenate_len}--concatenate_emb.size()[0]: {concatenate_emb.size()[0]}")
            self.pred_left = torch.full((concatenate_emb.size()[0], self.concatenate_len), ignore_index, device=self.cur_device)
            if attention_mask is not None:
                if labels is not None:
                    print(f"Lucas--lucas_module.py--attention_mask size: {attention_mask.size()}--labels size: {labels.size()}")
                else:
                    print(f"Lucas--lucas_module.py--labels is None")
            else:
                print(f"Lucas--lucas_module.py--attention_mask is None")
            # self.pred_right = torch.where(attention_mask == 1, labels, torch.tensor(ignore_index).to(self.cur_device)).to(self.cur_device)  # replace <pad> with ignore_index
            self.pred_right = torch.where(attention_mask == 1, labels, ignore_index).to(self.cur_device)  # replace <pad> with ignore_index
            prediction = torch.cat([self.pred_left, self.pred_right], 1)  # (batch_size, total_len)
            print(f"Lucas--lucas_module.py--pad_input: {pad_input}--concatenate_emb: {concatenate_emb}--prediction: {prediction}")
            print(f"Lucas--lucas_module.py--pad_input size: {pad_input.size()}--prediction size: {prediction.size()}--concatenate_emb size: {concatenate_emb.size()}")
            print(f"Lucas--lucas_module.py--pad_input device: {pad_input.device}--prediction device: {prediction.device}--concatenate_emb device: {concatenate_emb.device}")
            print(f"Lucas--lucas_module.pu--model device: {self.llm_model.device}")

            # Lucas 240327 added for rating prediction
            print(f"Lucas GMF size: {output_GMF.size()}; MLP size: {output_MLP.size()}; user_emb size: {user_emb.size()}; item emb size: {item_emb.size()}")
            concat_emb = torch.cat((output_GMF, output_MLP, user_emb, item_emb), -1)
            predict_ratings = self.predict_layer(concat_emb)
            print(f"Lucas--predict_ratings: {predict_ratings}; original ratings: {rating}")
            rating_loss = self.rating_loss(predict_ratings.view(-1), rating)
            # print(f"Lucas--rating_loss: {rating_loss}")
            # return_loss_dict["scores"] = prediction.view(-1)

            # GRL classification loss
            print(f"Lucas--predict_domains: {predict_domain}; original domains: {domain}")
            predict_domains_loss = self.exp_loss_fn(predict_domain, domain)
            predict_ncf_domain_loss = self.exp_loss_fn(user_NCF_dom_label, domain)
            # print(f"Lucas--domain prediction loss: {predict_domains_loss}")

            # lucas231122 change the following code to return llm_model directly
            # print(f"Lucas--lucas_module.py--model-forward--\npad_input: {pad_input}")
            # print(f"concatenate_emb: {concatenate_emb}")
            # print(f"prediction: {prediction}")
            # output = self.llm_model.forward(attention_mask=pad_input, inputs_embeds=concatenate_emb, labels=prediction)e
            print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
            output = self.llm_model.forward(inputs_embeds=concatenate_emb, attention_mask=pad_input,
                                               labels=prediction)   # {'loss':, 'logits':, 'past_key_values', 'hidden_states', 'attentions'}
            print(f"Lucas--llm loss: {output['loss']}; rating loss: {rating_loss}; domain prediction loss: {predict_domains_loss}")
            print(f"Lucas--llm loss device: {output['loss'].device}; rating loss device: {rating_loss.device}; domain loss dtype: {predict_domains_loss.device}; predict_ncf_domain loss: {predict_ncf_domain_loss.device}")
            output['loss'] = self.lambda3*output['loss'].to(torch.float32) + self.lambda1*rating_loss.to(output['loss'].device) + self.lambda2*predict_domains_loss.to(output['loss'].device) + self.lambda2*predict_ncf_domain_loss.to(output['loss'].device)
            # output = self.llm_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # print(f"Lucas--lucas_module V7 forward result: {output}")
            output['rating_prediction'] = predict_ratings.view(-1)
            output['domain_prediction'] = predict_domain
            output['ncf_domain_prediction'] = user_NCF_dom_label
            print(f"Lucas--final loss: {output}")
            return output

    # lucas 240117 add inputs interface for generation, otherwise generate func do not take user/item/labels as input
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                # "attention_mask": attention_mask,
                "user": kwargs.get("user"),
                "item": kwargs.get("item"),
                # "labels": kwargs.get("labels")
            }
        )
        return model_inputs


class FeaturePrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def forward(self, context, explanation, exp_mask, ignore_index=-100):
        device = context.device
        text = torch.cat([context, explanation], 1)  # (batch_size, total_len)
        src = self.transformer.wte(text)  # (batch_size, total_len, emsize)

        if exp_mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones_like(context, dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, exp_mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full_like(context, ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(exp_mask == 1, explanation, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)


class DiscretePromptLearning(FeaturePrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)


class MF(nn.Module):
    def __init__(self):
        super(MF, self).__init__()

    def forward(self, user, item):  # (batch_size, emsize)
        rating = torch.sum(user * item, 1)  # (batch_size,)
        return rating


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLP(nn.Module):
    def __init__(self, emsize, hidden_size=400, num_layers=2):
        super(MLP, self).__init__()
        self.first_layer = nn.Linear(emsize * 2, hidden_size)
        self.last_layer = nn.Linear(hidden_size, 1)
        layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(layer, num_layers)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-initrange, initrange)
        self.last_layer.bias.data.zero_()
        for layer in self.layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, user, item):  # (batch_size, emsize)
        ui_cat = torch.cat([user, item], 1)  # (batch_size, emsize * 2)
        hidden = self.sigmoid(self.first_layer(ui_cat))  # (batch_size, hidden_size)
        for layer in self.layers:
            hidden = self.sigmoid(layer(hidden))  # (batch_size, hidden_size)
        rating = torch.squeeze(self.last_layer(hidden))  # (batch_size,)
        return rating


# lucas 240701 added contrastive learning loss
def info_nce_loss(anchor, positive, negative, temperature=0.1):
    """
    Compute the InfoNCE loss.

    Args:
    - anchor: Tensor of shape (batch_size, embedding_dim)
    - positive: Tensor of shape (batch_size, embedding_dim)
    - negative: Tensor of shape (batch_size, num_negatives, embedding_dim)
    - temperature: A scalar temperature parameter

    Returns:
    - loss: The InfoNCE loss
    """
    # Normalize the embeddings
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negative = F.normalize(negative, dim=2)

    # Compute logits
    pos_logit = torch.sum(anchor * positive, dim=1) / temperature
    neg_logits = torch.bmm(negative, anchor.unsqueeze(2)).squeeze(2) / temperature

    # Compute the InfoNCE loss
    logits = torch.cat([pos_logit.unsqueeze(1), neg_logits], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    loss = F.cross_entropy(logits, labels)

    return loss


# Lucas 240709 added for feature selection in NCF-GAN
class FeatureSelector(nn.Module):
    def __init__(self, input_dim):
        super(FeatureSelector, self).__init__()
        self.importance = nn.Parameter(torch.ones(input_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        importance = self.sigmoid(self.importance)
        return x * importance


# Lucas 240705 add NCF-GAN for ablation study and code test
# NCF + GAN-->0.63512, 1.04697, 0.46975, 0.88817
class NCF_GAN(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, lambda_grl=1.0, temperature=0.1, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_GAN, self).__init__()
        self.device_map = device_map
        self.num_u = num_u
        self.num_i = num_i
        self.num_neg_i = num_neg_i    # added for contrastive learning
        self.temperature = temperature  # added for contrastive learning
        self.factor_num = factor_num
        self.ncf_layer_num = ncf_layer_num
        self.drop_out = drop_out

        # self.peft_config = args[0]
        # self.model_config = config
        self.emb_size = emb_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda_grl = lambda_grl
        print(f"Lucas module num_u: {self.num_u}; num_i: {self.num_i}; lambda1: {self.lambda1}; lambda2: {self.lambda2}; lambda3: {self.lambda3}")
        print(f"Lucas module factor_num: {self.factor_num}")

        # print(f"Lucas--lucas_module--PeftPromptLlama2_V4--load_pretrained--device_map: {self.device_map}")

        # lucas 240402 NCF module
        self.embed_user_GMF = nn.Embedding(self.num_u, self.factor_num)
        self.embed_item_GMF = nn.Embedding(self.num_i, self.factor_num)
        # lucas 230327 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        self.embed_user_MLP = nn.Embedding(
            self.num_u, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.num_i, self.factor_num * (2 ** (self.ncf_layer_num - 1)))

        # nn.Sigmoid()
        # layer1：256-128；layer2: 128-64; layer3: 64-32
        mlp_modules = []
        mlp_out_size = []
        for i in range(self.ncf_layer_num):
            input_size = self.factor_num * (2 ** (self.ncf_layer_num - i))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.Dropout(p=self.drop_out))
            mlp_modules.append(nn.BatchNorm1d(input_size // 2))
            mlp_out_size.append(input_size)
            mlp_out_size.append(input_size // 2)
            mlp_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*mlp_modules)

        # Lucas 240731 added for grl and cl
        self.emb_fusion = nn.Sequential(
            nn.Linear(self.factor_num + mlp_out_size[-1], self.factor_num + mlp_out_size[-1]),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num + mlp_out_size[-1]),
            nn.Sigmoid(),
        )

        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.grl = GRL(self.lambda_grl)  # Lucas 240708 changed grl
        print(f"Lucas GRL mlp_out size: {mlp_out_size}")
        # self.domain_fc = nn.Linear(self.factor_num + mlp_out_size[-1], 2)
        self.domain_fc = nn.Sequential(
            # nn.Linear(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)), self.factor_num),
            nn.Linear(self.factor_num + mlp_out_size[-1], self.factor_num),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, self.factor_num),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, 2)
        )
        self.grl_activation = nn.Sigmoid()

        # # Lucas 240709 added for feature selection
        # self.feature_selector = FeatureSelector(self.factor_num + mlp_out_size[-1])

        self.concatenate_len = 2  # added user and item embedding
        self.user_embeddings = nn.Embedding(self.num_u, self.emb_size)#.to(device_map[''])
        self.item_embeddings = nn.Embedding(self.num_i, self.emb_size)#.to(device_map[''])

        # Lucas 240326 added grl for user/item embeddings
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.emb_size),
            # nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.emb_size),
            # nn.LayerNorm(self.emb_size),
        )
        self.user_semantic_classifier = Classifier(self.emb_size)  # 768, 10

        # Lucas 240326 added 最后预测层
        # self.rating_loss = nn.MSELoss()
        self.rating_loss = nn.SmoothL1Loss()
        self.exp_loss_fn = nn.CrossEntropyLoss()


        # self.predict_emb = self.factor_num * 2 # + self.emb_size * 2
        self.predict_emb = self.factor_num + mlp_out_size[-1]
        print(f"Lucas predict_emb: {self.predict_emb}")
        # Lucas 240416 change the predict layer
        # self.predict_layer = nn.Linear(self.predict_emb, 1)
        dropout = 0.2
        self.activation = nn.Tanh()
        self.predict_layer = nn.Sequential(nn.Linear(self.predict_emb, self.predict_emb),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.BatchNorm1d(self.predict_emb),
                                           nn.Linear(self.predict_emb, self.factor_num),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(self.factor_num, 1))

        # # Lucas 240701 added for Contrastive Learning
        # self.contrastive_learn = nn.Linear(self.predict_emb, int(self.predict_emb / 2))

        # ******************************lucas 240326 initialize the parameters of modules
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        # nn.init.xavier_uniform_(self.transform_matrix.weight)
        initrange = 0.1
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        for m in self.emb_fusion:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        for m in self.domain_fc:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        # for m in self.modules():
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         m.bias.data.zero_()

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        # Lucas 240416 change the prediction layer initialization
        # nn.init.kaiming_uniform_(self.predict_layer.weight,
        #                          a=1, nonlinearity='sigmoid')
        for layer in self.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

        print(f"Lucas finish initialize model---PrefixTune_NCF_GAN_Mixtral2")

    def forward(
            self,
            user: torch.LongTensor = None,
            item: torch.LongTensor = None,
            rating: torch.LongTensor = None,
            domain: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if rating is not None:
            rating = rating.to(torch.float32)
        if domain is not None:
            domain = domain.to(torch.long)
        # print(f"Lucas model forward-----user: {user};----item:{item};----rating: {rating};----domain: {domain}")

        embed_u_gmf = self.embed_user_GMF(user)
        embed_i_gmf = self.embed_item_GMF(item)
        output_gmf = embed_u_gmf * embed_i_gmf
        # # Lucas 240702 added for contrastive learning
        # user_extend = user.unsqueeze(1).repeat(1, self.num_neg_i).view(-1)
        # print(f"Lucas model forward-----extend user: {user_extend};")
        # embed_neg_u_gmf = self.embed_user_GMF(user_extend)
        # embed_neg_i_gmf = self.embed_item_GMF(neg_items)
        # output_neg_gmf = embed_neg_u_gmf * embed_neg_i_gmf

        embed_u_mlp = self.embed_user_MLP(user)
        embed_i_mlp = self.embed_item_MLP(item)
        interaction = torch.cat((embed_u_mlp, embed_i_mlp), -1)
        output_mlp = self.MLP_layers(interaction)
        # # Lucas 240702 added for contrastive learning
        # embed_neg_u_mlp = self.embed_user_MLP(user_extend)
        # embed_neg_i_mlp = self.embed_item_MLP(neg_items)
        # neg_inter = torch.cat((embed_neg_u_mlp, embed_neg_i_mlp), -1)
        # output_neg_mlp = self.MLP_layers(neg_inter)

        # lucas 240402 added GRL layer for user embeddings in NCF
        # emb_u_ncf = torch.cat((embed_u_gmf, embed_u_mlp), -1)
        # u_ncf_dom_label = self.user_NCF_classifier(emb_u_ncf, self.p)
        # Lucas 240708 changed GRL input layer
        concat_emb = torch.cat((output_gmf, output_mlp), -1)
        concat_emb = self.emb_fusion(concat_emb)    # lucas 240731added for feature selection

        # Lucas 240709 added feature selection layer, 效果没有提升
        # selected_concat_emb = self.feature_selector(concat_emb)

        # Domain classification task
        concat_emb_grl = self.grl(concat_emb)
        u_ncf_dom_label = self.grl_activation(self.domain_fc(concat_emb_grl))

        # # Lucas 240702 added for contrastive learning
        # emb_neg_u_ncf = torch.cat((embed_neg_u_gmf, embed_neg_u_mlp), -1)
        # neg_u_ncf_dom_label = self.user_NCF_classifier(emb_neg_u_ncf, self.p)

        # user_emb = self.user_embeddings(user)#.to(self.device_map[''])  # (batch_size, emsize)
        # item_emb = self.item_embeddings(item)#.to(self.device_map[''])  # (batch_size, emsize)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.user_embeddings(user_extend)
        # neg_i_emb = self.item_embeddings(neg_items)

        # lucas 240326 added for tune user/item embedding
        # user_emb = self.hidden_layers(user_emb)
        # item_emb = self.hidden_layers(item_emb)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.hidden_layers(neg_u_emb)
        # neg_i_emb = self.hidden_layers(neg_i_emb)
        # 230326 通过GRLayer GAN生成器判断source和target类别
        # p = float(batch_int + (epoch - dann_epoch) * num_batch / (args.epochs - dann_epoch) / num_batch)
        # p = 2. / (1. + np.exp(-10 * p)) - 1
        # predict_domain = self.user_semantic_classifier(user_emb, self.p)  # classifier，输入40维度，输出10维度,经过GRL再经过线性层+sigmoid输出2维分类
        # # Lucas 240702 added for contrastive learning
        # predict_neg_domain = self.user_semantic_classifier(neg_u_emb, self.p)

        # *****************************************training
        # Lucas 240327 added for rating prediction
        print(f"Lucas GMF size: {output_gmf.size()}; MLP size: {output_mlp.size()};")
        # concat_emb = torch.cat((output_gmf, output_mlp), -1)
        predict_ratings = self.predict_layer(concat_emb)
        # # Lucas 240702 added for contrastive learning
        # concat_neg_emb = torch.cat((output_neg_gmf, output_neg_mlp, neg_u_emb, neg_i_emb), -1)
        # predict_neg_ratings = self.predict_layer(concat_neg_emb)

        # # Lucas 240702 added contrastive learning
        # cl_pos_emb = self.contrastive_learn(concat_emb)
        # cl_neg_emb = self.contrastive_learn(concat_neg_emb)
        # cl_neg_emb = cl_neg_emb.view(user.shape[0], self.num_neg_i, -1)

        # ******************************************Loss
        # # Calculate InfoNCE loss ---contrastive learning loss
        # contrastive_loss = info_nce_loss(cl_pos_emb, cl_pos_emb, cl_neg_emb, self.temperature)
        # print(f"Lucas--contrastive_loss: {contrastive_loss};")

        rating_loss = self.rating_loss(predict_ratings.view(-1), rating)
        print(f"Lucas--predict_ratings: {predict_ratings}; original ratings: {rating}; rating loss: {rating_loss}")
        # print(f"Lucas--rating_loss: {rating_loss}")
        # return_loss_dict["scores"] = prediction.view(-1)

        # # Lucas 240702 added for contrastive learning
        # neg_rating_loss = self.rating_loss(predict_neg_ratings.view(-1), neg_rating)
        # print(f"Lucas--neg predict_ratings: {predict_neg_ratings}; original neg ratings: {neg_rating}; neg rating loss: {neg_rating_loss}")

        # GRL classification loss
        # pred_domains_loss = self.exp_loss_fn(predict_domain, domain)
        # print(f"Lucas--predict_domains: {predict_domain}; original domains: {domain}; prefix pre domain loss: {pred_domains_loss}")
        pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain}; ncf pre domain loss: {pred_ncf_domain_loss}")
        # # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        print(f"Lucas--rating loss: {rating_loss}; domain prediction loss: {pred_ncf_domain_loss}")
        # print(f"Lucas--rating loss device: {rating_loss.device}; domain loss dtype: {pred_ncf_domain_loss.device}; predict_ncf_domain loss: {pred_ncf_domain_loss.device}")

        total_loss = self.lambda1*rating_loss + self.lambda2*pred_ncf_domain_loss
        print(f"Lucas--final total_loss: {total_loss}")
        loss_detail = [rating_loss, pred_ncf_domain_loss]
        return total_loss, loss_detail, predict_ratings.view(-1)


# Lucas 240805 add test NCF-GAN with nn.LayerNorm
# NCF + GAN-->
class NCF_GAN2(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, lambda_grl=1.0, temperature=0.1, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_GAN2, self).__init__()
        self.device_map = device_map
        self.num_u = num_u
        self.num_i = num_i
        self.num_neg_i = num_neg_i    # added for contrastive learning
        self.temperature = temperature  # added for contrastive learning
        self.factor_num = factor_num
        self.ncf_layer_num = ncf_layer_num
        self.drop_out = drop_out

        # self.peft_config = args[0]
        # self.model_config = config
        self.emb_size = emb_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda_grl = lambda_grl
        print(f"Lucas module num_u: {self.num_u}; num_i: {self.num_i}; lambda1: {self.lambda1}; lambda2: {self.lambda2}; lambda3: {self.lambda3}")
        print(f"Lucas module factor_num: {self.factor_num}")

        # print(f"Lucas--lucas_module--PeftPromptLlama2_V4--load_pretrained--device_map: {self.device_map}")

        # lucas 240402 NCF module
        self.embed_user_GMF = nn.Embedding(self.num_u, self.factor_num)
        self.embed_item_GMF = nn.Embedding(self.num_i, self.factor_num)
        # lucas 230327 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        self.embed_user_MLP = nn.Embedding(
            self.num_u, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.num_i, self.factor_num * (2 ** (self.ncf_layer_num - 1)))

        # nn.Sigmoid()
        # layer1：256-128；layer2: 128-64; layer3: 64-32
        mlp_modules = []
        mlp_out_size = []
        for i in range(self.ncf_layer_num):
            input_size = self.factor_num * (2 ** (self.ncf_layer_num - i))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.Dropout(p=self.drop_out))
            mlp_modules.append(nn.LayerNorm(input_size // 2))
            mlp_out_size.append(input_size)
            mlp_out_size.append(input_size // 2)
            mlp_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*mlp_modules)

        # Lucas 240731 added for grl and cl
        self.emb_fusion = nn.Sequential(
            nn.Linear(self.factor_num + mlp_out_size[-1], self.factor_num + mlp_out_size[-1]),
            nn.Dropout(p=self.drop_out),
            nn.LayerNorm(self.factor_num + mlp_out_size[-1]),
            nn.Sigmoid(),
        )

        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.grl = GRL(self.lambda_grl)  # Lucas 240708 changed grl
        print(f"Lucas GRL mlp_out size: {mlp_out_size}")
        # self.domain_fc = nn.Linear(self.factor_num + mlp_out_size[-1], 2)
        self.domain_fc = nn.Sequential(
            # nn.Linear(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)), self.factor_num),
            nn.Linear(self.factor_num + mlp_out_size[-1], self.factor_num),
            nn.Dropout(p=self.drop_out),
            nn.LayerNorm(self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, self.factor_num),
            nn.Dropout(p=self.drop_out),
            nn.LayerNorm(self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, 2)
        )
        self.grl_activation = nn.Sigmoid()

        # # Lucas 240709 added for feature selection
        # self.feature_selector = FeatureSelector(self.factor_num + mlp_out_size[-1])

        self.concatenate_len = 2  # added user and item embedding
        self.user_embeddings = nn.Embedding(self.num_u, self.emb_size)#.to(device_map[''])
        self.item_embeddings = nn.Embedding(self.num_i, self.emb_size)#.to(device_map[''])

        # Lucas 240326 added grl for user/item embeddings
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
            # nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
            # nn.LayerNorm(self.emb_size),
        )
        self.user_semantic_classifier = Classifier(self.emb_size)  # 768, 10

        # Lucas 240326 added 最后预测层
        self.rating_loss = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss()

        # self.predict_emb = self.factor_num * 2 # + self.emb_size * 2
        self.predict_emb = self.factor_num + mlp_out_size[-1]
        print(f"Lucas predict_emb: {self.predict_emb}")
        # Lucas 240416 change the predict layer
        # self.predict_layer = nn.Linear(self.predict_emb, 1)
        dropout = 0.2
        self.activation = nn.Tanh()
        self.predict_layer = nn.Sequential(nn.Linear(self.predict_emb, self.predict_emb),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.LayerNorm(self.predict_emb),
                                           nn.Linear(self.predict_emb, self.factor_num),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.LayerNorm(self.factor_num),
                                           nn.Linear(self.factor_num, 1))

        # # Lucas 240701 added for Contrastive Learning
        # self.contrastive_learn = nn.Linear(self.predict_emb, int(self.predict_emb / 2))

        # ******************************lucas 240326 initialize the parameters of modules
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        # nn.init.xavier_uniform_(self.transform_matrix.weight)
        initrange = 0.1
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        for m in self.emb_fusion:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        for m in self.domain_fc:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        # for m in self.modules():
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         m.bias.data.zero_()

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        # Lucas 240416 change the prediction layer initialization
        # nn.init.kaiming_uniform_(self.predict_layer.weight,
        #                          a=1, nonlinearity='sigmoid')
        for layer in self.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

        print(f"Lucas finish initialize model---PrefixTune_NCF_GAN_Mixtral2")

    def forward(
            self,
            user: torch.LongTensor = None,
            item: torch.LongTensor = None,
            rating: torch.LongTensor = None,
            domain: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if rating is not None:
            rating = rating.to(torch.float32)
        if domain is not None:
            domain = domain.to(torch.long)
        # print(f"Lucas model forward-----user: {user};----item:{item};----rating: {rating};----domain: {domain}")

        embed_u_gmf = self.embed_user_GMF(user)
        embed_i_gmf = self.embed_item_GMF(item)
        output_gmf = embed_u_gmf * embed_i_gmf
        # # Lucas 240702 added for contrastive learning
        # user_extend = user.unsqueeze(1).repeat(1, self.num_neg_i).view(-1)
        # print(f"Lucas model forward-----extend user: {user_extend};")
        # embed_neg_u_gmf = self.embed_user_GMF(user_extend)
        # embed_neg_i_gmf = self.embed_item_GMF(neg_items)
        # output_neg_gmf = embed_neg_u_gmf * embed_neg_i_gmf

        embed_u_mlp = self.embed_user_MLP(user)
        embed_i_mlp = self.embed_item_MLP(item)
        interaction = torch.cat((embed_u_mlp, embed_i_mlp), -1)
        output_mlp = self.MLP_layers(interaction)
        # # Lucas 240702 added for contrastive learning
        # embed_neg_u_mlp = self.embed_user_MLP(user_extend)
        # embed_neg_i_mlp = self.embed_item_MLP(neg_items)
        # neg_inter = torch.cat((embed_neg_u_mlp, embed_neg_i_mlp), -1)
        # output_neg_mlp = self.MLP_layers(neg_inter)

        # lucas 240402 added GRL layer for user embeddings in NCF
        # emb_u_ncf = torch.cat((embed_u_gmf, embed_u_mlp), -1)
        # u_ncf_dom_label = self.user_NCF_classifier(emb_u_ncf, self.p)
        # Lucas 240708 changed GRL input layer
        concat_emb = torch.cat((output_gmf, output_mlp), -1)
        concat_emb = self.emb_fusion(concat_emb)    # lucas 240731added for feature selection

        # Lucas 240709 added feature selection layer, 效果没有提升
        # selected_concat_emb = self.feature_selector(concat_emb)

        # Domain classification task
        concat_emb_grl = self.grl(concat_emb)
        u_ncf_dom_label = self.grl_activation(self.domain_fc(concat_emb_grl))

        # # Lucas 240702 added for contrastive learning
        # emb_neg_u_ncf = torch.cat((embed_neg_u_gmf, embed_neg_u_mlp), -1)
        # neg_u_ncf_dom_label = self.user_NCF_classifier(emb_neg_u_ncf, self.p)

        # user_emb = self.user_embeddings(user)#.to(self.device_map[''])  # (batch_size, emsize)
        # item_emb = self.item_embeddings(item)#.to(self.device_map[''])  # (batch_size, emsize)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.user_embeddings(user_extend)
        # neg_i_emb = self.item_embeddings(neg_items)

        # lucas 240326 added for tune user/item embedding
        # user_emb = self.hidden_layers(user_emb)
        # item_emb = self.hidden_layers(item_emb)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.hidden_layers(neg_u_emb)
        # neg_i_emb = self.hidden_layers(neg_i_emb)
        # 230326 通过GRLayer GAN生成器判断source和target类别
        # p = float(batch_int + (epoch - dann_epoch) * num_batch / (args.epochs - dann_epoch) / num_batch)
        # p = 2. / (1. + np.exp(-10 * p)) - 1
        # predict_domain = self.user_semantic_classifier(user_emb, self.p)  # classifier，输入40维度，输出10维度,经过GRL再经过线性层+sigmoid输出2维分类
        # # Lucas 240702 added for contrastive learning
        # predict_neg_domain = self.user_semantic_classifier(neg_u_emb, self.p)

        # *****************************************training
        # Lucas 240327 added for rating prediction
        print(f"Lucas GMF size: {output_gmf.size()}; MLP size: {output_mlp.size()};")
        # concat_emb = torch.cat((output_gmf, output_mlp), -1)
        predict_ratings = self.predict_layer(concat_emb)
        # # Lucas 240702 added for contrastive learning
        # concat_neg_emb = torch.cat((output_neg_gmf, output_neg_mlp, neg_u_emb, neg_i_emb), -1)
        # predict_neg_ratings = self.predict_layer(concat_neg_emb)

        # # Lucas 240702 added contrastive learning
        # cl_pos_emb = self.contrastive_learn(concat_emb)
        # cl_neg_emb = self.contrastive_learn(concat_neg_emb)
        # cl_neg_emb = cl_neg_emb.view(user.shape[0], self.num_neg_i, -1)

        # ******************************************Loss
        # # Calculate InfoNCE loss ---contrastive learning loss
        # contrastive_loss = info_nce_loss(cl_pos_emb, cl_pos_emb, cl_neg_emb, self.temperature)
        # print(f"Lucas--contrastive_loss: {contrastive_loss};")

        rating_loss = self.rating_loss(predict_ratings.view(-1), rating)
        print(f"Lucas--predict_ratings: {predict_ratings}; original ratings: {rating}; rating loss: {rating_loss}")
        # print(f"Lucas--rating_loss: {rating_loss}")
        # return_loss_dict["scores"] = prediction.view(-1)

        # # Lucas 240702 added for contrastive learning
        # neg_rating_loss = self.rating_loss(predict_neg_ratings.view(-1), neg_rating)
        # print(f"Lucas--neg predict_ratings: {predict_neg_ratings}; original neg ratings: {neg_rating}; neg rating loss: {neg_rating_loss}")

        # GRL classification loss
        # pred_domains_loss = self.exp_loss_fn(predict_domain, domain)
        # print(f"Lucas--predict_domains: {predict_domain}; original domains: {domain}; prefix pre domain loss: {pred_domains_loss}")
        pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain}; ncf pre domain loss: {pred_ncf_domain_loss}")
        # # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        print(f"Lucas--rating loss: {rating_loss}; domain prediction loss: {pred_ncf_domain_loss}")
        # print(f"Lucas--rating loss device: {rating_loss.device}; domain loss dtype: {pred_ncf_domain_loss.device}; predict_ncf_domain loss: {pred_ncf_domain_loss.device}")

        total_loss = self.lambda1*rating_loss + self.lambda2*pred_ncf_domain_loss
        print(f"Lucas--final total_loss: {total_loss}")
        loss_detail = [rating_loss, pred_ncf_domain_loss]
        return total_loss, loss_detail, predict_ratings.view(-1)

# Lucas 240705 add NCF-CL for ablation study and code test
# NCF + Contrastive learning
class NCF_CL(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, temperature=0.1, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_CL, self).__init__()
        self.device_map = device_map
        self.num_u = num_u
        self.num_i = num_i
        self.num_neg_i = num_neg_i    # added for contrastive learning
        self.temperature = temperature  # added for contrastive learning
        self.factor_num = factor_num
        self.ncf_layer_num = ncf_layer_num
        self.drop_out = drop_out
        self.p = 0.6

        # self.peft_config = args[0]
        # self.model_config = config
        self.emb_size = emb_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        print(f"Lucas module num_u: {self.num_u}; num_i: {self.num_i}; lambda1: {self.lambda1}; lambda2: {self.lambda2}; lambda3: {self.lambda3}")
        print(f"Lucas module factor_num: {self.factor_num}")

        # print(f"Lucas--lucas_module--PeftPromptLlama2_V4--load_pretrained--device_map: {self.device_map}")

        # lucas 240402 NCF module
        self.embed_user_GMF = nn.Embedding(self.num_u, self.factor_num)
        self.embed_item_GMF = nn.Embedding(self.num_i, self.factor_num)
        # lucas 230327 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        self.embed_user_MLP = nn.Embedding(
            self.num_u, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.num_i, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # layer1：256-128；layer2: 128-64; layer3: 64-32
        MLP_modules = []
        for i in range(self.ncf_layer_num):
            input_size = self.factor_num * (2 ** (self.ncf_layer_num - i))
            MLP_modules.append(nn.Dropout(p=self.drop_out))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)


        self.concatenate_len = 2  # added user and item embedding
        self.user_embeddings = nn.Embedding(self.num_u, self.emb_size)#.to(device_map[''])
        self.item_embeddings = nn.Embedding(self.num_i, self.emb_size)#.to(device_map[''])

        # Lucas 240326 added grl for user/item embeddings
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
        )
        # self.user_semantic_classifier = Classifier(self.emb_size)  # 768, 10

        # Lucas 240326 added 最后预测层
        self.rating_loss = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss()

        self.predict_emb = self.factor_num * 2 # + self.emb_size * 2
        print(f"Lucas predict_emb: {self.predict_emb}")
        # Lucas 240416 change the predict layer
        # self.predict_layer = nn.Linear(self.predict_emb, 1)
        dropout = 0.2
        self.activation = nn.Tanh()
        self.predict_layer = nn.Sequential(nn.Linear(self.predict_emb, self.predict_emb),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(self.predict_emb, 512),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(512, 1))

        # Lucas 240701 added for Contrastive Learning
        self.contrastive_learn = nn.Linear(self.predict_emb, int(self.predict_emb / 2))

        # ******************************lucas 240326 initialize the parameters of modules
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        # nn.init.xavier_uniform_(self.transform_matrix.weight)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         m.bias.data.zero_()

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        # Lucas 240416 change the prediction layer initialization
        # nn.init.kaiming_uniform_(self.predict_layer.weight,
        #                          a=1, nonlinearity='sigmoid')
        for layer in self.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

        print(f"Lucas finish initialize model---PrefixTune_NCF_GAN_Mixtral2")

    def forward(
            self,
            user: torch.LongTensor = None,
            item: torch.LongTensor = None,
            neg_items: torch.LongTensor = None,
            rating: torch.LongTensor = None,
            neg_rating: torch.LongTensor = None,
            # domain: torch.LongTensor = None,
            # neg_domain: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if rating is not None:
            rating = rating.to(torch.float32)
        if neg_rating is not None:
            neg_rating = neg_rating.to(torch.float32)
        # if domain is not None:
        #     domain = domain.to(torch.long)
        # if neg_domain is not None:
        #     neg_domain = neg_domain.to(torch.long)
        print(f"Lucas model forward-----user: {user};----item:{item};----rating: {rating};")
        print(f"Lucas model forward-----neg_items: {neg_items};----neg_rating:{neg_rating};")

        embed_u_gmf = self.embed_user_GMF(user)
        embed_i_gmf = self.embed_item_GMF(item)
        output_gmf = embed_u_gmf * embed_i_gmf
        # Lucas 240702 added for contrastive learning
        user_extend = user.unsqueeze(1).repeat(1, self.num_neg_i).view(-1)
        print(f"Lucas model forward-----extend user: {user_extend};")
        embed_neg_u_gmf = self.embed_user_GMF(user_extend)
        embed_neg_i_gmf = self.embed_item_GMF(neg_items)
        output_neg_gmf = embed_neg_u_gmf * embed_neg_i_gmf

        embed_u_mlp = self.embed_user_MLP(user)
        embed_i_mlp = self.embed_item_MLP(item)
        interaction = torch.cat((embed_u_mlp, embed_i_mlp), -1)
        output_mlp = self.MLP_layers(interaction)
        # Lucas 240702 added for contrastive learning
        embed_neg_u_mlp = self.embed_user_MLP(user_extend)
        embed_neg_i_mlp = self.embed_item_MLP(neg_items)
        neg_inter = torch.cat((embed_neg_u_mlp, embed_neg_i_mlp), -1)
        output_neg_mlp = self.MLP_layers(neg_inter)

        # lucas 240402 added GRL layer for user embeddings in NCF
        # emb_u_ncf = torch.cat((embed_u_gmf, embed_u_mlp), -1)
        # u_ncf_dom_label = self.user_NCF_classifier(emb_u_ncf, self.p)
        # Lucas 240702 added for contrastive learning
        # emb_neg_u_ncf = torch.cat((embed_neg_u_gmf, embed_neg_u_mlp), -1)
        # neg_u_ncf_dom_label = self.user_NCF_classifier(emb_neg_u_ncf, self.p)

        # user_emb = self.user_embeddings(user)#.to(self.device_map[''])  # (batch_size, emsize)
        # item_emb = self.item_embeddings(item)#.to(self.device_map[''])  # (batch_size, emsize)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.user_embeddings(user_extend)
        # neg_i_emb = self.item_embeddings(neg_items)
        #
        # # lucas 240326 added for tune user/item embedding
        # user_emb = self.hidden_layers(user_emb)
        # item_emb = self.hidden_layers(item_emb)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.hidden_layers(neg_u_emb)
        # neg_i_emb = self.hidden_layers(neg_i_emb)
        # 230326 通过GRLayer GAN生成器判断source和target类别
        # p = float(batch_int + (epoch - dann_epoch) * num_batch / (args.epochs - dann_epoch) / num_batch)
        # p = 2. / (1. + np.exp(-10 * p)) - 1
        # predict_domain = self.user_semantic_classifier(user_emb, self.p)  # classifier，输入40维度，输出10维度,经过GRL再经过线性层+sigmoid输出2维分类
        # Lucas 240702 added for contrastive learning
        # predict_neg_domain = self.user_semantic_classifier(neg_u_emb, self.p)

        # *****************************************training
        # Lucas 240327 added for rating prediction
        print(f"Lucas GMF size: {output_gmf.size()}; MLP size: {output_mlp.size()};")
        concat_emb = torch.cat((output_gmf, output_mlp), -1)
        predict_ratings = self.predict_layer(concat_emb)
        # Lucas 240702 added for contrastive learning
        concat_neg_emb = torch.cat((output_neg_gmf, output_neg_mlp), -1)
        predict_neg_ratings = self.predict_layer(concat_neg_emb)

        # Lucas 240702 added contrastive learning
        cl_pos_emb = self.contrastive_learn(concat_emb)
        cl_neg_emb = self.contrastive_learn(concat_neg_emb)
        cl_neg_emb = cl_neg_emb.view(user.shape[0], self.num_neg_i, -1)

        # ******************************************Loss
        # Calculate InfoNCE loss ---contrastive learning loss
        contrastive_loss = info_nce_loss(cl_pos_emb, cl_pos_emb, cl_neg_emb, self.temperature)
        print(f"Lucas--contrastive_loss: {contrastive_loss};")

        rating_loss = self.rating_loss(predict_ratings.view(-1), rating)
        print(f"Lucas--predict_ratings: {predict_ratings}; original ratings: {rating}; rating loss: {rating_loss}")
        # print(f"Lucas--rating_loss: {rating_loss}")
        # return_loss_dict["scores"] = prediction.view(-1)

        # Lucas 240702 added for contrastive learning
        neg_rating_loss = self.rating_loss(predict_neg_ratings.view(-1), neg_rating)
        print(f"Lucas--neg predict_ratings: {predict_neg_ratings}; original neg ratings: {neg_rating}; neg rating loss: {neg_rating_loss}")

        # GRL classification loss
        # pred_domains_loss = self.exp_loss_fn(predict_domain, domain)
        # print(f"Lucas--predict_domains: {predict_domain}; original domains: {domain}; prefix pre domain loss: {pred_domains_loss}")
        # pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        # print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain}; ncf pre domain loss: {pred_ncf_domain_loss}")
        # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        print(f"Lucas--rating loss: {rating_loss};")
        # print(f"Lucas--rating loss device: {rating_loss.device};")

        total_loss = self.lambda1*rating_loss + self.lambda2*contrastive_loss # + self.lambda1*neg_rating_loss
        print(f"Lucas--final total_loss: {total_loss}")
        loss_detail = [rating_loss, neg_rating_loss, contrastive_loss]
        return total_loss, loss_detail, predict_ratings.view(-1)


# Lucas 240705 add NCF for ablation study and code test
# NCF
class NCF(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, temperature=0.1, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF, self).__init__()
        self.device_map = device_map
        self.num_u = num_u
        self.num_i = num_i
        self.num_neg_i = num_neg_i    # added for contrastive learning
        self.temperature = temperature  # added for contrastive learning
        self.factor_num = factor_num
        self.ncf_layer_num = ncf_layer_num
        self.drop_out = drop_out
        self.p = 0.6

        # self.peft_config = args[0]
        # self.model_config = config
        self.emb_size = emb_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        print(f"Lucas module num_u: {self.num_u}; num_i: {self.num_i}; lambda1: {self.lambda1}; lambda2: {self.lambda2}; lambda3: {self.lambda3}")
        print(f"Lucas module factor_num: {self.factor_num}")

        print(f"Lucas--lucas_module--PeftPromptLlama2_V4--load_pretrained--device_map: {self.device_map}")

        # lucas 240402 NCF module
        self.embed_user_GMF = nn.Embedding(self.num_u, self.factor_num)
        self.embed_item_GMF = nn.Embedding(self.num_i, self.factor_num)
        # lucas 230327 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        self.embed_user_MLP = nn.Embedding(
            self.num_u, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.num_i, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # layer1：256-128；layer2: 128-64; layer3: 64-32
        MLP_modules = []
        for i in range(self.ncf_layer_num):
            input_size = self.factor_num * (2 ** (self.ncf_layer_num - i))
            MLP_modules.append(nn.Dropout(p=self.drop_out))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)


        self.concatenate_len = 2  # added user and item embedding
        self.user_embeddings = nn.Embedding(self.num_u, self.emb_size)#.to(device_map[''])
        self.item_embeddings = nn.Embedding(self.num_i, self.emb_size)#.to(device_map[''])

        # Lucas 240326 added grl for user/item embeddings
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
        )
        # self.user_semantic_classifier = Classifier(self.emb_size)  # 768, 10

        # Lucas 240326 added 最后预测层
        self.rating_loss = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss()

        self.predict_emb = self.factor_num * 2 # + self.emb_size * 2
        print(f"Lucas predict_emb: {self.predict_emb}")
        # Lucas 240416 change the predict layer
        # self.predict_layer = nn.Linear(self.predict_emb, 1)
        dropout = 0.2
        self.activation = nn.Tanh()
        self.predict_layer = nn.Sequential(nn.Linear(self.predict_emb, self.predict_emb),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(self.predict_emb, 512),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(512, 1))

        # Lucas 240701 added for Contrastive Learning
        # self.contrastive_learn = nn.Linear(self.predict_emb, int(self.predict_emb / 2))

        # ******************************lucas 240326 initialize the parameters of modules
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        # nn.init.xavier_uniform_(self.transform_matrix.weight)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         m.bias.data.zero_()

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        # Lucas 240416 change the prediction layer initialization
        # nn.init.kaiming_uniform_(self.predict_layer.weight,
        #                          a=1, nonlinearity='sigmoid')
        for layer in self.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

        print(f"Lucas finish initialize model---PrefixTune_NCF_GAN_Mixtral2")

    def forward(
            self,
            user: torch.LongTensor = None,
            item: torch.LongTensor = None,
            rating: torch.LongTensor = None,
            domain: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if rating is not None:
            rating = rating.to(torch.float32)
        if domain is not None:
            domain = domain.to(torch.long)
        print(f"Lucas model forward-----user: {user};----item:{item};----rating: {rating};----domain: {domain}")

        embed_u_gmf = self.embed_user_GMF(user)
        embed_i_gmf = self.embed_item_GMF(item)
        output_gmf = embed_u_gmf * embed_i_gmf
        # # Lucas 240702 added for contrastive learning
        # user_extend = user.unsqueeze(1).repeat(1, self.num_neg_i).view(-1)
        # print(f"Lucas model forward-----extend user: {user_extend};")
        # embed_neg_u_gmf = self.embed_user_GMF(user_extend)
        # embed_neg_i_gmf = self.embed_item_GMF(neg_items)
        # output_neg_gmf = embed_neg_u_gmf * embed_neg_i_gmf

        embed_u_mlp = self.embed_user_MLP(user)
        embed_i_mlp = self.embed_item_MLP(item)
        interaction = torch.cat((embed_u_mlp, embed_i_mlp), -1)
        output_mlp = self.MLP_layers(interaction)
        # # Lucas 240702 added for contrastive learning
        # embed_neg_u_mlp = self.embed_user_MLP(user_extend)
        # embed_neg_i_mlp = self.embed_item_MLP(neg_items)
        # neg_inter = torch.cat((embed_neg_u_mlp, embed_neg_i_mlp), -1)
        # output_neg_mlp = self.MLP_layers(neg_inter)

        # lucas 240402 added GRL layer for user embeddings in NCF
        emb_u_ncf = torch.cat((embed_u_gmf, embed_u_mlp), -1)
        # u_ncf_dom_label = self.user_NCF_classifier(emb_u_ncf, self.p)
        # # Lucas 240702 added for contrastive learning
        # emb_neg_u_ncf = torch.cat((embed_neg_u_gmf, embed_neg_u_mlp), -1)
        # neg_u_ncf_dom_label = self.user_NCF_classifier(emb_neg_u_ncf, self.p)

        # user_emb = self.user_embeddings(user)#.to(self.device_map[''])  # (batch_size, emsize)
        # item_emb = self.item_embeddings(item)#.to(self.device_map[''])  # (batch_size, emsize)
        # # # Lucas 240702 added for contrastive learning
        # # neg_u_emb = self.user_embeddings(user_extend)
        # # neg_i_emb = self.item_embeddings(neg_items)
        #
        # # lucas 240326 added for tune user/item embedding
        # user_emb = self.hidden_layers(user_emb)
        # item_emb = self.hidden_layers(item_emb)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.hidden_layers(neg_u_emb)
        # neg_i_emb = self.hidden_layers(neg_i_emb)
        # 230326 通过GRLayer GAN生成器判断source和target类别
        # p = float(batch_int + (epoch - dann_epoch) * num_batch / (args.epochs - dann_epoch) / num_batch)
        # p = 2. / (1. + np.exp(-10 * p)) - 1
        # predict_domain = self.user_semantic_classifier(user_emb, self.p)  # classifier，输入40维度，输出10维度,经过GRL再经过线性层+sigmoid输出2维分类
        # # Lucas 240702 added for contrastive learning
        # predict_neg_domain = self.user_semantic_classifier(neg_u_emb, self.p)

        # *****************************************training
        # Lucas 240327 added for rating prediction
        print(f"Lucas GMF size: {output_gmf.size()}; MLP size: {output_mlp.size()};")
        concat_emb = torch.cat((output_gmf, output_mlp), -1)
        predict_ratings = self.predict_layer(concat_emb)
        # # Lucas 240702 added for contrastive learning
        # concat_neg_emb = torch.cat((output_neg_gmf, output_neg_mlp, neg_u_emb, neg_i_emb), -1)
        # predict_neg_ratings = self.predict_layer(concat_neg_emb)

        # # Lucas 240702 added contrastive learning
        # cl_pos_emb = self.contrastive_learn(concat_emb)
        # cl_neg_emb = self.contrastive_learn(concat_neg_emb)
        # cl_neg_emb = cl_neg_emb.view(user.shape[0], self.num_neg_i, -1)

        # ******************************************Loss
        # Calculate InfoNCE loss ---contrastive learning loss
        # contrastive_loss = info_nce_loss(cl_pos_emb, cl_pos_emb, cl_neg_emb, self.temperature)
        # print(f"Lucas--contrastive_loss: {contrastive_loss};")

        rating_loss = self.rating_loss(predict_ratings.view(-1), rating)
        print(f"Lucas--predict_ratings: {predict_ratings}; original ratings: {rating}; rating loss: {rating_loss}")
        # print(f"Lucas--rating_loss: {rating_loss}")
        # return_loss_dict["scores"] = prediction.view(-1)

        # # Lucas 240702 added for contrastive learning
        # neg_rating_loss = self.rating_loss(predict_neg_ratings.view(-1), neg_rating)
        # print(f"Lucas--neg predict_ratings: {predict_neg_ratings}; original neg ratings: {neg_rating}; neg rating loss: {neg_rating_loss}")

        # GRL classification loss
        # pred_domains_loss = self.exp_loss_fn(predict_domain, domain)
        # print(f"Lucas--predict_domains: {predict_domain}; original domains: {domain}; prefix pre domain loss: {pred_domains_loss}")
        # pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        # print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain}; ncf pre domain loss: {pred_ncf_domain_loss}")
        # # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        # print(f"Lucas--rating loss: {rating_loss}; domain prediction loss: {pred_domains_loss}")
        # print(f"Lucas--rating loss device: {rating_loss.device}; domain loss dtype: {pred_domains_loss.device}; predict_ncf_domain loss: {pred_ncf_domain_loss.device}")

        total_loss = self.lambda1*rating_loss
        print(f"Lucas--final total_loss: {total_loss}")
        # loss_detail = [rating_loss, pred_domains_loss, pred_neg_domains_loss, pred_ncf_domain_loss, pred_neg_ncf_domain_loss, contrastive_loss]
        return total_loss, predict_ratings.view(-1)

# Lucas 240718 added for testing dynamic control weight of contrastive learning
def curriculum_weight(epoch, max_epoch, max_weight, strategy='linear'):
    if strategy == 'linear':
        return max_weight * min(epoch / max_epoch, 1.0)
    elif strategy == 'exp':
        return max_weight * (1 - np.exp(-5 * epoch / max_epoch))
    elif strategy == 'step':
        return max_weight * min(epoch // (max_epoch // 3) * 0.5, 1.0)
    else:
        raise ValueError("Unknown curriculum strategy")


# Lucas 240716 test NCF_Contrastive_MemoryBank for contrastive learning
class NCF_Contrastive_MemoryBank(nn.Module):
    def __init__(self, num_users, num_items, num_neg_i, layers, embedding_dim=128, queue_size=4096, momentum=0.9, device_map=None,
                 factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0,
                 lambda4=1.0, temperature=0.1, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_Contrastive_MemoryBank, self).__init__()
        self.device_map = device_map
        self.num_u = num_users
        self.num_i = num_items
        self.num_neg_i = num_neg_i  # added for contrastive learning
        self.temperature = temperature  # added for contrastive learning
        self.factor_num = factor_num
        self.ncf_layer_num = ncf_layer_num
        self.drop_out = drop_out
        self.p = 0.6

        # self.peft_config = args[0]
        # self.model_config = config
        self.emb_size = embedding_dim
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        print(
            f"Lucas module num_u: {self.num_u}; num_i: {self.num_i}; lambda1: {self.lambda1}; lambda2: {self.lambda2}; lambda3: {self.lambda3}")
        print(f"Lucas module factor_num: {self.factor_num}")

        # print(f"Lucas--lucas_module--PeftPromptLlama2_V4--load_pretrained--device_map: {self.device_map}")

        # lucas 240402 NCF module
        self.embed_user_GMF = nn.Embedding(self.num_u, self.factor_num)
        self.embed_item_GMF = nn.Embedding(self.num_i, self.factor_num)
        # lucas 230327 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        self.embed_user_MLP = nn.Embedding(
            self.num_u, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.num_i, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # layer1：256-128；layer2: 128-64; layer3: 64-32
        MLP_modules = []
        for i in range(self.ncf_layer_num):
            input_size = self.factor_num * (2 ** (self.ncf_layer_num - i))
            MLP_modules.append(nn.Dropout(p=self.drop_out))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.concatenate_len = 2  # added user and item embedding
        self.user_embeddings = nn.Embedding(self.num_u, self.emb_size)  # .to(device_map[''])
        self.item_embeddings = nn.Embedding(self.num_i, self.emb_size)  # .to(device_map[''])

        # Lucas 240326 added grl for user/item embeddings
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
        )
        # self.user_semantic_classifier = Classifier(self.emb_size)  # 768, 10

        # Lucas 240326 added 最后预测层
        self.rating_loss = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss()

        self.predict_emb = self.factor_num * 2  # + self.emb_size * 2
        print(f"Lucas predict_emb: {self.predict_emb}")
        # Lucas 240416 change the predict layer
        # self.predict_layer = nn.Linear(self.predict_emb, 1)
        dropout = 0.2
        self.activation = nn.Tanh()
        self.predict_layer = nn.Sequential(nn.Linear(self.predict_emb, self.predict_emb),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(self.predict_emb, 512),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(512, 1))

        # Lucas 240701 added for Contrastive Learning
        self.contrastive_learn = nn.Linear(self.predict_emb, int(self.predict_emb / 2))

        # Lucas 240717 added for testing Contrastive_MemoryBank
        self.projection = nn.Sequential(
            nn.Linear(layers[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.queue_size = queue_size
        self.momentum = momentum
        self.register_buffer("queue", torch.randn(queue_size, 64))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.temperature = nn.Parameter(torch.FloatTensor([0.07]))

        # ******************************lucas 240326 initialize the parameters of modules
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        # nn.init.xavier_uniform_(self.transform_matrix.weight)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        # Lucas 240416 change the prediction layer initialization
        # nn.init.kaiming_uniform_(self.predict_layer.weight,
        #                          a=1, nonlinearity='sigmoid')
        for layer in self.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

        print(f"Lucas finish initialize model---PrefixTune_NCF_GAN_Mixtral2")

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.queue_size:
            self.queue[ptr:] = keys[:self.queue_size - ptr]
            self.queue[:batch_size - (self.queue_size - ptr)] = keys[self.queue_size - ptr:]
        else:
            self.queue[ptr:ptr + batch_size] = keys

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def contrastive_loss(self, q):
        # Positive logits: Nx1 (self-contrast)
        l_pos = torch.einsum('nc,nc->n', [q, q]).unsqueeze(-1)
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.t()])

        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature
        logits /= self.temperature

        # Labels: positives are the 0-th
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)

    def forward(
            self,
            user: torch.LongTensor = None,
            item: torch.LongTensor = None,
            neg_items: torch.LongTensor = None,
            rating: torch.LongTensor = None,
            neg_rating: torch.LongTensor = None,
            # domain: torch.LongTensor = None,
            # neg_domain: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if rating is not None:
            rating = rating.to(torch.float32)
        if neg_rating is not None:
            neg_rating = neg_rating.to(torch.float32)
        # if domain is not None:
        #     domain = domain.to(torch.long)
        # if neg_domain is not None:
        #     neg_domain = neg_domain.to(torch.long)
        print(f"Lucas model forward-----user: {user};----item:{item};----rating: {rating};")
        print(f"Lucas model forward-----neg_items: {neg_items};----neg_rating:{neg_rating};")

        embed_u_gmf = self.embed_user_GMF(user)
        embed_i_gmf = self.embed_item_GMF(item)
        output_gmf = embed_u_gmf * embed_i_gmf
        # Lucas 240702 added for contrastive learning
        user_extend = user.unsqueeze(1).repeat(1, self.num_neg_i).view(-1)
        print(f"Lucas model forward-----extend user: {user_extend};")
        embed_neg_u_gmf = self.embed_user_GMF(user_extend)
        embed_neg_i_gmf = self.embed_item_GMF(neg_items)
        output_neg_gmf = embed_neg_u_gmf * embed_neg_i_gmf

        embed_u_mlp = self.embed_user_MLP(user)
        embed_i_mlp = self.embed_item_MLP(item)
        interaction = torch.cat((embed_u_mlp, embed_i_mlp), -1)
        output_mlp = self.MLP_layers(interaction)
        # Lucas 240702 added for contrastive learning
        embed_neg_u_mlp = self.embed_user_MLP(user_extend)
        embed_neg_i_mlp = self.embed_item_MLP(neg_items)
        neg_inter = torch.cat((embed_neg_u_mlp, embed_neg_i_mlp), -1)
        output_neg_mlp = self.MLP_layers(neg_inter)

        print(f"Lucas GMF size: {output_gmf.size()}; MLP size: {output_mlp.size()};")
        concat_emb = torch.cat((output_gmf, output_mlp), -1)
        predict_ratings = self.predict_layer(concat_emb)
        # Lucas 240702 added for contrastive learning
        concat_neg_emb = torch.cat((output_neg_gmf, output_neg_mlp), -1)
        predict_neg_ratings = self.predict_layer(concat_neg_emb)

        # Lucas 240702 added contrastive learning
        cl_pos_emb = self.contrastive_learn(concat_emb)
        cl_neg_emb = self.contrastive_learn(concat_neg_emb)
        cl_neg_emb = cl_neg_emb.view(user.shape[0], self.num_neg_i, -1)

        # Lucas 240717 try contrastive_Memory Bank method
        q_concat_emb = self.projection(concat_emb)
        q_concat_emb = F.normalize(q_concat_emb, dim=1)

        # ******************************************Loss
        # Lucas 240717 try contrastive_Memory Bank method, not use original contrastive learning
        # # Calculate InfoNCE loss ---contrastive learning loss
        # contrastive_loss = info_nce_loss(cl_pos_emb, cl_pos_emb, cl_neg_emb, self.temperature)
        # print(f"Lucas--contrastive_loss: {contrastive_loss};")

        # Lucas 240717 try contrastive_Memory Bank method Contrastive loss
        contrastive_loss = self.contrastive_loss(q_concat_emb)

        rating_loss = self.rating_loss(predict_ratings.view(-1), rating)
        print(f"Lucas--predict_ratings: {predict_ratings}; original ratings: {rating}; rating loss: {rating_loss}")
        # print(f"Lucas--rating_loss: {rating_loss}")
        # return_loss_dict["scores"] = prediction.view(-1)

        # Lucas 240702 added for contrastive learning
        neg_rating_loss = self.rating_loss(predict_neg_ratings.view(-1), neg_rating)
        print(
            f"Lucas--neg predict_ratings: {predict_neg_ratings}; original neg ratings: {neg_rating}; neg rating loss: {neg_rating_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        print(f"Lucas--rating loss: {rating_loss};")
        # print(f"Lucas--rating loss device: {rating_loss.device};")
        print(f"Lucas--final contrastive_loss: {contrastive_loss}")
        total_loss = self.lambda1 * rating_loss + self.lambda2 * contrastive_loss  # + self.lambda1*neg_rating_loss
        print(f"Lucas--final total_loss: {total_loss}")
        loss_detail = [rating_loss, neg_rating_loss, contrastive_loss]
        return total_loss, loss_detail, predict_ratings.view(-1)


# Lucas 240716 test NCF_Contrastive_MemoryBank for contrastive learning with Curriculum Learning
class NCF_ContLea_MemoryBank_CurriLea(nn.Module):
    def __init__(self, num_users, num_items, num_neg_i, layers, embedding_dim=128, queue_size=4096, momentum=0.9, device_map=None,
                 factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0,
                 lambda4=1.0, temperature=0.1, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_ContLea_MemoryBank_CurriLea, self).__init__()
        self.device_map = device_map
        self.num_u = num_users
        self.num_i = num_items
        self.num_neg_i = num_neg_i  # added for contrastive learning
        self.temperature = temperature  # added for contrastive learning
        self.factor_num = factor_num
        self.ncf_layer_num = ncf_layer_num
        self.drop_out = drop_out
        self.p = 0.6

        # self.peft_config = args[0]
        # self.model_config = config
        self.emb_size = embedding_dim
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        print(
            f"Lucas module num_u: {self.num_u}; num_i: {self.num_i}; lambda1: {self.lambda1}; lambda2: {self.lambda2}; lambda3: {self.lambda3}")
        print(f"Lucas module factor_num: {self.factor_num}")

        # print(f"Lucas--lucas_module--PeftPromptLlama2_V4--load_pretrained--device_map: {self.device_map}")

        # lucas 240402 NCF module
        self.embed_user_GMF = nn.Embedding(self.num_u, self.factor_num)
        self.embed_item_GMF = nn.Embedding(self.num_i, self.factor_num)
        # lucas 230327 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        self.embed_user_MLP = nn.Embedding(
            self.num_u, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.num_i, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # layer1：256-128；layer2: 128-64; layer3: 64-32
        MLP_modules = []
        for i in range(self.ncf_layer_num):
            input_size = self.factor_num * (2 ** (self.ncf_layer_num - i))
            MLP_modules.append(nn.Dropout(p=self.drop_out))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.concatenate_len = 2  # added user and item embedding
        self.user_embeddings = nn.Embedding(self.num_u, self.emb_size)  # .to(device_map[''])
        self.item_embeddings = nn.Embedding(self.num_i, self.emb_size)  # .to(device_map[''])

        # Lucas 240326 added grl for user/item embeddings
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
        )
        # self.user_semantic_classifier = Classifier(self.emb_size)  # 768, 10

        # Lucas 240326 added 最后预测层
        self.rating_loss = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss()

        self.predict_emb = self.factor_num * 2  # + self.emb_size * 2
        print(f"Lucas predict_emb: {self.predict_emb}")
        # Lucas 240416 change the predict layer
        # self.predict_layer = nn.Linear(self.predict_emb, 1)
        dropout = 0.2
        self.activation = nn.Tanh()
        self.predict_layer = nn.Sequential(nn.Linear(self.predict_emb, self.predict_emb),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(self.predict_emb, 512),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(512, 1))

        # Lucas 240701 added for Contrastive Learning
        self.contrastive_learn = nn.Linear(self.predict_emb, int(self.predict_emb / 2))

        # Lucas 240717 added for testing Contrastive_MemoryBank
        self.projection = nn.Sequential(
            nn.Linear(layers[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.queue_size = queue_size
        self.momentum = momentum
        self.register_buffer("queue", torch.randn(queue_size, 64))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.temperature = nn.Parameter(torch.FloatTensor([0.07]))

        # ******************************lucas 240326 initialize the parameters of modules
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        # nn.init.xavier_uniform_(self.transform_matrix.weight)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        # Lucas 240416 change the prediction layer initialization
        # nn.init.kaiming_uniform_(self.predict_layer.weight,
        #                          a=1, nonlinearity='sigmoid')
        for layer in self.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

        print(f"Lucas finish initialize model---PrefixTune_NCF_GAN_Mixtral2")

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.queue_size:
            self.queue[ptr:] = keys[:self.queue_size - ptr]
            self.queue[:batch_size - (self.queue_size - ptr)] = keys[self.queue_size - ptr:]
        else:
            self.queue[ptr:ptr + batch_size] = keys

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def contrastive_loss(self, q):
        # Positive logits: Nx1 (self-contrast)
        l_pos = torch.einsum('nc,nc->n', [q, q]).unsqueeze(-1)
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.t()])

        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature
        logits /= self.temperature

        # Labels: positives are the 0-th
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)

    def forward(
            self,
            user: torch.LongTensor = None,
            item: torch.LongTensor = None,
            neg_items: torch.LongTensor = None,
            rating: torch.LongTensor = None,
            neg_rating: torch.LongTensor = None,
            epoch: float = 0,
            num_epoch: float = 0,
            # domain: torch.LongTensor = None,
            # neg_domain: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if rating is not None:
            rating = rating.to(torch.float32)
        if neg_rating is not None:
            neg_rating = neg_rating.to(torch.float32)
        # if domain is not None:
        #     domain = domain.to(torch.long)
        # if neg_domain is not None:
        #     neg_domain = neg_domain.to(torch.long)
        print(f"Lucas model forward-----user: {user};----item:{item};----rating: {rating};----epoch: {epoch};----num_epoch: {num_epoch};")
        print(f"Lucas model forward-----neg_items: {neg_items};----neg_rating:{neg_rating};")

        embed_u_gmf = self.embed_user_GMF(user)
        embed_i_gmf = self.embed_item_GMF(item)
        output_gmf = embed_u_gmf * embed_i_gmf
        # Lucas 240702 added for contrastive learning
        user_extend = user.unsqueeze(1).repeat(1, self.num_neg_i).view(-1)
        print(f"Lucas model forward-----extend user: {user_extend};")
        embed_neg_u_gmf = self.embed_user_GMF(user_extend)
        embed_neg_i_gmf = self.embed_item_GMF(neg_items)
        output_neg_gmf = embed_neg_u_gmf * embed_neg_i_gmf

        embed_u_mlp = self.embed_user_MLP(user)
        embed_i_mlp = self.embed_item_MLP(item)
        interaction = torch.cat((embed_u_mlp, embed_i_mlp), -1)
        output_mlp = self.MLP_layers(interaction)
        # Lucas 240702 added for contrastive learning
        embed_neg_u_mlp = self.embed_user_MLP(user_extend)
        embed_neg_i_mlp = self.embed_item_MLP(neg_items)
        neg_inter = torch.cat((embed_neg_u_mlp, embed_neg_i_mlp), -1)
        output_neg_mlp = self.MLP_layers(neg_inter)

        print(f"Lucas GMF size: {output_gmf.size()}; MLP size: {output_mlp.size()};")
        concat_emb = torch.cat((output_gmf, output_mlp), -1)
        predict_ratings = self.predict_layer(concat_emb)
        # Lucas 240702 added for contrastive learning
        concat_neg_emb = torch.cat((output_neg_gmf, output_neg_mlp), -1)
        predict_neg_ratings = self.predict_layer(concat_neg_emb)

        # Lucas 240702 added contrastive learning
        cl_pos_emb = self.contrastive_learn(concat_emb)
        cl_neg_emb = self.contrastive_learn(concat_neg_emb)
        cl_neg_emb = cl_neg_emb.view(user.shape[0], self.num_neg_i, -1)

        # Lucas 240717 try contrastive_Memory Bank method
        q_concat_emb = self.projection(concat_emb)
        q_concat_emb = F.normalize(q_concat_emb, dim=1)

        # ******************************************Loss
        # Lucas 240717 try contrastive_Memory Bank method, not use original contrastive learning
        # # Calculate InfoNCE loss ---contrastive learning loss
        # contrastive_loss = info_nce_loss(cl_pos_emb, cl_pos_emb, cl_neg_emb, self.temperature)
        # print(f"Lucas--contrastive_loss: {contrastive_loss};")

        # Lucas 240717 try contrastive_Memory Bank method Contrastive loss
        contrastive_loss = self.contrastive_loss(q_concat_emb)

        rating_loss = self.rating_loss(predict_ratings.view(-1), rating)
        print(f"Lucas--predict_ratings: {predict_ratings}; original ratings: {rating}; rating loss: {rating_loss}")
        # print(f"Lucas--rating_loss: {rating_loss}")
        # return_loss_dict["scores"] = prediction.view(-1)

        # Lucas 240702 added for contrastive learning
        neg_rating_loss = self.rating_loss(predict_neg_ratings.view(-1), neg_rating)
        print(
            f"Lucas--neg predict_ratings: {predict_neg_ratings}; original neg ratings: {neg_rating}; neg rating loss: {neg_rating_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        print(f"Lucas--rating loss: {rating_loss};")
        # print(f"Lucas--rating loss device: {rating_loss.device};")
        print(f"Lucas--final contrastive_loss: {contrastive_loss}")

        # Calculate curriculum weights
        contrastive_weight = curriculum_weight(epoch, num_epoch, 0.6, strategy='exp')
        domain_weight = curriculum_weight(epoch, num_epoch, 0.6, strategy='linear')

        # Lucas 240719 try curriculm weight to contral the contrastive learning loss
        # total_loss = self.lambda1 * rating_loss + self.lambda2 * contrastive_loss  # + self.lambda1*neg_rating_loss
        total_loss = self.lambda1 * rating_loss + contrastive_weight * contrastive_loss  # + self.lambda1*neg_rating_loss
        print(f"Lucas--final total_loss: {total_loss}")
        loss_detail = [rating_loss, neg_rating_loss, contrastive_loss]
        return total_loss, loss_detail, predict_ratings.view(-1)


# Define an expert network
class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the gating network
class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_size, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)

class MoEModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_experts):
        super(MoEModel, self).__init__()
        self.output_size = output_size
        self.gating = GatingNetwork(input_size, num_experts)
        self.experts = nn.ModuleList([Expert(input_size, hidden_size, output_size) for _ in range(num_experts)])

    def forward(self, x):
        # print(f"x size: {x.size()}")
        gates = self.gating(x)
        # print(f"gates size: {gates.size()}")
        final_output = torch.zeros(x.size(0), self.output_size).to('cuda')
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            # print(f"expert_output size: {expert_output.size()}")
            # print(f"gates[:, i].unsqueeze(1) size: {gates[:, i].unsqueeze(1).size()}")
            final_output += gates[:, i].unsqueeze(1) * expert_output
        return final_output

class TransformerExpert(nn.Module):
    def __init__(self, num_pref, pref_dim=64, num_heads=4, num_layers=2, out_fea=128, device=''):
        super(TransformerExpert, self).__init__()
        self.gpu_device = device
        self.pref_embeddings = nn.Embedding(num_embeddings=num_pref, embedding_dim=pref_dim)  # Assuming SNPs are categorical
        self.positional_encoding = nn.Parameter(torch.zeros(1,num_pref, pref_dim))  # Sequence length 100
        encoder_layers = nn.TransformerEncoderLayer(d_model=pref_dim, nhead=num_heads, dim_feedforward=pref_dim*2)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(pref_dim * num_pref, out_fea)

    def forward(self, x):
        # x shape: (batch_size, num_pref)
        batch_size, num_pref = x.shape

        # Get embeddings for each SNP
        embedded_pref = self.pref_embeddings(torch.arange(num_pref).to(self.gpu_device))#(num_pref, pref_dim)

        # Multiply input data with corresponding SNP embeddings
        x_embedded = x.unsqueeze(2) * embedded_pref.unsqueeze(0)  # (batch_size, num_pref, pref_dim)

        x = x_embedded + self.positional_encoding
        x = self.transformer(x)
        # x = torch.mean(x, dim=1)  # Pooling across sequence dimension
        # Flatten the embedded input
        x_flat = x.view(batch_size, -1)  # (batch_size, num_pref * pref_dim)
        x_flat = self.fc(x_flat)
        return x_flat

class TransMoEModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_experts, device):
        super(TransMoEModel, self).__init__()
        self.device = device
        self.output_size = output_size
        self.gating = GatingNetwork(input_size, num_experts)
        self.experts = nn.ModuleList([Expert(input_size, hidden_size, output_size) for _ in range(num_experts)])
        self.experts = nn.ModuleList([TransformerExpert(input_size, pref_dim=64, num_heads=4, num_layers=2, out_fea=output_size, device=self.device) for _ in range(num_experts)])


    def forward(self, x):
        # print(f"x size: {x.size()}")
        gates = self.gating(x)
        # print(f"gates size: {gates.size()}")
        final_output = torch.zeros(x.size(0), self.output_size).to('cuda')
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            # print(f"expert_output size: {expert_output.size()}")
            # print(f"gates[:, i].unsqueeze(1) size: {gates[:, i].unsqueeze(1).size()}")
            final_output += gates[:, i].unsqueeze(1) * expert_output
        return final_output

# Define the Mixture of Experts model
class MoEModel_CD(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_experts):
        super(MoEModel_CD, self).__init__()
        self.output_size = output_size
        self.s_gating = GatingNetwork(input_size, num_experts)
        self.t_gating = GatingNetwork(input_size, num_experts)
        self.experts = nn.ModuleList([Expert(input_size, hidden_size, output_size) for _ in range(num_experts)])

    def forward(self, x, is_source):
        # print(f"x size: {x.size()}")
        if is_source:
            gates = self.s_gating(x)
        else:
            gates = self.t_gating(x)
        # print(f"gates size: {gates.size()}")
        final_output = torch.zeros(x.size(0), self.output_size).to('cuda')
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            # print(f"expert_output size: {expert_output.size()}")
            # print(f"gates[:, i].unsqueeze(1) size: {gates[:, i].unsqueeze(1).size()}")
            final_output += gates[:, i].unsqueeze(1) * expert_output
        return final_output


# Lucas 240730 add NCF-GAN-CL for experiments
# NCF + GAN + Contrastive learning
class NCF_GAN_CL(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, lambda_grl=1.0, temperature=0.1, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_GAN_CL, self).__init__()
        self.device_map = device_map
        self.num_u = num_u
        self.num_i = num_i
        self.num_neg_i = num_neg_i    # added for contrastive learning
        self.temperature = temperature  # added for contrastive learning
        self.factor_num = factor_num
        self.ncf_layer_num = ncf_layer_num
        self.drop_out = drop_out

        # self.peft_config = args[0]
        # self.model_config = config
        self.emb_size = emb_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda_grl = lambda_grl
        print(f"Lucas module num_u: {self.num_u}; num_i: {self.num_i}; lambda1: {self.lambda1}; lambda2: {self.lambda2}; lambda3: {self.lambda3}")
        print(f"Lucas module factor_num: {self.factor_num}")

        # print(f"Lucas--lucas_module--PeftPromptLlama2_V4--load_pretrained--device_map: {self.device_map}")

        # lucas 240402 NCF module
        self.embed_user_GMF = nn.Embedding(self.num_u, self.factor_num)
        self.embed_item_GMF = nn.Embedding(self.num_i, self.factor_num)
        # lucas 230327 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        self.embed_user_MLP = nn.Embedding(
            self.num_u, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.num_i, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # layer1：256-128；layer2: 128-64; layer3: 64-32
        mlp_modules = []
        mlp_out_size = []
        for i in range(self.ncf_layer_num):
            input_size = self.factor_num * (2 ** (self.ncf_layer_num - i))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.Dropout(p=self.drop_out))
            mlp_modules.append(nn.BatchNorm1d(input_size // 2))
            mlp_out_size.append(input_size)
            mlp_out_size.append(input_size // 2)
            mlp_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*mlp_modules)

        # Lucas 240731 added for grl and cl
        self.emb_fusion = nn.Sequential(
            nn.Linear(self.factor_num + mlp_out_size[-1] + self.emb_size*2, self.factor_num + mlp_out_size[-1]),
            nn.Dropout(p=self.drop_out),
            nn.Sigmoid(),
        )

        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.grl = GRL(self.lambda_grl)  # Lucas 240708 changed grl
        print(f"Lucas GRL mlp_out size: {mlp_out_size}")
        # self.domain_fc = nn.Linear(self.factor_num + mlp_out_size[-1], 2)
        self.domain_fc = nn.Sequential(
            nn.Linear(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)), self.factor_num),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, self.factor_num),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, 2)
        )
        # Lucas 240802 added for semantic domain grl
        self.sem_domain_fc = nn.Sequential(
            nn.Linear(self.emb_size, self.factor_num),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, self.factor_num),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, 2)
        )
        self.grl_activation = nn.Sigmoid()

        # Lucas 240709 added for feature selection
        self.feature_selector = FeatureSelector(self.factor_num + mlp_out_size[-1])

        self.concatenate_len = 2  # added user and item embedding
        self.user_embeddings = nn.Embedding(self.num_u, self.emb_size)#.to(device_map[''])
        self.item_embeddings = nn.Embedding(self.num_i, self.emb_size)#.to(device_map[''])

        # Lucas 240326 added grl for user/item embeddings
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.emb_size),
        )
        # self.user_semantic_classifier = Classifier(self.emb_size)  # 768, 10

        # Lucas 240326 added 最后预测层
        self.rating_loss = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss()

        self.predict_emb = self.factor_num + mlp_out_size[-1]
        print(f"Lucas predict_emb: {self.predict_emb}")
        # Lucas 240416 change the predict layer
        # self.predict_layer = nn.Linear(self.predict_emb, 1)
        dropout = 0.2
        self.activation = nn.Tanh()
        self.predict_layer = nn.Sequential(nn.Linear(self.predict_emb, self.predict_emb),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.BatchNorm1d(self.predict_emb),
                                           nn.Linear(self.predict_emb, self.factor_num),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.BatchNorm1d(self.factor_num),
                                           nn.Linear(self.factor_num, 1))

        # Lucas 240701 added for Contrastive Learning
        self.contrastive_learn = nn.Linear(self.predict_emb, int(self.predict_emb))

        # ******************************lucas 240326 initialize the parameters of modules
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        # nn.init.xavier_uniform_(self.transform_matrix.weight)
        initrange = 0.1
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        for m in self.emb_fusion:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        for m in self.domain_fc:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        # for m in self.modules():
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         m.bias.data.zero_()

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        # Lucas 240416 change the prediction layer initialization
        # nn.init.kaiming_uniform_(self.predict_layer.weight,
        #                          a=1, nonlinearity='sigmoid')
        for layer in self.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

        print(f"Lucas finish initialize model---PrefixTune_NCF_GAN_Mixtral2")

    def forward(
            self,
            user: torch.LongTensor = None,
            item: torch.LongTensor = None,
            neg_items: torch.LongTensor = None,
            rating: torch.LongTensor = None,
            neg_rating: torch.LongTensor = None,
            domain: torch.LongTensor = None,
            neg_domain: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if rating is not None:
            rating = rating.to(torch.float32)
        if neg_rating is not None:
            neg_rating = neg_rating.to(torch.float32)
        if domain is not None:
            domain = domain.to(torch.long)
        # if neg_domain is not None:
        #     neg_domain = neg_domain.to(torch.long)
        # print(f"Lucas model forward-----user: {user};----item:{item};----rating: {rating};----domain: {domain}")
        # print(f"Lucas model forward-----neg_items: {neg_items};----neg_rating:{neg_rating};")

        embed_u_gmf = self.embed_user_GMF(user)
        embed_i_gmf = self.embed_item_GMF(item)
        output_gmf = embed_u_gmf * embed_i_gmf
        # Lucas 240702 added for contrastive learning
        user_extend = user.unsqueeze(1).repeat(1, self.num_neg_i).view(-1)
        # print(f"Lucas model forward-----extend user: {user_extend};")
        embed_neg_u_gmf = self.embed_user_GMF(user_extend)
        embed_neg_i_gmf = self.embed_item_GMF(neg_items)
        output_neg_gmf = embed_neg_u_gmf * embed_neg_i_gmf

        embed_u_mlp = self.embed_user_MLP(user)
        embed_i_mlp = self.embed_item_MLP(item)
        interaction = torch.cat((embed_u_mlp, embed_i_mlp), -1)
        output_mlp = self.MLP_layers(interaction)
        # Lucas 240702 added for contrastive learning
        embed_neg_u_mlp = self.embed_user_MLP(user_extend)
        embed_neg_i_mlp = self.embed_item_MLP(neg_items)
        neg_inter = torch.cat((embed_neg_u_mlp, embed_neg_i_mlp), -1)
        output_neg_mlp = self.MLP_layers(neg_inter)

        # Lucas 240802 added another embedding for grl
        user_emb = self.user_embeddings(user)  # .to(self.device_map[''])  # (batch_size, emsize)
        item_emb = self.item_embeddings(item)
        # lucas 240326 added for tune user/item embedding
        user_emb = self.hidden_layers(user_emb)
        user_emb = self.grl(user_emb)
        predict_domain = self.grl_activation(self.sem_domain_fc(user_emb))

        # lucas 240402 added GRL layer for user embeddings in NCF
        # emb_u_ncf = torch.cat((embed_u_gmf, embed_u_mlp), -1)
        # u_ncf_dom_label = self.user_NCF_classifier(emb_u_ncf, self.p)
        # Lucas 240708 changed GRL input layer
        concat_emb = torch.cat((output_gmf, output_mlp, user_emb, item_emb), -1)
        concat_emb = self.emb_fusion(concat_emb)    # lucas 240731added for feature selection

        # Lucas 240709 added feature selection layer, 效果没有提升
        # selected_concat_emb = self.feature_selector(concat_emb)

        # Lucas 240731 change the grl embedding, only use user embedding
        # gmf+mlp grl
        concat_u_emb = torch.cat((output_gmf, embed_u_mlp), -1)
        concat_u_emb = self.grl(concat_u_emb)
        u_ncf_dom_label = self.grl_activation(self.domain_fc(concat_u_emb))

        # 230326 通过GRLayer GAN生成器判断source和target类别
        # p = float(batch_int + (epoch - dann_epoch) * num_batch / (args.epochs - dann_epoch) / num_batch)
        # p = 2. / (1. + np.exp(-10 * p)) - 1
        # # Lucas 240702 added for contrastive learning
        # emb_neg_u_ncf = torch.cat((embed_neg_u_gmf, embed_neg_u_mlp), -1)
        # neg_u_ncf_dom_label = self.user_NCF_classifier(emb_neg_u_ncf, self.p)

        # user_emb = self.user_embeddings(user)#.to(self.device_map[''])  # (batch_size, emsize)
        # item_emb = self.item_embeddings(item)#.to(self.device_map[''])  # (batch_size, emsize)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.user_embeddings(user_extend)
        # neg_i_emb = self.item_embeddings(neg_items)
        #
        # # lucas 240326 added for tune user/item embedding
        # user_emb = self.hidden_layers(user_emb)
        # item_emb = self.hidden_layers(item_emb)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.hidden_layers(neg_u_emb)
        # neg_i_emb = self.hidden_layers(neg_i_emb)
        # 230326 通过GRLayer GAN生成器判断source和target类别
        # p = float(batch_int + (epoch - dann_epoch) * num_batch / (args.epochs - dann_epoch) / num_batch)
        # p = 2. / (1. + np.exp(-10 * p)) - 1
        # predict_domain = self.user_semantic_classifier(user_emb, self.p)  # classifier，输入40维度，输出10维度,经过GRL再经过线性层+sigmoid输出2维分类
        # Lucas 240702 added for contrastive learning
        # predict_neg_domain = self.user_semantic_classifier(neg_u_emb, self.p)

        # *****************************************training
        # Lucas 240327 added for rating prediction
        # print(f"Lucas GMF size: {output_gmf.size()}; MLP size: {output_mlp.size()};")
        predict_ratings = self.predict_layer(concat_emb)

        # Lucas 240802 added another embedding for grl o
        user_neg_emb = self.user_embeddings(user_extend)  # .to(self.device_map[''])  # (batch_size, emsize)
        item_neg_emb = self.item_embeddings(neg_items)
        # lucas 240326 added for tune user/item embedding
        user_neg_emb = self.hidden_layers(user_neg_emb)
        user_neg_emb = self.grl(user_neg_emb)
        neg_sem_u_domain = self.grl_activation(self.sem_domain_fc(user_neg_emb))
        # Lucas 240702 embedding for negative rating prediction
        concat_neg_emb = torch.cat((output_neg_gmf, output_neg_mlp, user_neg_emb, item_neg_emb), -1)
        concat_neg_emb = self.emb_fusion(concat_neg_emb)  # lucas 240731added for feature

        # negative gmf+mlp grl
        concat_neg_u_emb = torch.cat((output_neg_gmf, embed_neg_u_mlp), -1)
        concat_neg_u_emb = self.grl(concat_neg_u_emb)
        neg_u_ncf_dom_label = self.grl_activation(self.domain_fc(concat_neg_u_emb))

        predict_neg_ratings = self.predict_layer(concat_neg_emb)

        # Lucas 240702 added contrastive learning
        cl_pos_emb = self.contrastive_learn(concat_emb)
        cl_neg_emb = self.contrastive_learn(concat_neg_emb)
        cl_neg_emb = cl_neg_emb.view(user.shape[0], self.num_neg_i, -1)

        # ******************************************Loss
        # Calculate InfoNCE loss ---contrastive learning loss
        contrastive_loss = info_nce_loss(cl_pos_emb, cl_pos_emb, cl_neg_emb, self.temperature)

        rating_loss = self.rating_loss(predict_ratings.view(-1), rating)

        # Lucas 240702 added for contrastive learning
        neg_rating_loss = self.rating_loss(predict_neg_ratings.view(-1), neg_rating)

        # GRL classification loss
        pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        sem_u_grl_loss = self.exp_loss_fn(predict_domain, domain)
        neg_ncf_grl_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        neg_sem_u_grl_loss = self.exp_loss_fn(neg_sem_u_domain, neg_domain)
        print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain};")
        # # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        # print(f"Lucas--rating loss: {rating_loss}; domain prediction loss: {pred_ncf_domain_loss}")
        # print(f"Lucas--rating loss device: {rating_loss.device}; domain loss dtype: {pred_ncf_domain_loss.device}; predict_ncf_domain loss: {pred_ncf_domain_loss.device}")

        # total_loss = self.lambda1*rating_loss + self.lambda1*neg_rating_loss + self.lambda2*contrastive_loss + self.lambda3*pred_ncf_domain_loss + self.lambda3*sem_u_grl_loss # + self.lambda3*neg_ncf_grl_loss + self.lambda3*neg_sem_u_grl_loss # + self.lambda1*neg_rating_loss
        # total_loss = self.lambda1 * rating_loss + self.lambda1 * neg_rating_loss + self.lambda3 * pred_ncf_domain_loss + self.lambda3 * sem_u_grl_loss + self.lambda3*neg_ncf_grl_loss + self.lambda3*neg_sem_u_grl_loss
        # total_loss = self.lambda1 * rating_loss + self.lambda3 * pred_ncf_domain_loss + self.lambda3 * sem_u_grl_loss
        total_loss = self.lambda1 * rating_loss
        # total_loss = self.lambda1 * rating_loss + self.lambda3 * pred_ncf_domain_loss
        print(f"Lucas--final total_loss: {total_loss}; rating_loss: {rating_loss}; contrastive_loss: {contrastive_loss}; pred_ncf_domain_loss: {pred_ncf_domain_loss}")
        loss_detail = [rating_loss, neg_rating_loss, contrastive_loss, pred_ncf_domain_loss, sem_u_grl_loss, neg_ncf_grl_loss, neg_sem_u_grl_loss]
        return total_loss, loss_detail, predict_ratings.view(-1)


# Lucas 240805 add MOE into NCF for experiments
# NCF + MOE
class NCF_MOE(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, moe_layer_num=3, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, lambda_grl=1.0, temperature=0.1, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_MOE, self).__init__()
        self.device_map = device_map
        self.num_u = num_u
        self.num_i = num_i
        self.num_neg_i = num_neg_i    # added for contrastive learning
        self.temperature = temperature  # added for contrastive learning
        self.factor_num = factor_num
        self.ncf_layer_num = ncf_layer_num
        self.moe_layer_num = moe_layer_num  # added for moe layer
        self.drop_out = drop_out

        # self.peft_config = args[0]
        # self.model_config = config
        self.emb_size = emb_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda_grl = lambda_grl
        print(f"Lucas module num_u: {self.num_u}; num_i: {self.num_i}; lambda1: {self.lambda1}; lambda2: {self.lambda2}; lambda3: {self.lambda3}")
        print(f"Lucas module factor_num: {self.factor_num}")

        # print(f"Lucas--lucas_module--PeftPromptLlama2_V4--load_pretrained--device_map: {self.device_map}")

        # lucas 240402 NCF module
        self.embed_user_GMF = nn.Embedding(self.num_u, self.factor_num)
        self.embed_item_GMF = nn.Embedding(self.num_i, self.factor_num)
        # lucas 230327 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        self.embed_user_MLP = nn.Embedding(
            self.num_u, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.num_i, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # layer1：256-128；layer2: 128-64; layer3: 64-32
        mlp_modules = []
        mlp_out_size = []
        for i in range(self.ncf_layer_num):
            input_size = self.factor_num * (2 ** (self.ncf_layer_num - i))
            mlp_modules.append(nn.Dropout(p=self.drop_out))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_out_size.append(input_size)
            mlp_out_size.append(input_size // 2)
            mlp_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*mlp_modules)

        # Lucas 240731 added feature integration
        self.emb_fusion = nn.Sequential(
            nn.Linear(self.factor_num + mlp_out_size[-1], self.factor_num + mlp_out_size[-1]),
            nn.Sigmoid(),
        )

        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.grl = GRL(self.lambda_grl)  # Lucas 240708 changed grl
        print(f"Lucas GRL mlp_out size: {mlp_out_size}")
        # self.domain_fc = nn.Linear(self.factor_num + mlp_out_size[-1], 2)
        self.domain_fc = nn.Sequential(
            nn.Linear(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)), self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, 2)
        )
        self.grl_activation = nn.Sigmoid()

        # Lucas 240709 added for feature selection
        self.feature_selector = FeatureSelector(self.factor_num + mlp_out_size[-1])

        self.concatenate_len = 2  # added user and item embedding
        self.user_embeddings = nn.Embedding(self.num_u, self.emb_size)#.to(device_map[''])
        self.item_embeddings = nn.Embedding(self.num_i, self.emb_size)#.to(device_map[''])

        # Lucas 240326 added grl for user/item embeddings
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
        )
        # self.user_semantic_classifier = Classifier(self.emb_size)  # 768, 10

        # Lucas 240326 added 最后预测层
        self.rating_loss = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss()

        self.predict_emb = self.factor_num * 2 # + self.emb_size * 2
        # print(f"Lucas predict_emb: {self.predict_emb}")

        # Lucas 240731 added MOE layer
        self.moe_layer = MoEModel_CD(self.predict_emb, self.predict_emb*2, self.predict_emb, self.moe_layer_num)

        # Lucas 240416 change the predict layer
        # self.predict_layer = nn.Linear(self.predict_emb, 1)
        dropout = 0.2
        self.activation = nn.Tanh()
        self.predict_layer = nn.Sequential(nn.Linear(self.predict_emb, self.predict_emb),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.BatchNorm1d(self.predict_emb),
                                           nn.Linear(self.predict_emb, self.factor_num),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(self.factor_num, 1))

        # Lucas 240701 added for Contrastive Learning
        self.contrastive_learn = nn.Linear(self.predict_emb, int(self.predict_emb))

        # ******************************lucas 240326 initialize the parameters of modules
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        # nn.init.xavier_uniform_(self.transform_matrix.weight)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         m.bias.data.zero_()

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        # Lucas 240416 change the prediction layer initialization
        # nn.init.kaiming_uniform_(self.predict_layer.weight,
        #                          a=1, nonlinearity='sigmoid')
        for layer in self.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

        print(f"Lucas finish initialize model---PrefixTune_NCF_GAN_Mixtral2")

    def forward(
            self,
            user: torch.LongTensor = None,
            item: torch.LongTensor = None,
            neg_items: torch.LongTensor = None,
            rating: torch.LongTensor = None,
            neg_rating: torch.LongTensor = None,
            domain: torch.LongTensor = None,
            is_source: bool = None
            # neg_domain: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if rating is not None:
            rating = rating.to(torch.float32)
        if neg_rating is not None:
            neg_rating = neg_rating.to(torch.float32)
        if domain is not None:
            domain = domain.to(torch.long)
        # if neg_domain is not None:
        #     neg_domain = neg_domain.to(torch.long)
        print(f"Lucas model forward-----user: {user};----item:{item};----rating: {rating};----domain: {domain}--is_source: {is_source}")
        # print(f"Lucas model forward-----neg_items: {neg_items};----neg_rating:{neg_rating};")

        embed_u_gmf = self.embed_user_GMF(user)
        embed_i_gmf = self.embed_item_GMF(item)
        output_gmf = embed_u_gmf * embed_i_gmf
        # Lucas 240702 added for contrastive learning
        user_extend = user.unsqueeze(1).repeat(1, self.num_neg_i).view(-1)
        print(f"Lucas model forward-----extend user: {user_extend};")
        embed_neg_u_gmf = self.embed_user_GMF(user_extend)
        embed_neg_i_gmf = self.embed_item_GMF(neg_items)
        output_neg_gmf = embed_neg_u_gmf * embed_neg_i_gmf

        embed_u_mlp = self.embed_user_MLP(user)
        embed_i_mlp = self.embed_item_MLP(item)
        interaction = torch.cat((embed_u_mlp, embed_i_mlp), -1)
        output_mlp = self.MLP_layers(interaction)
        # Lucas 240702 added for contrastive learning
        embed_neg_u_mlp = self.embed_user_MLP(user_extend)
        embed_neg_i_mlp = self.embed_item_MLP(neg_items)
        neg_inter = torch.cat((embed_neg_u_mlp, embed_neg_i_mlp), -1)
        output_neg_mlp = self.MLP_layers(neg_inter)

        # lucas 240402 added GRL layer for user embeddings in NCF
        # emb_u_ncf = torch.cat((embed_u_gmf, embed_u_mlp), -1)
        # u_ncf_dom_label = self.user_NCF_classifier(emb_u_ncf, self.p)
        # Lucas 240708 changed GRL input layer
        concat_emb = torch.cat((output_gmf, output_mlp), -1)
        concat_emb = self.emb_fusion(concat_emb)    # lucas 240731added for feature selection


        # Lucas 240709 added feature selection layer, 效果没有提升
        # selected_concat_emb = self.feature_selector(concat_emb)

        # Domain classification task
        # Lucas 240731 change the grl embedding, only use user embedding
        # concat_emb_grl = self.grl(concat_emb)
        concat_u_emb = torch.cat((output_gmf, embed_u_mlp), -1)
        concat_u_emb = self.grl(concat_u_emb)
        u_ncf_dom_label = self.grl_activation(self.domain_fc(concat_u_emb))
        # # Lucas 240702 added for contrastive learning
        # emb_neg_u_ncf = torch.cat((embed_neg_u_gmf, embed_neg_u_mlp), -1)
        # neg_u_ncf_dom_label = self.user_NCF_classifier(emb_neg_u_ncf, self.p)

        # user_emb = self.user_embeddings(user)#.to(self.device_map[''])  # (batch_size, emsize)
        # item_emb = self.item_embeddings(item)#.to(self.device_map[''])  # (batch_size, emsize)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.user_embeddings(user_extend)
        # neg_i_emb = self.item_embeddings(neg_items)
        #
        # # lucas 240326 added for tune user/item embedding
        # user_emb = self.hidden_layers(user_emb)
        # item_emb = self.hidden_layers(item_emb)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.hidden_layers(neg_u_emb)
        # neg_i_emb = self.hidden_layers(neg_i_emb)
        # 230326 通过GRLayer GAN生成器判断source和target类别
        # p = float(batch_int + (epoch - dann_epoch) * num_batch / (args.epochs - dann_epoch) / num_batch)
        # p = 2. / (1. + np.exp(-10 * p)) - 1
        # predict_domain = self.user_semantic_classifier(user_emb, self.p)  # classifier，输入40维度，输出10维度,经过GRL再经过线性层+sigmoid输出2维分类
        # Lucas 240702 added for contrastive learning
        # predict_neg_domain = self.user_semantic_classifier(neg_u_emb, self.p)

        # *****************************************training
        # Lucas 240327 added for rating prediction
        print(f"Lucas GMF size: {output_gmf.size()}; MLP size: {output_mlp.size()};")
        print(f"concat_emb: {concat_emb.size()}")
        moe_concat_emb = self.moe_layer(concat_emb, is_source)
        predict_ratings = self.predict_layer(moe_concat_emb)
        # Lucas 240702 added for contrastive learning
        concat_neg_emb = torch.cat((output_neg_gmf, output_neg_mlp), -1)
        concat_neg_emb = self.emb_fusion(concat_neg_emb)  # lucas 240731added for feature
        predict_neg_ratings = self.predict_layer(concat_neg_emb)

        # Lucas 240702 added contrastive learning
        cl_pos_emb = self.contrastive_learn(concat_emb)
        cl_neg_emb = self.contrastive_learn(concat_neg_emb)
        cl_neg_emb = cl_neg_emb.view(user.shape[0], self.num_neg_i, -1)

        # ******************************************Loss
        # Calculate InfoNCE loss ---contrastive learning loss
        contrastive_loss = info_nce_loss(cl_pos_emb, cl_pos_emb, cl_neg_emb, self.temperature)
        print(f"Lucas--contrastive_loss: {contrastive_loss};")

        rating_loss = self.rating_loss(predict_ratings.view(-1), rating)
        print(f"Lucas--predict_ratings: {predict_ratings}; original ratings: {rating}; rating loss: {rating_loss}")
        # print(f"Lucas--rating_loss: {rating_loss}")
        # return_loss_dict["scores"] = prediction.view(-1)

        # Lucas 240702 added for contrastive learning
        neg_rating_loss = self.rating_loss(predict_neg_ratings.view(-1), neg_rating)
        print(f"Lucas--neg predict_ratings: {predict_neg_ratings}; original neg ratings: {neg_rating}; neg rating loss: {neg_rating_loss}")

        # GRL classification loss
        # pred_domains_loss = self.exp_loss_fn(predict_domain, domain)
        # print(f"Lucas--predict_domains: {predict_domain}; original domains: {domain}; prefix pre domain loss: {pred_domains_loss}")
        pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain}; ncf pre domain loss: {pred_ncf_domain_loss}")
        # # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        # print(f"Lucas--rating loss: {rating_loss}; domain prediction loss: {pred_ncf_domain_loss}")
        # print(f"Lucas--rating loss device: {rating_loss.device}; domain loss dtype: {pred_ncf_domain_loss.device}; predict_ncf_domain loss: {pred_ncf_domain_loss.device}")

        # total_loss = self.lambda1*rating_loss + self.lambda2*contrastive_loss + self.lambda3*pred_ncf_domain_loss # + self.lambda1*neg_rating_loss
        total_loss = self.lambda1*rating_loss
        print(f"Lucas--final total_loss: {total_loss}")
        loss_detail = [rating_loss, neg_rating_loss, contrastive_loss, pred_ncf_domain_loss]
        return total_loss, loss_detail, predict_ratings.view(-1)


class NCF_GAN_MOE(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, lambda_grl=1.0, temperature=0.1, moe_layer_num=3, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_GAN_MOE, self).__init__()
        self.device_map = device_map
        self.num_u = num_u
        self.num_i = num_i
        self.num_neg_i = num_neg_i    # added for contrastive learning
        self.temperature = temperature  # added for contrastive learning
        self.factor_num = factor_num
        self.ncf_layer_num = ncf_layer_num
        self.drop_out = drop_out

        # self.peft_config = args[0]
        # self.model_config = config
        self.emb_size = emb_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda_grl = lambda_grl
        self.moe_layer_num = moe_layer_num
        print(f"Lucas module num_u: {self.num_u}; num_i: {self.num_i}; lambda1: {self.lambda1}; lambda2: {self.lambda2}; lambda3: {self.lambda3}")
        print(f"Lucas module factor_num: {self.factor_num}")

        # print(f"Lucas--lucas_module--PeftPromptLlama2_V4--load_pretrained--device_map: {self.device_map}")

        # lucas 240402 NCF module
        self.embed_user_GMF = nn.Embedding(self.num_u, self.factor_num)
        self.embed_item_GMF = nn.Embedding(self.num_i, self.factor_num)
        # lucas 230327 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        self.embed_user_MLP = nn.Embedding(
            self.num_u, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.num_i, self.factor_num * (2 ** (self.ncf_layer_num - 1)))

        # nn.Sigmoid()
        # layer1：256-128；layer2: 128-64; layer3: 64-32
        mlp_modules = []
        mlp_out_size = []
        for i in range(self.ncf_layer_num):
            input_size = self.factor_num * (2 ** (self.ncf_layer_num - i))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.Dropout(p=self.drop_out))
            mlp_modules.append(nn.BatchNorm1d(input_size // 2))
            mlp_out_size.append(input_size)
            mlp_out_size.append(input_size // 2)
            mlp_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*mlp_modules)

        # Lucas 240731 added for grl and cl
        self.emb_fusion = nn.Sequential(
            nn.Linear(self.factor_num + mlp_out_size[-1], self.factor_num + mlp_out_size[-1]),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num + mlp_out_size[-1]),
            nn.Sigmoid(),
        )

        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.grl = GRL(self.lambda_grl)  # Lucas 240708 changed grl
        print(f"Lucas GRL mlp_out size: {mlp_out_size}")
        # self.domain_fc = nn.Linear(self.factor_num + mlp_out_size[-1], 2)
        self.domain_fc = nn.Sequential(
            # nn.Linear(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)), self.factor_num),
            nn.Linear(self.factor_num + mlp_out_size[-1], self.factor_num),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, self.factor_num),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, 2)
        )
        self.grl_activation = nn.Sigmoid()

        # # Lucas 240709 added for feature selection
        # self.feature_selector = FeatureSelector(self.factor_num + mlp_out_size[-1])

        self.concatenate_len = 2  # added user and item embedding
        self.user_embeddings = nn.Embedding(self.num_u, self.emb_size)#.to(device_map[''])
        self.item_embeddings = nn.Embedding(self.num_i, self.emb_size)#.to(device_map[''])

        # Lucas 240326 added grl for user/item embeddings
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.emb_size),
            # nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.emb_size),
            # nn.LayerNorm(self.emb_size),
        )
        self.user_semantic_classifier = Classifier(self.emb_size)  # 768, 10

        # Lucas 240326 added 最后预测层
        self.rating_loss = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss()

        # self.predict_emb = self.factor_num * 2 # + self.emb_size * 2
        self.predict_emb = self.factor_num + mlp_out_size[-1]
        print(f"Lucas predict_emb: {self.predict_emb}")
        # Lucas 240416 change the predict layer
        # self.predict_layer = nn.Linear(self.predict_emb, 1)
        dropout = 0.2
        self.activation = nn.Tanh()
        self.predict_layer = nn.Sequential(nn.Linear(self.predict_emb, self.predict_emb),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.BatchNorm1d(self.predict_emb),
                                           nn.Linear(self.predict_emb, self.factor_num),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.BatchNorm1d(self.factor_num),
                                           nn.Linear(self.factor_num, 1))
        # Lucas 240731 added MOE layer
        self.moe_layer = MoEModel(self.predict_emb, self.predict_emb * 2, self.predict_emb, self.moe_layer_num)

        # # Lucas 240701 added for Contrastive Learning
        # self.contrastive_learn = nn.Linear(self.predict_emb, int(self.predict_emb / 2))

        # ******************************lucas 240326 initialize the parameters of modules
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        # nn.init.xavier_uniform_(self.transform_matrix.weight)
        initrange = 0.1
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        for m in self.emb_fusion:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        for m in self.domain_fc:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        # for m in self.modules():
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         m.bias.data.zero_()

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        # Lucas 240416 change the prediction layer initialization
        # nn.init.kaiming_uniform_(self.predict_layer.weight,
        #                          a=1, nonlinearity='sigmoid')
        for layer in self.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

        print(f"Lucas finish initialize model---PrefixTune_NCF_GAN_Mixtral2")

    def forward(
            self,
            user: torch.LongTensor = None,
            item: torch.LongTensor = None,
            rating: torch.LongTensor = None,
            domain: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if rating is not None:
            rating = rating.to(torch.float32)
        if domain is not None:
            domain = domain.to(torch.long)
        # print(f"Lucas model forward-----user: {user};----item:{item};----rating: {rating};----domain: {domain}")

        embed_u_gmf = self.embed_user_GMF(user)
        embed_i_gmf = self.embed_item_GMF(item)
        output_gmf = embed_u_gmf * embed_i_gmf
        # # Lucas 240702 added for contrastive learning
        # user_extend = user.unsqueeze(1).repeat(1, self.num_neg_i).view(-1)
        # print(f"Lucas model forward-----extend user: {user_extend};")
        # embed_neg_u_gmf = self.embed_user_GMF(user_extend)
        # embed_neg_i_gmf = self.embed_item_GMF(neg_items)
        # output_neg_gmf = embed_neg_u_gmf * embed_neg_i_gmf

        embed_u_mlp = self.embed_user_MLP(user)
        embed_i_mlp = self.embed_item_MLP(item)
        interaction = torch.cat((embed_u_mlp, embed_i_mlp), -1)
        output_mlp = self.MLP_layers(interaction)
        # # Lucas 240702 added for contrastive learning
        # embed_neg_u_mlp = self.embed_user_MLP(user_extend)
        # embed_neg_i_mlp = self.embed_item_MLP(neg_items)
        # neg_inter = torch.cat((embed_neg_u_mlp, embed_neg_i_mlp), -1)
        # output_neg_mlp = self.MLP_layers(neg_inter)

        # lucas 240402 added GRL layer for user embeddings in NCF
        # emb_u_ncf = torch.cat((embed_u_gmf, embed_u_mlp), -1)
        # u_ncf_dom_label = self.user_NCF_classifier(emb_u_ncf, self.p)
        # Lucas 240708 changed GRL input layer
        concat_emb = torch.cat((output_gmf, output_mlp), -1)
        concat_emb = self.emb_fusion(concat_emb)    # lucas 240731added for feature selection
        concat_emb = self.moe_layer(concat_emb)
        # Lucas 240709 added feature selection layer, 效果没有提升
        # selected_concat_emb = self.feature_selector(concat_emb)

        # Domain classification task
        concat_emb_grl = self.grl(concat_emb)
        u_ncf_dom_label = self.grl_activation(self.domain_fc(concat_emb_grl))

        # # Lucas 240702 added for contrastive learning
        # emb_neg_u_ncf = torch.cat((embed_neg_u_gmf, embed_neg_u_mlp), -1)
        # neg_u_ncf_dom_label = self.user_NCF_classifier(emb_neg_u_ncf, self.p)

        # user_emb = self.user_embeddings(user)#.to(self.device_map[''])  # (batch_size, emsize)
        # item_emb = self.item_embeddings(item)#.to(self.device_map[''])  # (batch_size, emsize)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.user_embeddings(user_extend)
        # neg_i_emb = self.item_embeddings(neg_items)

        # lucas 240326 added for tune user/item embedding
        # user_emb = self.hidden_layers(user_emb)
        # item_emb = self.hidden_layers(item_emb)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.hidden_layers(neg_u_emb)
        # neg_i_emb = self.hidden_layers(neg_i_emb)
        # 230326 通过GRLayer GAN生成器判断source和target类别
        # p = float(batch_int + (epoch - dann_epoch) * num_batch / (args.epochs - dann_epoch) / num_batch)
        # p = 2. / (1. + np.exp(-10 * p)) - 1
        # predict_domain = self.user_semantic_classifier(user_emb, self.p)  # classifier，输入40维度，输出10维度,经过GRL再经过线性层+sigmoid输出2维分类
        # # Lucas 240702 added for contrastive learning
        # predict_neg_domain = self.user_semantic_classifier(neg_u_emb, self.p)

        # *****************************************training
        # Lucas 240327 added for rating prediction
        print(f"Lucas GMF size: {output_gmf.size()}; MLP size: {output_mlp.size()};")
        # concat_emb = torch.cat((output_gmf, output_mlp), -1)

        predict_ratings = self.predict_layer(concat_emb)
        # # Lucas 240702 added for contrastive learning
        # concat_neg_emb = torch.cat((output_neg_gmf, output_neg_mlp, neg_u_emb, neg_i_emb), -1)
        # predict_neg_ratings = self.predict_layer(concat_neg_emb)

        # # Lucas 240702 added contrastive learning
        # cl_pos_emb = self.contrastive_learn(concat_emb)
        # cl_neg_emb = self.contrastive_learn(concat_neg_emb)
        # cl_neg_emb = cl_neg_emb.view(user.shape[0], self.num_neg_i, -1)

        # ******************************************Loss
        # # Calculate InfoNCE loss ---contrastive learning loss
        # contrastive_loss = info_nce_loss(cl_pos_emb, cl_pos_emb, cl_neg_emb, self.temperature)
        # print(f"Lucas--contrastive_loss: {contrastive_loss};")

        rating_loss = self.rating_loss(predict_ratings.view(-1), rating)
        print(f"Lucas--predict_ratings: {predict_ratings}; original ratings: {rating}; rating loss: {rating_loss}")
        # print(f"Lucas--rating_loss: {rating_loss}")
        # return_loss_dict["scores"] = prediction.view(-1)

        # # Lucas 240702 added for contrastive learning
        # neg_rating_loss = self.rating_loss(predict_neg_ratings.view(-1), neg_rating)
        # print(f"Lucas--neg predict_ratings: {predict_neg_ratings}; original neg ratings: {neg_rating}; neg rating loss: {neg_rating_loss}")

        # GRL classification loss
        # pred_domains_loss = self.exp_loss_fn(predict_domain, domain)
        # print(f"Lucas--predict_domains: {predict_domain}; original domains: {domain}; prefix pre domain loss: {pred_domains_loss}")
        pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain}; ncf pre domain loss: {pred_ncf_domain_loss}")
        # # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        print(f"Lucas--rating loss: {rating_loss}; domain prediction loss: {pred_ncf_domain_loss}")
        # print(f"Lucas--rating loss device: {rating_loss.device}; domain loss dtype: {pred_ncf_domain_loss.device}; predict_ncf_domain loss: {pred_ncf_domain_loss.device}")

        total_loss = self.lambda1*rating_loss + self.lambda2*pred_ncf_domain_loss
        print(f"Lucas--final total_loss: {total_loss}")
        loss_detail = [rating_loss, pred_ncf_domain_loss]
        return total_loss, loss_detail, predict_ratings.view(-1)

# 将MOE中expert改为transformer结构
class NCF_GAN_TRNSMOE(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, lambda_grl=1.0, temperature=0.1, moe_layer_num=3, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_GAN_TRNSMOE, self).__init__()
        self.device_map = device_map
        self.num_u = num_u
        self.num_i = num_i
        self.num_neg_i = num_neg_i    # added for contrastive learning
        self.temperature = temperature  # added for contrastive learning
        self.factor_num = factor_num
        self.ncf_layer_num = ncf_layer_num
        self.drop_out = drop_out

        # self.peft_config = args[0]
        # self.model_config = config
        self.emb_size = emb_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda_grl = lambda_grl
        self.moe_layer_num = moe_layer_num
        print(f"Lucas module num_u: {self.num_u}; num_i: {self.num_i}; lambda1: {self.lambda1}; lambda2: {self.lambda2}; lambda3: {self.lambda3}")
        print(f"Lucas module factor_num: {self.factor_num}")

        # print(f"Lucas--lucas_module--PeftPromptLlama2_V4--load_pretrained--device_map: {self.device_map}")

        # lucas 240402 NCF module
        self.embed_user_GMF = nn.Embedding(self.num_u, self.factor_num)
        self.embed_item_GMF = nn.Embedding(self.num_i, self.factor_num)
        # lucas 230327 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        self.embed_user_MLP = nn.Embedding(
            self.num_u, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.num_i, self.factor_num * (2 ** (self.ncf_layer_num - 1)))

        # nn.Sigmoid()
        # layer1：256-128；layer2: 128-64; layer3: 64-32
        mlp_modules = []
        mlp_out_size = []
        for i in range(self.ncf_layer_num):
            input_size = self.factor_num * (2 ** (self.ncf_layer_num - i))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.Dropout(p=self.drop_out))
            mlp_modules.append(nn.BatchNorm1d(input_size // 2))
            mlp_out_size.append(input_size)
            mlp_out_size.append(input_size // 2)
            mlp_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*mlp_modules)

        # Lucas 240731 added for grl and cl
        self.emb_fusion = nn.Sequential(
            nn.Linear(self.factor_num + mlp_out_size[-1], self.factor_num + mlp_out_size[-1]),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num + mlp_out_size[-1]),
            nn.Sigmoid(),
        )

        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.grl = GRL(self.lambda_grl)  # Lucas 240708 changed grl
        print(f"Lucas GRL mlp_out size: {mlp_out_size}")
        # self.domain_fc = nn.Linear(self.factor_num + mlp_out_size[-1], 2)
        self.domain_fc = nn.Sequential(
            # nn.Linear(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)), self.factor_num),
            nn.Linear(self.factor_num + mlp_out_size[-1], self.factor_num),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, self.factor_num),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, 2)
        )
        self.grl_activation = nn.Sigmoid()

        # # Lucas 240709 added for feature selection
        # self.feature_selector = FeatureSelector(self.factor_num + mlp_out_size[-1])

        self.concatenate_len = 2  # added user and item embedding
        self.user_embeddings = nn.Embedding(self.num_u, self.emb_size)#.to(device_map[''])
        self.item_embeddings = nn.Embedding(self.num_i, self.emb_size)#.to(device_map[''])

        # Lucas 240326 added grl for user/item embeddings
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.emb_size),
            # nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.emb_size),
            # nn.LayerNorm(self.emb_size),
        )
        self.user_semantic_classifier = Classifier(self.emb_size)  # 768, 10

        # Lucas 240326 added 最后预测层
        self.rating_loss = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss()

        # self.predict_emb = self.factor_num * 2 # + self.emb_size * 2
        self.predict_emb = self.factor_num + mlp_out_size[-1]
        print(f"Lucas predict_emb: {self.predict_emb}")
        # Lucas 240416 change the predict layer
        # self.predict_layer = nn.Linear(self.predict_emb, 1)
        dropout = 0.2
        self.activation = nn.Tanh()
        self.predict_layer = nn.Sequential(nn.Linear(self.predict_emb, self.predict_emb),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.BatchNorm1d(self.predict_emb),
                                           nn.Linear(self.predict_emb, self.factor_num),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.BatchNorm1d(self.factor_num),
                                           nn.Linear(self.factor_num, 1))
        # Lucas 240731 added MOE layer
        self.moe_layer = TransMoEModel(self.predict_emb, self.predict_emb * 2, self.predict_emb, self.moe_layer_num, self.device_map)

        # # Lucas 240701 added for Contrastive Learning
        # self.contrastive_learn = nn.Linear(self.predict_emb, int(self.predict_emb / 2))

        # ******************************lucas 240326 initialize the parameters of modules
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        # nn.init.xavier_uniform_(self.transform_matrix.weight)
        initrange = 0.1
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        for m in self.emb_fusion:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        for m in self.domain_fc:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        # for m in self.modules():
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         m.bias.data.zero_()

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        # Lucas 240416 change the prediction layer initialization
        # nn.init.kaiming_uniform_(self.predict_layer.weight,
        #                          a=1, nonlinearity='sigmoid')
        for layer in self.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

        print(f"Lucas finish initialize model---PrefixTune_NCF_GAN_Mixtral2")

    def forward(
            self,
            user: torch.LongTensor = None,
            item: torch.LongTensor = None,
            rating: torch.LongTensor = None,
            domain: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if rating is not None:
            rating = rating.to(torch.float32)
        if domain is not None:
            domain = domain.to(torch.long)
        # print(f"Lucas model forward-----user: {user};----item:{item};----rating: {rating};----domain: {domain}")

        embed_u_gmf = self.embed_user_GMF(user)
        embed_i_gmf = self.embed_item_GMF(item)
        output_gmf = embed_u_gmf * embed_i_gmf
        # # Lucas 240702 added for contrastive learning
        # user_extend = user.unsqueeze(1).repeat(1, self.num_neg_i).view(-1)
        # print(f"Lucas model forward-----extend user: {user_extend};")
        # embed_neg_u_gmf = self.embed_user_GMF(user_extend)
        # embed_neg_i_gmf = self.embed_item_GMF(neg_items)
        # output_neg_gmf = embed_neg_u_gmf * embed_neg_i_gmf

        embed_u_mlp = self.embed_user_MLP(user)
        embed_i_mlp = self.embed_item_MLP(item)
        interaction = torch.cat((embed_u_mlp, embed_i_mlp), -1)
        output_mlp = self.MLP_layers(interaction)
        # # Lucas 240702 added for contrastive learning
        # embed_neg_u_mlp = self.embed_user_MLP(user_extend)
        # embed_neg_i_mlp = self.embed_item_MLP(neg_items)
        # neg_inter = torch.cat((embed_neg_u_mlp, embed_neg_i_mlp), -1)
        # output_neg_mlp = self.MLP_layers(neg_inter)

        # lucas 240402 added GRL layer for user embeddings in NCF
        # emb_u_ncf = torch.cat((embed_u_gmf, embed_u_mlp), -1)
        # u_ncf_dom_label = self.user_NCF_classifier(emb_u_ncf, self.p)
        # Lucas 240708 changed GRL input layer
        concat_emb = torch.cat((output_gmf, output_mlp), -1)
        concat_emb = self.emb_fusion(concat_emb)    # lucas 240731added for feature selection
        concat_emb = self.moe_layer(concat_emb)
        # Lucas 240709 added feature selection layer, 效果没有提升
        # selected_concat_emb = self.feature_selector(concat_emb)

        # Domain classification task
        concat_emb_grl = self.grl(concat_emb)
        u_ncf_dom_label = self.grl_activation(self.domain_fc(concat_emb_grl))

        # # Lucas 240702 added for contrastive learning
        # emb_neg_u_ncf = torch.cat((embed_neg_u_gmf, embed_neg_u_mlp), -1)
        # neg_u_ncf_dom_label = self.user_NCF_classifier(emb_neg_u_ncf, self.p)

        # user_emb = self.user_embeddings(user)#.to(self.device_map[''])  # (batch_size, emsize)
        # item_emb = self.item_embeddings(item)#.to(self.device_map[''])  # (batch_size, emsize)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.user_embeddings(user_extend)
        # neg_i_emb = self.item_embeddings(neg_items)

        # lucas 240326 added for tune user/item embedding
        # user_emb = self.hidden_layers(user_emb)
        # item_emb = self.hidden_layers(item_emb)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.hidden_layers(neg_u_emb)
        # neg_i_emb = self.hidden_layers(neg_i_emb)
        # 230326 通过GRLayer GAN生成器判断source和target类别
        # p = float(batch_int + (epoch - dann_epoch) * num_batch / (args.epochs - dann_epoch) / num_batch)
        # p = 2. / (1. + np.exp(-10 * p)) - 1
        # predict_domain = self.user_semantic_classifier(user_emb, self.p)  # classifier，输入40维度，输出10维度,经过GRL再经过线性层+sigmoid输出2维分类
        # # Lucas 240702 added for contrastive learning
        # predict_neg_domain = self.user_semantic_classifier(neg_u_emb, self.p)

        # *****************************************training
        # Lucas 240327 added for rating prediction
        print(f"Lucas GMF size: {output_gmf.size()}; MLP size: {output_mlp.size()};")
        # concat_emb = torch.cat((output_gmf, output_mlp), -1)

        predict_ratings = self.predict_layer(concat_emb)
        # # Lucas 240702 added for contrastive learning
        # concat_neg_emb = torch.cat((output_neg_gmf, output_neg_mlp, neg_u_emb, neg_i_emb), -1)
        # predict_neg_ratings = self.predict_layer(concat_neg_emb)

        # # Lucas 240702 added contrastive learning
        # cl_pos_emb = self.contrastive_learn(concat_emb)
        # cl_neg_emb = self.contrastive_learn(concat_neg_emb)
        # cl_neg_emb = cl_neg_emb.view(user.shape[0], self.num_neg_i, -1)

        # ******************************************Loss
        # # Calculate InfoNCE loss ---contrastive learning loss
        # contrastive_loss = info_nce_loss(cl_pos_emb, cl_pos_emb, cl_neg_emb, self.temperature)
        # print(f"Lucas--contrastive_loss: {contrastive_loss};")

        rating_loss = self.rating_loss(predict_ratings.view(-1), rating)
        print(f"Lucas--predict_ratings: {predict_ratings}; original ratings: {rating}; rating loss: {rating_loss}")
        # print(f"Lucas--rating_loss: {rating_loss}")
        # return_loss_dict["scores"] = prediction.view(-1)

        # # Lucas 240702 added for contrastive learning
        # neg_rating_loss = self.rating_loss(predict_neg_ratings.view(-1), neg_rating)
        # print(f"Lucas--neg predict_ratings: {predict_neg_ratings}; original neg ratings: {neg_rating}; neg rating loss: {neg_rating_loss}")

        # GRL classification loss
        # pred_domains_loss = self.exp_loss_fn(predict_domain, domain)
        # print(f"Lucas--predict_domains: {predict_domain}; original domains: {domain}; prefix pre domain loss: {pred_domains_loss}")
        pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain}; ncf pre domain loss: {pred_ncf_domain_loss}")
        # # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        print(f"Lucas--rating loss: {rating_loss}; domain prediction loss: {pred_ncf_domain_loss}")
        # print(f"Lucas--rating loss device: {rating_loss.device}; domain loss dtype: {pred_ncf_domain_loss.device}; predict_ncf_domain loss: {pred_ncf_domain_loss.device}")

        total_loss = self.lambda1*rating_loss + self.lambda2*pred_ncf_domain_loss
        print(f"Lucas--final total_loss: {total_loss}")
        loss_detail = [rating_loss, pred_ncf_domain_loss]
        return total_loss, loss_detail, predict_ratings.view(-1)

# Lucas 240730 add MOE into NCF-GAN-CL for experiments
# NCF + GAN + Contrastive learning + MOE
class NCF_GAN_CL_MOE(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, moe_layer_num=3, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, lambda_grl=1.0, temperature=0.1, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_GAN_CL_MOE, self).__init__()
        self.device_map = device_map
        self.num_u = num_u
        self.num_i = num_i
        self.num_neg_i = num_neg_i    # added for contrastive learning
        self.temperature = temperature  # added for contrastive learning
        self.factor_num = factor_num
        self.ncf_layer_num = ncf_layer_num
        self.moe_layer_num = moe_layer_num  # added for moe layer
        self.drop_out = drop_out

        # self.peft_config = args[0]
        # self.model_config = config
        self.emb_size = emb_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda_grl = lambda_grl
        print(f"Lucas module num_u: {self.num_u}; num_i: {self.num_i}; lambda1: {self.lambda1}; lambda2: {self.lambda2}; lambda3: {self.lambda3}")
        print(f"Lucas module factor_num: {self.factor_num}")

        # print(f"Lucas--lucas_module--PeftPromptLlama2_V4--load_pretrained--device_map: {self.device_map}")

        # lucas 240402 NCF module
        self.embed_user_GMF = nn.Embedding(self.num_u, self.factor_num)
        self.embed_item_GMF = nn.Embedding(self.num_i, self.factor_num)
        # lucas 230327 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        self.embed_user_MLP = nn.Embedding(
            self.num_u, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.num_i, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        # layer1：256-128；layer2: 128-64; layer3: 64-32
        mlp_modules = []
        mlp_out_size = []
        for i in range(self.ncf_layer_num):
            input_size = self.factor_num * (2 ** (self.ncf_layer_num - i))
            mlp_modules.append(nn.Dropout(p=self.drop_out))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_out_size.append(input_size)
            mlp_out_size.append(input_size // 2)
            mlp_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*mlp_modules)

        # Lucas 240731 added feature integration
        self.emb_fusion = nn.Sequential(
            nn.Linear(self.factor_num + mlp_out_size[-1], self.factor_num + mlp_out_size[-1]),
            nn.Sigmoid(),
        )

        # Lucas 240731 added MOE layer
        self.moe_layers = nn.ModuleList()
        for i in range(len(self.moe_layer_num) - 1):
            self.moe_layers.append(MoELayer(self.moe_layer_num[i], self.moe_layer_num[i + 1], 5))

        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.grl = GRL(self.lambda_grl)  # Lucas 240708 changed grl
        print(f"Lucas GRL mlp_out size: {mlp_out_size}")
        # self.domain_fc = nn.Linear(self.factor_num + mlp_out_size[-1], 2)
        self.domain_fc = nn.Sequential(
            nn.Linear(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)), self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, 2)
        )
        self.grl_activation = nn.Sigmoid()

        # Lucas 240709 added for feature selection
        self.feature_selector = FeatureSelector(self.factor_num + mlp_out_size[-1])

        self.concatenate_len = 2  # added user and item embedding
        self.user_embeddings = nn.Embedding(self.num_u, self.emb_size)#.to(device_map[''])
        self.item_embeddings = nn.Embedding(self.num_i, self.emb_size)#.to(device_map[''])

        # Lucas 240326 added grl for user/item embeddings
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.emb_size),
        )
        # self.user_semantic_classifier = Classifier(self.emb_size)  # 768, 10

        # Lucas 240326 added 最后预测层
        self.rating_loss = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss()

        self.predict_emb = self.factor_num * 2 # + self.emb_size * 2
        print(f"Lucas predict_emb: {self.predict_emb}")
        # Lucas 240416 change the predict layer
        # self.predict_layer = nn.Linear(self.predict_emb, 1)
        dropout = 0.2
        self.activation = nn.Tanh()
        self.predict_layer = nn.Sequential(nn.Linear(self.predict_emb, self.predict_emb),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(self.predict_emb, self.factor_num),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(self.factor_num, 1))

        # Lucas 240701 added for Contrastive Learning
        self.contrastive_learn = nn.Linear(self.predict_emb, int(self.predict_emb))

        # ******************************lucas 240326 initialize the parameters of modules
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        # nn.init.xavier_uniform_(self.transform_matrix.weight)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         m.bias.data.zero_()

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        # Lucas 240416 change the prediction layer initialization
        # nn.init.kaiming_uniform_(self.predict_layer.weight,
        #                          a=1, nonlinearity='sigmoid')
        for layer in self.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

        print(f"Lucas finish initialize model---PrefixTune_NCF_GAN_Mixtral2")

    def forward(
            self,
            user: torch.LongTensor = None,
            item: torch.LongTensor = None,
            neg_items: torch.LongTensor = None,
            rating: torch.LongTensor = None,
            neg_rating: torch.LongTensor = None,
            domain: torch.LongTensor = None,
            # neg_domain: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if rating is not None:
            rating = rating.to(torch.float32)
        if neg_rating is not None:
            neg_rating = neg_rating.to(torch.float32)
        if domain is not None:
            domain = domain.to(torch.long)
        # if neg_domain is not None:
        #     neg_domain = neg_domain.to(torch.long)
        print(f"Lucas model forward-----user: {user};----item:{item};----rating: {rating};----domain: {domain}")
        print(f"Lucas model forward-----neg_items: {neg_items};----neg_rating:{neg_rating};")

        embed_u_gmf = self.embed_user_GMF(user)
        embed_i_gmf = self.embed_item_GMF(item)
        output_gmf = embed_u_gmf * embed_i_gmf
        # Lucas 240702 added for contrastive learning
        user_extend = user.unsqueeze(1).repeat(1, self.num_neg_i).view(-1)
        print(f"Lucas model forward-----extend user: {user_extend};")
        embed_neg_u_gmf = self.embed_user_GMF(user_extend)
        embed_neg_i_gmf = self.embed_item_GMF(neg_items)
        output_neg_gmf = embed_neg_u_gmf * embed_neg_i_gmf

        embed_u_mlp = self.embed_user_MLP(user)
        embed_i_mlp = self.embed_item_MLP(item)
        interaction = torch.cat((embed_u_mlp, embed_i_mlp), -1)
        output_mlp = self.MLP_layers(interaction)
        # Lucas 240702 added for contrastive learning
        embed_neg_u_mlp = self.embed_user_MLP(user_extend)
        embed_neg_i_mlp = self.embed_item_MLP(neg_items)
        neg_inter = torch.cat((embed_neg_u_mlp, embed_neg_i_mlp), -1)
        output_neg_mlp = self.MLP_layers(neg_inter)

        # lucas 240402 added GRL layer for user embeddings in NCF
        # emb_u_ncf = torch.cat((embed_u_gmf, embed_u_mlp), -1)
        # u_ncf_dom_label = self.user_NCF_classifier(emb_u_ncf, self.p)
        # Lucas 240708 changed GRL input layer
        concat_emb = torch.cat((output_gmf, output_mlp), -1)
        concat_emb = self.emb_fusion(concat_emb)    # lucas 240731added for feature selection


        # Lucas 240709 added feature selection layer, 效果没有提升
        # selected_concat_emb = self.feature_selector(concat_emb)

        # Domain classification task
        # Lucas 240731 change the grl embedding, only use user embedding
        # concat_emb_grl = self.grl(concat_emb)
        concat_u_emb = torch.cat((output_gmf, embed_u_mlp), -1)
        concat_u_emb = self.grl(concat_u_emb)
        u_ncf_dom_label = self.grl_activation(self.domain_fc(concat_u_emb))
        # # Lucas 240702 added for contrastive learning
        # emb_neg_u_ncf = torch.cat((embed_neg_u_gmf, embed_neg_u_mlp), -1)
        # neg_u_ncf_dom_label = self.user_NCF_classifier(emb_neg_u_ncf, self.p)

        # user_emb = self.user_embeddings(user)#.to(self.device_map[''])  # (batch_size, emsize)
        # item_emb = self.item_embeddings(item)#.to(self.device_map[''])  # (batch_size, emsize)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.user_embeddings(user_extend)
        # neg_i_emb = self.item_embeddings(neg_items)
        #
        # # lucas 240326 added for tune user/item embedding
        # user_emb = self.hidden_layers(user_emb)
        # item_emb = self.hidden_layers(item_emb)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.hidden_layers(neg_u_emb)
        # neg_i_emb = self.hidden_layers(neg_i_emb)
        # 230326 通过GRLayer GAN生成器判断source和target类别
        # p = float(batch_int + (epoch - dann_epoch) * num_batch / (args.epochs - dann_epoch) / num_batch)
        # p = 2. / (1. + np.exp(-10 * p)) - 1
        # predict_domain = self.user_semantic_classifier(user_emb, self.p)  # classifier，输入40维度，输出10维度,经过GRL再经过线性层+sigmoid输出2维分类
        # Lucas 240702 added for contrastive learning
        # predict_neg_domain = self.user_semantic_classifier(neg_u_emb, self.p)

        # *****************************************training
        # Lucas 240327 added for rating prediction
        print(f"Lucas GMF size: {output_gmf.size()}; MLP size: {output_mlp.size()};")
        predict_ratings = self.predict_layer(concat_emb)
        # Lucas 240702 added for contrastive learning
        concat_neg_emb = torch.cat((output_neg_gmf, output_neg_mlp), -1)
        concat_neg_emb = self.emb_fusion(concat_neg_emb)  # lucas 240731added for feature
        predict_neg_ratings = self.predict_layer(concat_neg_emb)

        # Lucas 240702 added contrastive learning
        cl_pos_emb = self.contrastive_learn(concat_emb)
        cl_neg_emb = self.contrastive_learn(concat_neg_emb)
        cl_neg_emb = cl_neg_emb.view(user.shape[0], self.num_neg_i, -1)

        # ******************************************Loss
        # Calculate InfoNCE loss ---contrastive learning loss
        contrastive_loss = info_nce_loss(cl_pos_emb, cl_pos_emb, cl_neg_emb, self.temperature)
        print(f"Lucas--contrastive_loss: {contrastive_loss};")

        rating_loss = self.rating_loss(predict_ratings.view(-1), rating)
        print(f"Lucas--predict_ratings: {predict_ratings}; original ratings: {rating}; rating loss: {rating_loss}")
        # print(f"Lucas--rating_loss: {rating_loss}")
        # return_loss_dict["scores"] = prediction.view(-1)

        # Lucas 240702 added for contrastive learning
        neg_rating_loss = self.rating_loss(predict_neg_ratings.view(-1), neg_rating)
        print(f"Lucas--neg predict_ratings: {predict_neg_ratings}; original neg ratings: {neg_rating}; neg rating loss: {neg_rating_loss}")

        # GRL classification loss
        # pred_domains_loss = self.exp_loss_fn(predict_domain, domain)
        # print(f"Lucas--predict_domains: {predict_domain}; original domains: {domain}; prefix pre domain loss: {pred_domains_loss}")
        pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain}; ncf pre domain loss: {pred_ncf_domain_loss}")
        # # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        # print(f"Lucas--rating loss: {rating_loss}; domain prediction loss: {pred_ncf_domain_loss}")
        # print(f"Lucas--rating loss device: {rating_loss.device}; domain loss dtype: {pred_ncf_domain_loss.device}; predict_ncf_domain loss: {pred_ncf_domain_loss.device}")

        total_loss = self.lambda1*rating_loss + self.lambda2*contrastive_loss + self.lambda3*pred_ncf_domain_loss # + self.lambda1*neg_rating_loss
        print(f"Lucas--final total_loss: {total_loss}")
        loss_detail = [rating_loss, neg_rating_loss, contrastive_loss, pred_ncf_domain_loss]
        return total_loss, loss_detail, predict_ratings.view(-1)


class LNLayer(nn.Module):
    def __init__(self, hidden_size1, hidden_size, layer_norm_eps):
        super().__init__()
        self.dense = nn.Linear(hidden_size1, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        # self.dense2 = nn.Linear(hidden_size * 2, hidden_size)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.LayerNorm = nn.BatchNorm1d(hidden_size)
        # self.act1 = nn.ReLU()

    def forward(self, hidden_states, input_tensor):
        hidden_out = self.dense(hidden_states)
        # hidden_out = self.act1(hidden_out)
        # hidden_out = self.dense2(hidden_out)
        # hidden_states = self.dropout(hidden_states)
        # input_tensor = self.dense2(input_tensor)
        # input_tensor = self.dropout(input_tensor)
        hidden_out = self.LayerNorm(hidden_out + input_tensor)

        return hidden_out

# 240902 NCF GAN SenBert
class NCF_GAN_SenBert(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, lambda_grl=1.0, temperature=0.1, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_GAN_SenBert, self).__init__()
        self.device_map = device_map
        self.num_u = num_u
        self.num_i = num_i
        self.num_neg_i = num_neg_i    # added for contrastive learning
        self.temperature = temperature  # added for contrastive learning
        self.factor_num = factor_num
        self.ncf_layer_num = ncf_layer_num
        self.drop_out = drop_out

        # self.peft_config = args[0]
        # self.model_config = config
        self.emb_size = emb_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda_grl = lambda_grl
        print(f"Lucas module num_u: {self.num_u}; num_i: {self.num_i}; lambda1: {self.lambda1}; lambda2: {self.lambda2}; lambda3: {self.lambda3}")
        print(f"Lucas module factor_num: {self.factor_num}")

        # print(f"Lucas--lucas_module--PeftPromptLlama2_V4--load_pretrained--device_map: {self.device_map}")

        # lucas 240402 NCF module
        self.embed_user_GMF = nn.Embedding(self.num_u, self.factor_num)
        self.embed_item_GMF = nn.Embedding(self.num_i, self.factor_num)
        # lucas 230327 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        self.embed_user_MLP = nn.Embedding(
            self.num_u, self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.num_i, self.factor_num * (2 ** (self.ncf_layer_num - 1)))

        # nn.Sigmoid()
        # layer1：256-128；layer2: 128-64; layer3: 64-32
        mlp_modules = []
        mlp_out_size = []
        for i in range(self.ncf_layer_num):
            input_size = self.factor_num * (2 ** (self.ncf_layer_num - i))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.Dropout(p=self.drop_out))
            mlp_modules.append(nn.BatchNorm1d(input_size // 2))
            mlp_out_size.append(input_size)
            mlp_out_size.append(input_size // 2)
            mlp_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*mlp_modules)

        # Lucas 240731 added for grl and cl
        self.emb_fusion = nn.Sequential(
            nn.Linear(self.factor_num + mlp_out_size[-1], self.factor_num + mlp_out_size[-1]),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num + mlp_out_size[-1]),
            nn.Sigmoid(),
        )

        # lucas 240402 添加user跨域之后GAN中的判别器，判断是source域还是target域, 暂时没用到
        # self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)))
        self.grl = GRL(self.lambda_grl)  # Lucas 240708 changed grl
        print(f"Lucas GRL mlp_out size: {mlp_out_size}")
        # self.domain_fc = nn.Linear(self.factor_num + mlp_out_size[-1], 2)
        self.domain_fc = nn.Sequential(
            # nn.Linear(self.factor_num + self.factor_num * (2 ** (self.ncf_layer_num - 1)), self.factor_num),
            nn.Linear(self.factor_num + mlp_out_size[-1], self.factor_num),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, self.factor_num),
            nn.Dropout(p=self.drop_out),
            nn.BatchNorm1d(self.factor_num),
            nn.ReLU(),
            nn.Linear(self.factor_num, 2)
        )
        self.grl_activation = nn.Sigmoid()

        # # Lucas 240709 added for feature selection
        # self.feature_selector = FeatureSelector(self.factor_num + mlp_out_size[-1])

        self.concatenate_len = 2  # added user and item embedding
        self.user_embeddings = nn.Embedding(self.num_u, self.emb_size)#.to(device_map[''])
        self.item_embeddings = nn.Embedding(self.num_i, self.emb_size)#.to(device_map[''])

        # Lucas 240326 added grl for user/item embeddings
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.emb_size),
            # nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.emb_size),
            # nn.LayerNorm(self.emb_size),
        )


        self.sen_bert = SentenceTransformer('all-MiniLM-L6-v2')
        self.hidden_size1 = 384
        self.user_semantic_classifier = Classifier(self.hidden_size1)  # 768, 10
        self.output_LN = LNLayer(self.hidden_size1, self.hidden_size1, 1e-12)
        self.embed_user_sem = nn.Embedding(self.num_u, self.hidden_size1)
        self.embed_item_sem = nn.Embedding(self.num_i, self.hidden_size1)

        # Fusion layers
        self.user_fusion = nn.Linear(self.hidden_size1 * 2, self.hidden_size1)
        self.item_fusion = nn.Linear(self.hidden_size1 * 2, self.hidden_size1)
        # Lucas 240326 added 最后预测层
        # self.rating_loss = nn.MSELoss()
        self.rating_loss = nn.SmoothL1Loss()
        self.exp_loss_fn = nn.CrossEntropyLoss()


        # self.predict_emb = self.factor_num * 2 # + self.emb_size * 2
        self.predict_emb = self.factor_num + mlp_out_size[-1] + self.hidden_size1
        print(f"Lucas predict_emb: {self.predict_emb}")
        # Lucas 240416 change the predict layer
        # self.predict_layer = nn.Linear(self.predict_emb, 1)
        dropout = 0.2
        self.activation = nn.Tanh()
        self.predict_layer = nn.Sequential(nn.Linear(self.predict_emb, self.predict_emb),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.BatchNorm1d(self.predict_emb),
                                           nn.Linear(self.predict_emb, self.factor_num),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(self.factor_num, 1))

        # # Lucas 240701 added for Contrastive Learning
        # self.contrastive_learn = nn.Linear(self.predict_emb, int(self.predict_emb / 2))

        # ******************************lucas 240326 initialize the parameters of modules
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        # nn.init.xavier_uniform_(self.transform_matrix.weight)
        initrange = 0.1
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        for m in self.emb_fusion:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        for m in self.domain_fc:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-initrange, initrange)
                m.bias.data.zero_()
        # for m in self.modules():
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         m.bias.data.zero_()

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
        # Lucas 240416 change the prediction layer initialization
        # nn.init.kaiming_uniform_(self.predict_layer.weight,
        #                          a=1, nonlinearity='sigmoid')
        for layer in self.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

        print(f"Lucas finish initialize model---PrefixTune_NCF_GAN_Mixtral2")

    def forward(
            self,
            p,
            user: torch.LongTensor = None,
            item: torch.LongTensor = None,
            rating: torch.LongTensor = None,
            domain: torch.LongTensor = None,
            sentences: list = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if rating is not None:
            rating = rating.to(torch.float32)
        if domain is not None:
            domain = domain.to(torch.long)
        # print(f"Lucas model forward-----user: {user};----item:{item};----rating: {rating};----domain: {domain}")

        embed_u_gmf = self.embed_user_GMF(user)
        embed_i_gmf = self.embed_item_GMF(item)
        output_gmf = embed_u_gmf * embed_i_gmf
        # # Lucas 240702 added for contrastive learning
        # user_extend = user.unsqueeze(1).repeat(1, self.num_neg_i).view(-1)
        # print(f"Lucas model forward-----extend user: {user_extend};")
        # embed_neg_u_gmf = self.embed_user_GMF(user_extend)
        # embed_neg_i_gmf = self.embed_item_GMF(neg_items)
        # output_neg_gmf = embed_neg_u_gmf * embed_neg_i_gmf

        embed_u_mlp = self.embed_user_MLP(user)
        embed_i_mlp = self.embed_item_MLP(item)
        interaction = torch.cat((embed_u_mlp, embed_i_mlp), -1)
        output_mlp = self.MLP_layers(interaction)
        # # Lucas 240702 added for contrastive learning
        # embed_neg_u_mlp = self.embed_user_MLP(user_extend)
        # embed_neg_i_mlp = self.embed_item_MLP(neg_items)
        # neg_inter = torch.cat((embed_neg_u_mlp, embed_neg_i_mlp), -1)
        # output_neg_mlp = self.MLP_layers(neg_inter)

        # lucas 240402 added GRL layer for user embeddings in NCF
        # emb_u_ncf = torch.cat((embed_u_gmf, embed_u_mlp), -1)
        # u_ncf_dom_label = self.user_NCF_classifier(emb_u_ncf, self.p)
        # Lucas 240708 changed GRL input layer
        concat_emb = torch.cat((output_gmf, output_mlp), -1)
        concat_emb = self.emb_fusion(concat_emb)    # lucas 240731added for feature selection

        # Domain classification task
        concat_emb_grl = self.grl(concat_emb)
        u_ncf_dom_label = self.grl_activation(self.domain_fc(concat_emb_grl))

        # Lucas 240709 added feature selection layer, 效果没有提升
        # selected_concat_emb = self.feature_selector(concat_emb)

        embed_sent_BERT = self.sen_bert.encode(sentences, convert_to_tensor=True)
        # user_sent_emb = (self.embed_user_sem(user) + embed_sent_BERT)/2
        # item_sent_emb = (self.emdfbed_item_sem(item) + embed_sent_BERT)/2
        user_sent_emb = self.user_fusion(torch.cat([self.embed_user_sem(user), embed_sent_BERT], dim=-1))
        item_sent_emb = self.item_fusion(torch.cat([self.embed_item_sem(user), embed_sent_BERT], dim=-1))
        concat_sent_emb = user_sent_emb * item_sent_emb
        # embed_user_sBERT_LN = self.output_LN(embed_sent_BERT, self.embed_user_sem(user))
        # # 230326 通过GRLayer GAN生成器判断source和target类别
        # user_sem_dom_label = self.user_semantic_classifier(embed_user_sBERT_LN,p)  # classifier，输入40维度，输出10维度,经过GRL再经过线性层+sigmoid输出2维分类
        # # item_sem_label = self.item_classifier(item_embeds_s, p)
        # # embed_item_BERT = self.output_LN(dense_output['sentence_embedding'], self.embed_item_BERT(item))
        # embed_item_sBERT_LN = self.output_LN(embed_sent_BERT, self.embed_item_sem(item))

        # # interaction_BERT = torch.cat((embed_user_BERT, embed_item_BERT), -1)
        # # 改为element-wise product
        # interaction_BERT = embed_user_sBERT_LN * embed_item_sBERT_LN

        concat = torch.cat((concat_emb, embed_sent_BERT), -1)

        # # Lucas 240702 added for contrastive learning
        # emb_neg_u_ncf = torch.cat((embed_neg_u_gmf, embed_neg_u_mlp), -1)
        # neg_u_ncf_dom_label = self.user_NCF_classifier(emb_neg_u_ncf, self.p)

        # user_emb = self.user_embeddings(user)#.to(self.device_map[''])  # (batch_size, emsize)
        # item_emb = self.item_embeddings(item)#.to(self.device_map[''])  # (batch_size, emsize)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.user_embeddings(user_extend)
        # neg_i_emb = self.item_embeddings(neg_items)

        # lucas 240326 added for tune user/item embedding
        # user_emb = self.hidden_layers(user_emb)
        # item_emb = self.hidden_layers(item_emb)
        # # Lucas 240702 added for contrastive learning
        # neg_u_emb = self.hidden_layers(neg_u_emb)
        # neg_i_emb = self.hidden_layers(neg_i_emb)
        # 230326 通过GRLayer GAN生成器判断source和target类别
        # p = float(batch_int + (epoch - dann_epoch) * num_batch / (args.epochs - dann_epoch) / num_batch)
        # p = 2. / (1. + np.exp(-10 * p)) - 1
        # predict_domain = self.user_semantic_classifier(user_emb, self.p)  # classifier，输入40维度，输出10维度,经过GRL再经过线性层+sigmoid输出2维分类
        # # Lucas 240702 added for contrastive learning
        # predict_neg_domain = self.user_semantic_classifier(neg_u_emb, self.p)

        # *****************************************training
        # Lucas 240327 added for rating prediction
        print(f"Lucas GMF size: {output_gmf.size()}; MLP size: {output_mlp.size()};")
        # concat_emb = torch.cat((output_gmf, output_mlp), -1)
        predict_ratings = self.predict_layer(concat)
        # # Lucas 240702 added for contrastive learning
        # concat_neg_emb = torch.cat((output_neg_gmf, output_neg_mlp, neg_u_emb, neg_i_emb), -1)
        # predict_neg_ratings = self.predict_layer(concat_neg_emb)

        # # Lucas 240702 added contrastive learning
        # cl_pos_emb = self.contrastive_learn(concat_emb)
        # cl_neg_emb = self.contrastive_learn(concat_neg_emb)
        # cl_neg_emb = cl_neg_emb.view(user.shape[0], self.num_neg_i, -1)

        # ******************************************Loss
        # # Calculate InfoNCE loss ---contrastive learning loss
        # contrastive_loss = info_nce_loss(cl_pos_emb, cl_pos_emb, cl_neg_emb, self.temperature)
        # print(f"Lucas--contrastive_loss: {contrastive_loss};")

        rating_loss = self.rating_loss(predict_ratings.view(-1), rating)
        # print(f"Lucas--predict_ratings: {predict_ratings}; original ratings: {rating}; rating loss: {rating_loss}")
        # print(f"Lucas--rating_loss: {rating_loss}")
        # return_loss_dict["scores"] = prediction.view(-1)

        # # Lucas 240702 added for contrastive learning
        # neg_rating_loss = self.rating_loss(predict_neg_ratings.view(-1), neg_rating)
        # print(f"Lucas--neg predict_ratings: {predict_neg_ratings}; original neg ratings: {neg_rating}; neg rating loss: {neg_rating_loss}")

        # GRL classification loss
        # pred_domains_loss = self.exp_loss_fn(predict_domain, domain)
        # print(f"Lucas--predict_domains: {predict_domain}; original domains: {domain}; prefix pre domain loss: {pred_domains_loss}")
        pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        # pred_sem_domain_loss = self.exp_loss_fn(user_sem_dom_label, domain)
        # print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain}; ncf pre domain loss: {pred_ncf_domain_loss}")
        # print(f"Lucas--predict semantic domains: {user_sem_dom_label}; original ncf domains: {domain}; semantic pre domain loss: {pred_sem_domain_loss}")

        # # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        # print(f"Lucas--rating loss: {rating_loss}; NCF domain prediction loss: {pred_ncf_domain_loss}; semantic domain prediction loss: {pred_sem_domain_loss}")
        print(f"Lucas--rating loss: {rating_loss}; NCF domain prediction loss: {pred_ncf_domain_loss};")

        # print(f"Lucas--rating loss device: {rating_loss.device}; domain loss dtype: {pred_ncf_domain_loss.device}; predict_ncf_domain loss: {pred_ncf_domain_loss.device}")

        total_loss = self.lambda1*rating_loss + self.lambda2*pred_ncf_domain_loss # + self.lambda2*pred_sem_domain_loss
        print(f"Lucas--final total_loss: {total_loss}")
        # loss_detail = [rating_loss, pred_ncf_domain_loss, pred_sem_domain_loss]
        loss_detail = [rating_loss, pred_ncf_domain_loss, 0]
        return total_loss, loss_detail, predict_ratings.view(-1)

