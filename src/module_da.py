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
        embeds_revsers = ReverseLayerF.apply(input_data, p) #lucas疑问？gradient reverse layer?
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

# 4. MMD Loss Implementation
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    L2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd_loss(source, target):
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    print(f"XX size:{XX.shape}, yy size:{YY.shape}; XY size:{XY.shape}; YX size:{YX.shape}")
    loss = torch.mean(XX + YY - XY - YX)
    return loss

class MDDLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MDDLoss, self).__init__()
        self.margin = margin

    def forward(self, source_preds, target_preds):
        # Compute MDD loss
        mdd_loss = torch.mean(F.relu(self.margin - (source_preds - target_preds)))
        return mdd_loss

# MMD
class NCF_MMD(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, lambda_grl=1.0, temperature=0.1, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_MMD, self).__init__()
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

    def get_feature(self, user, item):
        user_mlp_emb = self.embed_user_MLP(user)
        item_mlp_emb = self.embed_item_MLP(item)
        user_gmf_emb = self.embed_user_GMF(user)
        item_gmf_emb = self.embed_item_GMF(item)
        return torch.cat([user_mlp_emb, item_mlp_emb, user_gmf_emb, item_gmf_emb], dim=-1)

    # 240825 added mmd loss
    def compute_mmd_loss(self, source_user, source_item, target_user, target_item):
        source_features = self.get_feature(source_user, source_item)
        target_features = self.get_feature(target_user, target_item)
        return mmd_loss(source_features, target_features)

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
        # concat_emb_grl = self.grl(concat_emb)
        # u_ncf_dom_label = self.grl_activation(self.domain_fc(concat_emb_grl))

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
        # pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        # print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain}; ncf pre domain loss: {pred_ncf_domain_loss}")
        # # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        # print(f"Lucas--rating loss: {rating_loss}; domain prediction loss: {pred_ncf_domain_loss}")
        # print(f"Lucas--rating loss device: {rating_loss.device}; domain loss dtype: {pred_ncf_domain_loss.device}; predict_ncf_domain loss: {pred_ncf_domain_loss.device}")

        total_loss = self.lambda1*rating_loss # + self.lambda2*pred_ncf_domain_loss
        # print(f"Lucas--final total_loss: {total_loss}")
        # loss_detail = [rating_loss]
        return total_loss, predict_ratings.view(-1)


class NCF_MDD(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, lambda_grl=1.0, temperature=0.1, margin=0.5, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_MDD, self).__init__()
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
                                           nn.Linear(self.factor_num, 1))
        self.mdd_loss = MDDLoss(margin)
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

    def get_feature(self, user, item):
        user_mlp_emb = self.embed_user_MLP(user)
        item_mlp_emb = self.embed_item_MLP(item)
        user_gmf_emb = self.embed_user_GMF(user)
        item_gmf_emb = self.embed_item_GMF(item)
        return torch.cat([user_mlp_emb, item_mlp_emb, user_gmf_emb, item_gmf_emb], dim=-1)

    # 240825 added mmd loss
    def compute_mmd_loss(self, source_user, source_item, target_user, target_item):
        source_features = self.get_feature(source_user, source_item)
        target_features = self.get_feature(target_user, target_item)
        return mmd_loss(source_features, target_features)

    def compute_mdd_loss(self, source_user, source_item, s_rating, s_domain, target_user, target_item, t_rating, t_domain):
        s_rat_loss, source_preds = self.forward(source_user, source_item, s_rating, s_domain)
        t_rat_loss, target_preds = self.forward(target_user, target_item, t_rating, t_domain)
        return self.mdd_loss(source_preds, target_preds)

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
        # concat_emb_grl = self.grl(concat_emb)
        # u_ncf_dom_label = self.grl_activation(self.domain_fc(concat_emb_grl))

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
        # pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        # print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain}; ncf pre domain loss: {pred_ncf_domain_loss}")
        # # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        # print(f"Lucas--rating loss: {rating_loss}; domain prediction loss: {pred_ncf_domain_loss}")
        # print(f"Lucas--rating loss device: {rating_loss.device}; domain loss dtype: {pred_ncf_domain_loss.device}; predict_ncf_domain loss: {pred_ncf_domain_loss.device}")

        total_loss = self.lambda1*rating_loss # + self.lambda2*pred_ncf_domain_loss
        # print(f"Lucas--final total_loss: {total_loss}")
        # loss_detail = [rating_loss]
        return total_loss, predict_ratings.view(-1)


class NCF(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, lambda_grl=1.0, temperature=0.1, margin=0.5, **kwargs):
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
                                           nn.Linear(self.factor_num, 1))
        self.mdd_loss = MDDLoss(margin)
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

    def get_feature(self, user, item):
        user_mlp_emb = self.embed_user_MLP(user)
        item_mlp_emb = self.embed_item_MLP(item)
        user_gmf_emb = self.embed_user_GMF(user)
        item_gmf_emb = self.embed_item_GMF(item)
        return torch.cat([user_mlp_emb, item_mlp_emb, user_gmf_emb, item_gmf_emb], dim=-1)

    # 240825 added mmd loss
    def compute_mmd_loss(self, source_user, source_item, target_user, target_item):
        source_features = self.get_feature(source_user, source_item)
        target_features = self.get_feature(target_user, target_item)
        return mmd_loss(source_features, target_features)

    def compute_mdd_loss(self, source_user, source_item, s_rating, s_domain, target_user, target_item, t_rating, t_domain):
        s_rat_loss, source_preds = self.forward(source_user, source_item, s_rating, s_domain)
        t_rat_loss, target_preds = self.forward(target_user, target_item, t_rating, t_domain)
        return self.mdd_loss(source_preds, target_preds)

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
        # concat_emb_grl = self.grl(concat_emb)
        # u_ncf_dom_label = self.grl_activation(self.domain_fc(concat_emb_grl))

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
        # pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        # print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain}; ncf pre domain loss: {pred_ncf_domain_loss}")
        # # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        # print(f"Lucas--rating loss: {rating_loss}; domain prediction loss: {pred_ncf_domain_loss}")
        # print(f"Lucas--rating loss device: {rating_loss.device}; domain loss dtype: {pred_ncf_domain_loss.device}; predict_ncf_domain loss: {pred_ncf_domain_loss.device}")

        total_loss = self.lambda1*rating_loss # + self.lambda2*pred_ncf_domain_loss
        # print(f"Lucas--final total_loss: {total_loss}")
        # loss_detail = [rating_loss]
        return total_loss, predict_ratings.view(-1)

# 4. MCD Loss Implementation
class MCDLoss(nn.Module):
    def __init__(self):
        super(MCDLoss, self).__init__()

    def forward(self, preds_1, preds_2):
        # Compute the L1 discrepancy between the two classifiers' predictions
        discrepancy = torch.mean(torch.abs(preds_1 - preds_2))
        return discrepancy

class NCF_MCD(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None,
                 factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0,
                 lambda4=1.0, lambda_grl=1.0, temperature=0.1, margin=0.5, **kwargs):
    # def __init__(self, num_users, num_items, embed_dim=32):
        super(NCF_MCD, self).__init__()


        self.ncf_1 = NCF(emb_size=emb_size, device_map=device_map, num_u=num_u, num_i=num_i, num_neg_i=num_neg_i, factor_num=factor_num,
                         ncf_layer_num=ncf_layer_num, drop_out=drop_out, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, lambda4=lambda4, lambda_grl=lambda_grl,
                         temperature=temperature, margin=margin)
        self.ncf_2 = NCF(emb_size=emb_size, device_map=device_map, num_u=num_u, num_i=num_i, num_neg_i=num_neg_i, factor_num=factor_num,
                         ncf_layer_num=ncf_layer_num, drop_out=drop_out, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, lambda4=lambda4, lambda_grl=lambda_grl,
                         temperature=temperature, margin=margin)
        self.mcd_loss = MCDLoss()

    def forward(self, user, item, rating, domain):
        # Predict using both classifiers
        ncf_1_loss, preds_1 = self.ncf_1(user, item, rating, domain)
        ncf_2_loss, preds_2 = self.ncf_2(user, item, rating, domain)
        return preds_1, preds_2, ncf_1_loss, ncf_2_loss

    def compute_mcd_loss(self, source_user, source_item, s_rating, s_domain, target_user, target_item, t_rating, t_domain):
        # Compute predictions for source and target domains
        source_preds_1, source_preds_2, s_loss_1, s_loss_2 = self.forward(source_user, source_item, s_rating, s_domain)
        target_preds_1, target_preds_2, t_loss_1, t_loss_2 = self.forward(target_user, target_item, t_rating, t_domain)

        # Compute discrepancy on target and source domains
        source_discrepancy = self.mcd_loss(source_preds_1, source_preds_2)
        target_discrepancy = self.mcd_loss(target_preds_1, target_preds_2)

        return source_discrepancy, target_discrepancy


class DSN(nn.Module):
    def __init__(self, input_dim, private_dim, shared_dim):
        super(DSN, self).__init__()
        self.private_encoder = nn.Sequential(
            nn.Linear(input_dim, private_dim),
            nn.ReLU(),
            nn.Linear(private_dim, private_dim)
        )
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(private_dim + shared_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, x):
        private = self.private_encoder(x)
        shared = self.shared_encoder(x)
        reconstructed = self.decoder(torch.cat([private, shared], dim=-1))
        return private, shared, reconstructed


# Combined DSN + NCF Model
class NCF_DSN(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, lambda_grl=1.0, temperature=0.1, margin=0.5, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_DSN, self).__init__()
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
                                           nn.Linear(self.factor_num, 1))
        self.dsn = DSN(self.predict_emb, self.factor_num, self.factor_num)
        self.mdd_loss = MDDLoss(margin)
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

    def get_feature(self, user, item):
        user_mlp_emb = self.embed_user_MLP(user)
        item_mlp_emb = self.embed_item_MLP(item)
        user_gmf_emb = self.embed_user_GMF(user)
        item_gmf_emb = self.embed_item_GMF(item)
        return torch.cat([user_mlp_emb, item_mlp_emb, user_gmf_emb, item_gmf_emb], dim=-1)

    # 240825 added mmd loss
    def compute_mmd_loss(self, source_user, source_item, target_user, target_item):
        source_features = self.get_feature(source_user, source_item)
        target_features = self.get_feature(target_user, target_item)
        return mmd_loss(source_features, target_features)

    def compute_mdd_loss(self, source_user, source_item, s_rating, s_domain, target_user, target_item, t_rating, t_domain):
        s_rat_loss, source_preds = self.forward(source_user, source_item, s_rating, s_domain)
        t_rat_loss, target_preds = self.forward(target_user, target_item, t_rating, t_domain)
        return self.mdd_loss(source_preds, target_preds)

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
        private_emb, shared_emb, concat_emb = self.dsn(concat_emb)   # 240826
        # Lucas 240709 added feature selection layer, 效果没有提升
        # selected_concat_emb = self.feature_selector(concat_emb)

        # Domain classification task
        # concat_emb_grl = self.grl(concat_emb)
        # u_ncf_dom_label = self.grl_activation(self.domain_fc(concat_emb_grl))

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
        # pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        # print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain}; ncf pre domain loss: {pred_ncf_domain_loss}")
        # # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        # print(f"Lucas--rating loss: {rating_loss}; domain prediction loss: {pred_ncf_domain_loss}")
        # print(f"Lucas--rating loss device: {rating_loss.device}; domain loss dtype: {pred_ncf_domain_loss.device}; predict_ncf_domain loss: {pred_ncf_domain_loss.device}")

        total_loss = self.lambda1*rating_loss # + self.lambda2*pred_ncf_domain_loss
        # print(f"Lucas--final total_loss: {total_loss}")
        # loss_detail = [rating_loss]
        return total_loss, predict_ratings.view(-1)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(DomainDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


class CDANLoss(nn.Module):
    def __init__(self, discriminator):
        super(CDANLoss, self).__init__()
        self.discriminator = discriminator

    def forward(self, features, predictions):
        print(f"features: {features}; predictions: {predictions}")
        # Combine features with classifier predictions for conditioning
        conditioned_features = features * predictions.unsqueeze(-1)
        # conditioned_features = features * predictions

        # Pass conditioned features through domain discriminator
        domain_pred = self.discriminator(conditioned_features)

        return domain_pred

class NCF_CDAN(nn.Module):
    def __init__(self, *args, emb_size=128, device_map=None, num_u=None, num_i=None, num_neg_i=None, factor_num=None, ncf_layer_num=None, drop_out=None, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, lambda_grl=1.0, temperature=0.1, margin=0.5, **kwargs):
        # super().__init__(config, *args, **kwargs)
        super(NCF_CDAN, self).__init__()
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
                                           nn.Linear(self.factor_num, 1))
        self.dsn = DSN(self.predict_emb, self.factor_num, self.factor_num)
        self.mdd_loss = MDDLoss(margin)
        self.domain_discriminator = DomainDiscriminator(self.predict_emb)
        self.cdan_loss = CDANLoss(self.domain_discriminator)
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

    def get_feature(self, user, item):
        user_mlp_emb = self.embed_user_MLP(user)
        item_mlp_emb = self.embed_item_MLP(item)
        user_gmf_emb = self.embed_user_GMF(user)
        item_gmf_emb = self.embed_item_GMF(item)
        return torch.cat([user_mlp_emb, item_mlp_emb, user_gmf_emb, item_gmf_emb], dim=-1)

    # 240825 added mmd loss
    def compute_mmd_loss(self, source_user, source_item, target_user, target_item):
        source_features = self.get_feature(source_user, source_item)
        target_features = self.get_feature(target_user, target_item)
        return mmd_loss(source_features, target_features)

    def compute_mdd_loss(self, source_user, source_item, s_rating, s_domain, target_user, target_item, t_rating, t_domain):
        s_rat_loss, source_preds = self.forward(source_user, source_item, s_rating, s_domain)
        t_rat_loss, target_preds = self.forward(target_user, target_item, t_rating, t_domain)
        return self.mdd_loss(source_preds, target_preds)

    def get_mlp_gmf_features(self, user, item):
        embed_u_gmf = self.embed_user_GMF(user)
        embed_i_gmf = self.embed_item_GMF(item)
        output_gmf = embed_u_gmf * embed_i_gmf

        embed_u_mlp = self.embed_user_MLP(user)
        embed_i_mlp = self.embed_item_MLP(item)
        interaction = torch.cat((embed_u_mlp, embed_i_mlp), -1)
        output_mlp = self.MLP_layers(interaction)

        concat_emb = torch.cat((output_gmf, output_mlp), -1)
        concat_emb = self.emb_fusion(concat_emb)
        return concat_emb

    def compute_cdan_loss(self, source_user, source_item, s_rating, s_domain, target_user, target_item, t_rating, t_domain):
        # Get features and predictions for source and target domains
        source_features = self.get_mlp_gmf_features(source_user, source_item)
        s_pred_loss, source_preds = self.forward(source_user, source_item, s_rating, s_domain)

        target_features = self.get_mlp_gmf_features(target_user, target_item)
        t_pred_loss, target_preds = self.forward(target_user, target_item, t_rating, t_domain)

        # Compute domain predictions for both domains
        source_domain_pred = self.cdan_loss(source_features, source_preds)
        target_domain_pred = self.cdan_loss(target_features, target_preds)

        return source_domain_pred, target_domain_pred

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
        # private_emb, shared_emb, concat_emb = self.dsn(concat_emb)   # 240826

        # Lucas 240709 added feature selection layer, 效果没有提升
        # selected_concat_emb = self.feature_selector(concat_emb)

        # Domain classification task
        # concat_emb_grl = self.grl(concat_emb)
        # u_ncf_dom_label = self.grl_activation(self.domain_fc(concat_emb_grl))

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
        # pred_ncf_domain_loss = self.exp_loss_fn(u_ncf_dom_label, domain)
        # print(f"Lucas--predict ncf domains: {u_ncf_dom_label}; original ncf domains: {domain}; ncf pre domain loss: {pred_ncf_domain_loss}")
        # # Lucas 240702 added for contrastive learning
        # pred_neg_domains_loss = self.exp_loss_fn(predict_neg_domain, neg_domain)
        # print(f"Lucas--neg predict_domains: {predict_neg_domain}; original neg domains: {neg_domain}; prefix neg pre domain loss: {pred_neg_domains_loss}")
        # pred_neg_ncf_domain_loss = self.exp_loss_fn(neg_u_ncf_dom_label, neg_domain)
        # print(f"Lucas--predict neg ncf domains: {neg_u_ncf_dom_label}; original neg ncf domains: {neg_domain}; neg ncf pre domain loss: {pred_neg_ncf_domain_loss}")

        # lucas231122 change the following code to return llm_model directly
        # print(f"Lucas--lucas_module.py--PeftPromptLlama2_V4_2--forward--user device: {user.device}")
        # print(f"Lucas--rating loss: {rating_loss}; domain prediction loss: {pred_ncf_domain_loss}")
        # print(f"Lucas--rating loss device: {rating_loss.device}; domain loss dtype: {pred_ncf_domain_loss.device}; predict_ncf_domain loss: {pred_ncf_domain_loss.device}")

        total_loss = self.lambda1*rating_loss # + self.lambda2*pred_ncf_domain_loss
        # print(f"Lucas--final total_loss: {total_loss}")
        # loss_detail = [rating_loss]
        return total_loss, predict_ratings.view(-1)