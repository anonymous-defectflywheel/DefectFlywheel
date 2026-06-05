import torch 
import torch.nn as nn
import torch.nn.init as init

class SoftPrompt(nn.Module):
    def __init__(self):
        super(SoftPrompt, self).__init__()
        #init num of learnable tokens
        self.pos_prompt = nn.Parameter(torch.randn(1, 24, 768))
        self.neg_prompt = nn.Parameter(torch.randn(1, 24, 768))
        # self.soft_prompt.requires_grad = True
        self.pos_attention = nn.MultiheadAttention(embed_dim=768, num_heads=1)
        self.neg_attention = nn.MultiheadAttention(embed_dim=768, num_heads=1)

        self.layer1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1, 24),
        )
        
        self.pos_layer0 = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU()
        )

        self.pos_layer1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.pos_layer2 = nn.Sequential(
            nn.Linear(16, 24),
            nn.Dropout(0.1),
            nn.LeakyReLU()
        )

        self.neg_layer0 = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU()
        )
        self.neg_layer1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.neg_layer2 = nn.Sequential(
            nn.Linear(16, 24),
            nn.Dropout(0.1),
            nn.LeakyReLU()
        )

        self.attn_pos_layer0 = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU()
        )

        self.attn_pos_layer1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.attn_pos_layer2 = nn.Sequential(
            nn.Linear(16, 24),
        )

        self.attn_neg_layer0 = nn.Sequential(
            nn.Linear(1, 16),
            # nn.Dropout(0.1),
            nn.LeakyReLU()
        )
        self.attn_neg_layer1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.attn_neg_layer2 = nn.Sequential(
            nn.Linear(16, 24),
        )

        self.blip_pos_layer1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.blip_pos_layer2 = nn.Sequential(
            nn.Linear(16, 16),
        )

        self.blip_neg_layer1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.blip_neg_layer2 = nn.Sequential(
            nn.Linear(16, 16),
        )

        self.blip_pos_proj1 = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU()
        )
        self.blip_pos_proj2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.LeakyReLU()
        )

        self.blip_neg_proj1 = nn.Sequential(
            nn.Linear(16, 8),
        )
        self.blip_neg_proj2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.LeakyReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                init.constant_(m.bias.data,0)
    
    def forward(self,query, vis_token):
        query = query.to(torch.float32)

        vis_prompt = self.layer1(vis_token)
        vis_prompt = self.layer2(vis_prompt.T)

        pos_vq_prompt = self.pos_layer0(vis_token.T).T
        pos_vis_prompt = self.pos_layer1(pos_vq_prompt)
        pos_vis_prompt = self.pos_layer2(pos_vis_prompt.T)


        neg_vq_prompt = self.neg_layer0(vis_token.T).T
        neg_vis_prompt = self.neg_layer1(neg_vq_prompt)
        neg_vis_prompt = self.neg_layer2(neg_vis_prompt.T)

        pos_query = self.blip_pos_layer1(query[0])
        pos_query = self.blip_pos_layer2(pos_query.T)

        neg_query = self.blip_neg_layer1(query[0])
        neg_query = self.blip_neg_layer2(neg_query.T)


        pos_attn_prompt1, _ = self.pos_attention(vis_token, pos_query.T, pos_query.T)

        pos_attn_prompt = self.attn_pos_layer0(pos_attn_prompt1.T).T
        pos_attn_prompt = self.attn_pos_layer1(pos_attn_prompt)
        pos_attn_prompt = self.attn_pos_layer2(pos_attn_prompt.T)



        neg_attn_prompt1, _ = self.neg_attention(vis_token, neg_query.T, neg_query.T)

        neg_attn_prompt = self.attn_neg_layer0(neg_attn_prompt1.T).T
        neg_attn_prompt = self.attn_neg_layer1(neg_attn_prompt)
        neg_attn_prompt = self.attn_neg_layer2(neg_attn_prompt.T)

        pos_prompt = (self.pos_prompt + vis_prompt.T + pos_attn_prompt.T + pos_vis_prompt.T)
        neg_prompt = (self.neg_prompt + vis_prompt.T + neg_attn_prompt.T + neg_vis_prompt.T)


        cat_pos_query = self.blip_pos_proj1(query.permute(0,2,1))
        cat_pos_query = self.blip_pos_proj2(cat_pos_query)
        cat_neg_query = self.blip_neg_proj1(query.permute(0,2,1))
        cat_neg_query = self.blip_neg_proj2(cat_neg_query)

        prompt_query_1 = torch.hstack((pos_prompt, cat_pos_query.permute(0,2,1)))
        prompt_query_2 = torch.hstack((neg_prompt, cat_neg_query.permute(0,2,1)))


        return prompt_query_1, prompt_query_2