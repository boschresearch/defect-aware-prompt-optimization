import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnablePrompt(nn.Module):
    def __init__(self, clip_model, prompt_state, normal_token_count, abnormal_token_count, prompt_count, token_depth, learnable_token_length, device, tokenizer):
        super(LearnablePrompt, self).__init__()

        self.emb_dim = clip_model.ln_final.weight.shape[0]
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_state = prompt_state
        self.normal_token_count = normal_token_count
        self.abnormal_token_count = abnormal_token_count
        self.prompt_count = prompt_count
        self.learnable_layer_token_count = learnable_token_length
        self.learnable_layer_token_depth = token_depth


        self.good_state = ["flawless {}"]
        self.class_names = ["object"]

        with torch.no_grad():
            text_embeddings = clip_model.token_embedding.weight
            mean = text_embeddings.mean(dim=0)
            std = text_embeddings.std(dim=0)

        self.learnable_normal_tokens = nn.Parameter(
            torch.randn(self.prompt_count, self.normal_token_count, self.emb_dim, device = device) * std + mean
        )

        self.learnable_abnormal_tokens = nn.Parameter(
            torch.randn(self.prompt_count, self.abnormal_token_count, self.emb_dim, device=device) * std + mean
        )

        self.learnable_layer_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(self.learnable_layer_token_count, self.emb_dim, device=device) * std + mean)
            for _ in range(self.learnable_layer_token_depth)
        ])


        # normal_template = [" ".join(['X'] * self.normal_token_count) + f" {state} industrial object"
        #                    for state in self.good_state
        #                    for _ in range(self.prompt_count)]
        
        for cls in self.class_names:
            normal_templates = []
            for state in self.good_state:
                for _ in range(self.prompt_count):
                    prompt = " ".join(["X"] * self.normal_token_count) + " " + state.format(cls)
                    normal_templates.append(prompt)


        for cls in self.class_names:
            abnormal_templates = []
            for state in self.prompt_state:
                for _ in range(self.prompt_count):
                    text_prompt = " ".join(["X"] * self.abnormal_token_count) + " " + state + " " + f"anomaly {cls}"
                    abnormal_templates.append(text_prompt)

        self.meta_net = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_dim // 16, self.emb_dim)
        )

        with torch.no_grad():
            normal_embeddings = [self._get_token_embedding(tempalte, clip_model, self.tokenizer)
                                                                       for tempalte in normal_templates]
            embedding_tokens = [self._get_token_embedding(template, clip_model, self.tokenizer)
                                                                                 for template in abnormal_templates]
            
        self.normal_object_token_embedding, self.normal_template_token = zip(*normal_embeddings)
        self.normal_object_token_embedding = torch.stack(list(self.normal_object_token_embedding))
        
        #self.normal_object_token_embedding = self.normal_object_token_embedding.reshape(1, len(self.good_state), self.prompt_count, 77, -1)

        self.normal_template_token = torch.stack(list(self.normal_template_token))
        self.normal_template_token = self.normal_template_token.squeeze()

        self.damaged_object_token_embedding, self.abnormal_template_token = zip(*embedding_tokens)
        self.damaged_object_token_embedding = torch.stack(list(self.damaged_object_token_embedding))
        self.damaged_object_token_embedding = self.damaged_object_token_embedding.reshape(len(prompt_state), self.prompt_count, 77, -1)

        self.abnormal_template_token = torch.stack(list(self.abnormal_template_token))
        self.abnormal_template_token = self.abnormal_template_token.squeeze()

        self.prefix_token = self.normal_object_token_embedding[:, :1, :]
        self.abnormal_prefix_token = self.damaged_object_token_embedding[:, :, :1, :]

        self.normal_suffix_token = self.normal_object_token_embedding[:, 1 + self.normal_token_count:, :]
        self.abnormal_suffix_token = self.damaged_object_token_embedding[:, :, 1 + self.abnormal_token_count:, :]

     
    def _get_token_embedding(self, word: str, model, tokenizer):
        token = tokenizer([word]).to(self.device)
        embedding = model.token_embedding(token).squeeze(0)
        return embedding, token
    
    def forward(self, img_emb):

        #inst_ctx = self.meta_net(img_emb)
        #inst_ctx = inst_ctx.unsqueeze(0)

        learnable_abnormal_tokens = (self.learnable_abnormal_tokens)
        learnable_abnormal_tokens = learnable_abnormal_tokens.expand(len(self.prompt_state), -1, -1, -1)
        learnable_normal_tokens = (self.learnable_normal_tokens)

        #normal_prompt_list = []
        normal_prompt = torch.cat([
            self.prefix_token,
            learnable_normal_tokens,
            self.normal_suffix_token
        ], dim=1).unsqueeze(0)

        #normal_prompt_list.append(normal_prompt)
        #normal_prompt = torch.stack(normal_prompt_list).unsqueeze(0)
        # abnormal_tokens = self.abnormal_template_token.squeeze(dim=1)
        
        #abnormal_prompt = []
    
        abnormal_prompt = torch.cat([
                self.abnormal_prefix_token,
                learnable_abnormal_tokens,
                self.abnormal_suffix_token
            ], dim=2)

        #abnormal_prompt.append(tokens)
        #abnormal_prompt = torch.stack(abnormal_prompt)

        template_tokens = torch.cat([self.normal_template_token, self.abnormal_template_token], dim=0)
        prompt = torch.cat([normal_prompt, abnormal_prompt], dim=0)

        return prompt, self.learnable_layer_tokens, template_tokens