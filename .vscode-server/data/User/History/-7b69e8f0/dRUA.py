import torch
import torch.nn as nn
from momentfm import MOMENTPipeline
from transformers import AutoModel, AutoImageProcessor


class MomentDino(nn.Module):
    def __init__(self, num_classes, dropout, input_size, seq_length): 
        super(MomentDino, self).__init__()
        
        embedding_moment = 2048
        embedding_dino = 2048   
        self.input_size = input_size
        self.seq_length = seq_length
        self.dropout = dropout
        self.attention_weights = []
        
        self.moment_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small", 
            model_kwargs={
                'task_name': 'classification',
                'n_channels': input_size, # number of input channels
                'seq_length': seq_length,
                'num_class': num_classes,
                'dropout': dropout,
            },
            )
        
        self.moment_model.init() 
        
        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dino-vitb16")
        self.dinov2 = AutoModel.from_pretrained("facebook/dino-vitb16", output_attentions=True)    
            
        self.ll = nn.Linear(384, 256)
        
        self.norm1 = nn.LayerNorm(embedding_moment + embedding_dino )
        self.dropout = nn.Dropout(dropout)
        
        self.fc_fusion = nn.Linear(embedding_moment + embedding_dino, 256)
        self.norm2 = nn.LayerNorm(256)
        self.final_fc = nn.Linear(256, num_classes)
    
    
    def forward(self, x1, x2):
        
        B, S, F = x1.size()
        frame_embeddings = []
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(1,0,2,3,4)
        
        output = self.moment_model(x1)
        moment_embeddings = output.embeddings
        moment_embeddings = moment_embeddings.view(B, S, -1)
        
        for frame in x2:
            frame = frame.to(x1.device).float()
            frame_input = self.dino_processor(images=frame, return_tensors="pt").pixel_values.to(frame.device)
            dino_output = self.dinov2(frame_input)
            frame_embeddings.append(dino_output.last_hidden_state.mean(dim=1))
            # self.attention_weights.append(attention_weight)
        
        dino_embedding = torch.stack(frame_embeddings, dim=0)
        combined_embedding = torch.cat((moment_embeddings, dino_embedding), dim=1)
        
        combined_embedding = self.dropout(combined_embedding)
        combined_embedding = self.norm1(combined_embedding)
        combined_embedding = nn.ReLU()(combined_embedding)
        
        out = self.fc_fusion(combined_embedding)
        
        out = self.dropout(out)
        out = self.norm2(out)
        out = nn.ReLU()(out)
        
        out = self.final_fc(out)
        
        return out