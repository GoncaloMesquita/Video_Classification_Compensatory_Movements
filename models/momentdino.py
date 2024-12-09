import torch
import torch.nn as nn
from momentfm import MOMENTPipeline
from transformers import AutoModel, AutoImageProcessor


class MomentDino(nn.Module):
    def __init__(self, num_classes, dropout, input_size, seq_length): 
        super(MomentDino, self).__init__()
        
        embedding_moment = 6336
        embedding = 384
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
        
        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dino-vits16")
        self.dinov2 = AutoModel.from_pretrained("facebook/dino-vits16", output_attentions=True)
            
        self.moment_projection = nn.Linear(embedding_moment, embedding)
        
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(2*embedding)
        self.dropout = nn.Dropout(dropout)
        
        self.fc_fusion = nn.Linear(2*embedding, 256)
        self.norm2 = nn.LayerNorm(256)
        self.final_fc = nn.Linear(256, num_classes)
    
    
    def forward(self, x1, x2):
        
        B, S, F = x1.size()
        frame_embeddings = []
        attention_maps = []
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(1,0,2,3,4)
        
        output = self.moment_model(x1)
        moment_embeddings = output.embeddings
        moment_embeddings = moment_embeddings.view(B, S, -1)
        moment_proj = self.moment_projection(moment_embeddings)
        
        for frame in x2:
            frame = frame[...,[2, 1, 0]]
            frame_input = self.dino_processor(images=frame, return_tensors="pt").pixel_values.to(frame.device)
            dino_output = self.dinov2(frame_input)
            frame_embeddings.append(dino_output.last_hidden_state.mean(dim=1))
            attention_maps.append(dino_output.attentions)
        
        self.attention_weights = attention_maps
        dino_embedding = torch.stack(frame_embeddings, dim=0)
        dino_embedding = dino_embedding.permute(1, 0, 2)
        
        combined_embedding = torch.cat((moment_proj, dino_embedding), dim=-1)
        
        combined_embedding = self.norm1(self.dropout(self.relu(combined_embedding)))

        out = self.fc_fusion(combined_embedding)
        out = self.norm2(self.dropout(self.relu(out)))
        
        out = out.mean(dim=1)

        out = self.final_fc(out)
        
        return out