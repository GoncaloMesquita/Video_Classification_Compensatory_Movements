import torch
import torch.nn as nn
from momentfm import MOMENTPipeline
from transformers import AutoModel, AutoImageProcessor


class MomentDino(nn.Module):
    def __init__(self, num_classes, dropout): 
        super(MomentDino, self).__init__()
        
        embedding_moment = 2048
        embedding_dino = 2048   
        self.dropout = dropout
        self.moment_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small", 
            model_kwargs={
                'task_name': 'classification',
                'n_channels': 99, # number of input channels
                'seq_length': 512,
                'num_class': num_classes,
                'dropout': dropout,
            },
            # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
            )
        
        self.moment_model.init() 
        
        
        model  = AutoModel.from_pretrained("facebook/dino-v2-vitb14", output_attentions=True)
        self.dinov2 = AutoImageProcessor.from_pretrained(model)
        
        self.fc_fusion = nn.Linear(embedding_moment + embedding_dino, 256)
        
        self.final_fc = nn.Linear(256, num_classes)
    
    
    def forward(self, x1, x2):
        x = x.permute(0, 2, 1)
        
        output = self.moment_model(x1)
        
        frame_embeddings = []
        attention_weights = []
        
        for frame in x2:
            frame_input = self.dino_processor(images=frame, return_tensors="pt").pixel_values.to(frame.device)
            dino_output, attention_weight = self.dino_model(frame_input)
            frame_embeddings.append(dino_output.last_hidden_state.mean(dim=1))
            attention_weights.append(attention_weight)
        
        dino_embedding = torch.stack(frame_embeddings, dim=0).mean(dim=0)  
        combined_embedding = torch.cat((output.embeddings, dino_embedding), dim=1)
        
        # Apply dropout, normalization, and non-linear function
        combined_embedding = nn.Dropout(p=self.dropout)(combined_embedding)
        combined_embedding = nn.LayerNorm(combined_embedding.size()[1:])(combined_embedding)
        combined_embedding = nn.ReLU()(combined_embedding)
        
        out = self.fc_fusion(combined_embedding)
        
        # Apply dropout, normalization, and non-linear function
        out = nn.Dropout(p=self.dropout)(out)
        out = nn.LayerNorm(out.size()[1:])(out)
        out = nn.ReLU()(out)
        
        out = self.final_fc(out)
        
        return output.logits