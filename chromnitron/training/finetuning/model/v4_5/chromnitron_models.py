import torch
import torch.nn as nn
import chromnitron.training.finetuning.model.v4_5.chromnitron_blocks as blocks
from chromnitron.training.pretraining.utils import read_config
import loralib as lora

def get_model(args):
    '''
    model = Chromnitron(args['num_features'],
                            hidden = args['hidden'],
                            num_attn_blocks = args['num_attn_blocks'],
                            num_of_scale = args['num_of_scale'],
                            num_targets = args['num_targets'],
                            no_confidence_predcition = args['no_confidence_prediction'],
                            prot_dim = args['prot_dim'],
                            sample_per_chunk = args['sample_per_chunk'])
    '''
    model = Chromnitron(**args)
    return model

def load_checkpoint(filepath, model):
    # Load the checkpoint
    checkpoint = torch.load(filepath)
    
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    # Adjust the keys
    '''
    if next(iter(checkpoint.keys())).startswith('module.'):
        # Create a new OrderedDict that does not have `module.` prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
    else:
        new_state_dict = checkpoint
    '''
    # Load the adjusted state dict into the model
    try:
        model.load_state_dict(checkpoint)
    except:
        print('Failed to load state dict, trying to load with different keys, CAUTION: this might cause the model not to work properly!')
        model.load_state_dict(checkpoint, strict = False)
    return model

class ChromnitronBase(nn.Module):
    def __init__(self, num_genomic_features, hidden = 512, num_attn_blocks = 16, num_of_scale = 2, num_targets = 1, no_confidence_prediction = False, *args, **kwargs):
        super(ChromnitronBase, self).__init__()
        print('Initializing Chromnitron')
        encoder_filter_size = 5
        num_blocks = num_of_scale
        self.encoder = blocks.MultiModalEncoder(num_genomic_features, hidden, encoder_filter_size, num_blocks=num_blocks - 1).cuda()
        self.tf = blocks.AttentionModule(hidden, num_attn_blocks).cuda()
        self.decoder = blocks.Decoder(hidden, hidden, 3, num_blocks=num_blocks, output_channel = num_targets, no_confidence_prediction = no_confidence_prediction).cuda()

    def forward(self, x):
        '''
        Input feature:
        batch_size, length * res, feature_dim
        '''
        x = self.encoder(x)

        x = x.transpose(1, 2).contiguous()
        x = self.tf(x)
        x = x.transpose(1, 2).contiguous()

        x = self.decoder(x)
        return x

class Chromnitron(ChromnitronBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = kwargs['hidden']
        prot_dim = kwargs['prot_dim']
        self.sample_per_chunk = kwargs['sample_per_chunk']
        self.prot_encoder = blocks.ProteinEncoder(prot_dim, hidden = hidden, filter_size = 5, num_blocks = 3)
        self.hidden = hidden

        if 'pretrained' in kwargs and kwargs['pretrained']['load']:
            print(f'Loading pretrained model from {kwargs["pretrained"]["load_path"]}')
            load_checkpoint(kwargs['pretrained']['load_path'], self)

        if 'fp16' in kwargs and kwargs['fp16']['enabled']:
            print('Setting up FP16')
            self.half()
        
        if 'lora' in kwargs and kwargs['lora']['enabled']:
            print('Setting up LoRA')
            load_lora_pretrained(self, kwargs['pretrained']['load_path'], kwargs['lora']['r'])



    def forward(self, seq_feature, prot_feature):
        seq_embedding = self.encoder(seq_feature)
        batch_size = seq_embedding.size(0)
        chunk_size, num_targets, prot_h, prot_len = prot_feature.size()
        assert chunk_size * self.sample_per_chunk == batch_size
        prot_feature = prot_feature.view(chunk_size * num_targets, prot_h, prot_len)
        prot_embedding_chunk = self.prot_encoder(prot_feature)
        prot_embedding_chunk = prot_embedding_chunk.view(chunk_size, num_targets, self.hidden, -1)

        seq_emb_length = seq_embedding.size(-1)

        seq_emb_repeat = seq_embedding.unsqueeze(1).repeat(1, num_targets, 1, 1)
        split_emb = torch.zeros(batch_size, num_targets, self.hidden, 1, device = seq_embedding.device)
        # Repeat the protein embedding within each chunk to save memory
        prot_emb_repeat = prot_embedding_chunk.repeat_interleave(self.sample_per_chunk, 0)
        joint_embedding = torch.cat([seq_emb_repeat, split_emb, prot_emb_repeat], dim = -1)

        joint_batch = joint_embedding.view(batch_size * num_targets, -1, joint_embedding.size(-1))

        x = joint_batch.transpose(1, 2).contiguous()
        x = self.tf(x)
        x = x.transpose(1, 2).contiguous()
        seq_tf_emb = x[:, :, :seq_emb_length]

        out = self.decoder(seq_tf_emb)
        out = [x.view(batch_size, num_targets, -1) for x in out]
        return out


def load_lora_pretrained(model, finetune_model_path, lora_r):
    print(f'Loading model from {finetune_model_path} with LoRA')
    replace_layers_with_lora(model, lora_layer_factory, r = lora_r)
    lora.mark_only_lora_as_trainable(model)
    weights = torch.load(finetune_model_path)
    if 'model' in weights:
        weights = weights['model']
    load_state_dict_to_lora(model, weights)
    return model

def load_state_dict_to_lora(model, state_dict):
    model_state_dict = model.state_dict()
    model_key_conv_list = list(model_state_dict.keys())
    no_conv_to_conv_dict = {}
    for key in model_key_conv_list:
        if '.conv.' in key:
            no_conv_key = key.replace('.conv.', '.')
            no_conv_to_conv_dict[no_conv_key] = key
        else:
            no_conv_to_conv_dict[key] = key
    # Modify the state dict
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key
        new_state_dict[no_conv_to_conv_dict[new_key]] = state_dict[key]
    model.load_state_dict(new_state_dict, strict = False)

def replace_layers_with_lora(model, lora_layer_factory, r):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            # Recursively apply to child modules
            replace_layers_with_lora(module, lora_layer_factory, r)
        else:
            # Replace the layer with a LoRA layer
            setattr(model, name, lora_layer_factory(module, r))
    return model

def lora_layer_factory(original_layer, r):
    import loralib as lora
    # Check the type of the original layer and create a corresponding LoRA layer
    if isinstance(original_layer, nn.Linear):
        #print(f'LoRA Linear with input size {original_layer.in_features} and output size {original_layer.out_features}')
        return lora.Linear(original_layer.in_features, original_layer.out_features, r = r)
    elif isinstance(original_layer, nn.Embedding):
        #print(f'LoRA Embedding with input size {original_layer.num_embeddings} and output size {original_layer.embedding_dim}')
        return lora.Embedding(original_layer.num_embeddings, original_layer.embedding_dim, r = r)
    elif isinstance(original_layer, nn.Conv1d):
        #print(f'LoRA Conv1d with input size {original_layer.in_channels} and output size {original_layer.out_channels}')
        return lora.Conv1d(original_layer.in_channels, 
                           original_layer.out_channels, 
                           original_layer.kernel_size[0], 
                           r = r, 
                           stride=original_layer.stride[0], padding=original_layer.padding[0], dilation=original_layer.dilation[0])
    elif isinstance(original_layer, nn.ConvTranspose1d):
        #print(f'LoRA ConvTranspose1d with input size {original_layer.in_channels} and output size {original_layer.out_channels}')
        return lora.ConvTranspose1d(original_layer.in_channels, 
                                    original_layer.out_channels, 
                                    original_layer.kernel_size[0], 
                                    r = r, 
                                    stride=original_layer.stride[0], padding=original_layer.padding[0], dilation=original_layer.dilation[0])
    else:
        return original_layer


def test():
    total_transcript_local = 2_000_000 * 16000
    #total_transcript_local = 3_000_000_000
    length = 8192
    bs = 8
    sample_per_chunk = 4
    num_samples = total_transcript_local // length // bs

    seq = torch.randn(bs, length, 5, device='cuda')
    feat = torch.randn(bs, length, 1, device='cuda')
    labels = torch.randn(bs, length, 1, device='cuda')
    seq = seq.transpose(1, 2)
    feat = feat.transpose(1, 2)
    labels = labels.transpose(1, 2)
    inputs = (seq, feat)

    prot_input = torch.randn(bs // sample_per_chunk, 5, 4096, 2560, device='cuda')
    prot_input = prot_input.transpose(-1, -2)

    #model = Chromnitron(1, hidden = 768, num_attn_blocks = 16, num_of_scale = 4, num_targets = 1, prot_dim = 2560, sample_per_chunk = sample_per_chunk)
    args = read_config('model_config.yaml')

    args = args['model']['args']

    model = get_model(args)
    model.eval()
    model.cuda()

    #model.decoder.conv_end = nn.Conv1d(1280, 1, kernel_size=1).cuda()

    # Turn off gradient for encoder and first half of tf
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.tf.pos_encoder.parameters():
        param.requires_grad = False
    num_attention_blocks = len(model.tf.module.layers)
    for param in model.tf.module.layers[:int(num_attention_blocks//3*2)].parameters():
        param.requires_grad = False

    '''
    with torch.no_grad():
        from tqdm import tqdm
        for i in tqdm(range(num_samples)):
            model(inputs)
    '''

    # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: ', num_params)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    outputs, error = model(inputs, prot_input)
    print(outputs.shape)
    loss = nn.MSELoss()(outputs, labels)
    print(loss)
    loss.backward()
    optimizer.step()
    breakpoint()

if __name__ == '__main__':
    test()
