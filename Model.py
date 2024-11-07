import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample
        self.resnet_conv_first = nn.Sequential(nn.GroupNorm(8, in_channels),
                                               nn.SiLU(),
                                               nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                                         stride=1, padding=1)
                                              )
        self.t_emb_layer = nn.Sequential(nn.SiLU(),
                                         nn.Linear(time_emb_dim, out_channels))
        self.resnet_conv_second = nn.Sequential(nn.GroupNorm(8, out_channels),
                                               nn.SiLU(),
                                               nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                                         stride=1, padding=1)
                                               )
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.res_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, 
                                         stride=2, padding=1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb):
        '''
        resnet conv block followed by concat with t embedding layer and then a second conv block.
        Op of second conv sent to self-attn and finally downsampling
        '''
        # Resnet Block
        out = self.resnet_conv_first(x)
        out = out + self.t_emb_layer(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.res_input_conv(x)
        
        # Attention Block
        batch_size, channels, h, w = out.shape
        attn_ip = out.reshape(batch_size, channels, h*w)
        attn_op = self.attention_norm(attn_ip)
        attn_op = attn_op.transpose(1, 2)
        attn_op, _ = self.attention(attn_op, attn_op, attn_op)
        attn_op = attn_op.transpose(1, 2).reshape(batch_size, channels, h, w)

        out = out + attn_op

        out = self.down_sample_conv(out)
        return out

class MiddleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_heads):
        super().__init__()
        self.resnet_conv_first = nn.ModuleList([nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )])
        self.t_emb_layer = nn.Sequential(nn.SiLU(),
                                         nn.Linear(time_emb_dim, out_channels))
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.res_input_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                            nn.Conv2d(out_channels, out_channels, kernel_size=1)])
        self.resnet_conv_second = nn.ModuleList([nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )])        
    
    def forward(self, x, t_emb):
        '''
        resnet conv block followed by self-attn and finally another resnet
        '''
        # Resnet Block
        out = self.resnet_conv_first[0](x)
        out = out + self.t_emb_layer(t_emb)[:, :, None, None]
        out = self.resnet_conv_first[1](out)
        out_res_1 = out + self.res_input_conv[0](x)
        
        # Attention Block
        out = out_res_1
        batch_size, channels, h, w = out.shape
        attn_ip = out.reshape(batch_size, channels, h*w)
        attn_op = self.attention_norm(attn_ip)
        attn_op = attn_op.transpose(1, 2)
        attn_op, _ = self.attention(attn_op, attn_op, attn_op)
        attn_op = attn_op.transpose(1, 2).reshape(batch_size, channels, h, w)

        # Resnet Block
        out = self.resnet_conv_second[0](attn_op)
        out = out + self.t_emb_layer(t_emb)[:, :, None, None]
        out = self.resnet_conv_second[1](out)

        out = out + self.res_input_conv[1](out_res_1)

        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up_sample, num_heads):
        super().__init__()
        self.up_sample = up_sample
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, 
                                         stride=2, padding=1) if self.up_sample else nn.Identity()
        self.resnet_conv_first = nn.Sequential(nn.GroupNorm(8, in_channels),
                                               nn.SiLU(),
                                               nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                                         stride=1, padding=1)
                                              )
        self.resnet_conv_second = nn.Sequential(nn.GroupNorm(8, out_channels),
                                               nn.SiLU(),
                                               nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                                         stride=1, padding=1)
                                               )
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.res_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.t_emb_layer = nn.Sequential(nn.SiLU(),
                                         nn.Linear(time_emb_dim, out_channels))

    def forward(self, x, out_down, t_emb):
        '''
        concat the outputs from correspoding downblocks into the upblock layers
        resnet conv block followed by concat with t embedding layer and then a second conv block.
        Op of second conv sent to self-attn and finally downsampling
        '''
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], axis=1)

        # Resnet Block
        out = self.resnet_conv_first(x)
        out = out + self.t_emb_layer(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.res_input_conv(x)
        
        # Attention Block
        batch_size, channels, h, w = out.shape
        attn_ip = out.reshape(batch_size, channels, h*w)
        attn_op = self.attention_norm(attn_ip)
        attn_op = attn_op.transpose(1, 2)
        attn_op, _ = self.attention(attn_op, attn_op, attn_op)
        attn_op = attn_op.transpose(1, 2).reshape(batch_size, channels, h, w)

        out = out + attn_op

        return out

class UNet(nn.Module):
    def __init__(self, im_channels):
        super().__init__()
        self.down_channels = [32, 64, 128, 256]
        self.mid_channels = [256, 256, 128]
        self.t_emb_dim = 128
        self.down_sample = [True, True, False]

        self.t_proj = nn.Sequential(nn.Linear(self.t_emb_dim, self.t_emb_dim),
                                    nn.SiLU(),
                                    nn.Linear(self.t_emb_dim, self.t_emb_dim)
                        )
        # self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim, 
                                        down_sample = self.down_sample[i], num_heads = 4))

        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MiddleBlock(self.mid_channels[i],  self.mid_channels[i+1], self.t_emb_dim, num_heads=4))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(UpBlock(self.down_channels[i]*2, self.down_channels[i-1] if i != 0 else 16,
                                    self.t_emb_dim, up_sample = self.down_sample[i], num_heads=4))

        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)

    def get_time_embedding(self, timesteps, time_emb_dim):
        '''
        this function takes in a batch size of time steps and
        returns a time embedding dimension for each timestep
        Input: (B,)
        Output: (B,time_emb_dim)
        10000^2i/d is the set of frequencies
        '''
        factor = 10000 ** (2*(torch.arange(start=0, end=time_emb_dim // 2, device=timesteps.device) / 
                                        (time_emb_dim)))
        t_emb = timesteps[:, None].repeat(1, time_emb_dim // 2) / factor
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

        return t_emb

    def forward(self, x, t):
        out = self.conv_in(x)
        t_emb = self.get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)

        for mid in self.mids:
            out = mid(out, t_emb)
        
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
        
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out