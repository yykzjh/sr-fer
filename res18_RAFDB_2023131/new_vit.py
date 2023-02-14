import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.dropout = nn.Dropout(0.3)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)


        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        # print("%%%%%%%%%%%%% x {} %%%%%%%%%%%%%%%%%%%%%%".format(x.size()))
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Transformer2(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        # print("%%%%%%%%%%%%% x {} %%%%%%%%%%%%%%%%%%%%%%".format(x.size()))
        for attn, ff in self.layers:
            x = attn(x) + x
            # x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches , dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        print("")

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)

        x = self.transformer(x)
        # print(x.size())

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # print(x.size())

        # x = self.to_latent(x)
        # return self.mlp_head(x)
        return x


class ch_ViT(nn.Module):
    def __init__(self, *, ch_embed_dim, num_heads, ch_num_patchs,ffn_dim):
        super().__init__()
        

        self.pos_embedding = nn.Parameter(torch.randn(1, ch_num_patchs , ch_embed_dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = Transformer2(ch_embed_dim, depth=2, heads=num_heads , dim_head=64, mlp_dim=ffn_dim, dropout=0)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(ch_embed_dim),
            nn.Linear(ch_embed_dim, ch_embed_dim)
        )
        # print("")
        self.bn = nn.BatchNorm2d(ch_num_patchs)

    def forward(self, img):
        b, c,h, w = img.shape
        x = img.reshape(b, c,-1)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :(n)]
        x = self.dropout(x)
        x = self.transformer(x)
        # z_max,index = torch.max(x,dim=2)
        # z_out = torch.sigmoid(z_max)
        # x = self.mlp_head(x) + x
        # print(x.size())
        bs, c, hw = x.size()
        # # x = x.mean(dim=-1)
        z_h = int(hw**0.5)
        x = x.view(bs, c, z_h, z_h)
        x = self.bn(x)
        # x = torch.sigmoid(x)

        # print(x.size())
        # z_out.unsqueeze(-1).unsqueeze(-1)

        return x


# image_size=(img_size,img_size),in_dim=ch_num_patchs,patch_size=patch_size

class sp_ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, out_dim):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # print(num_patches)
        patch_dim = dim * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, out_dim),
        )
        self.embedding = nn.Conv2d(dim, out_dim, kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches , out_dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = Transformer(out_dim, depth=1, heads=2, dim_head=64, mlp_dim=128, dropout=0.1)

        self.to_latent = nn.Identity()


    def forward(self, img):
        # x = self.to_patch_embedding(img)
        x = self.embedding(img)   # (n, c, gh, gw)
        x = x.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        # print(x.size())
        # print(emb.size())
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # print(x.size())
        x += self.pos_embedding[:, :(n)]
        # print(self.pos_embedding.size())
        # print(x.size())
        x = self.dropout(x)
        # print("%%%%%%%%%%%%% x {} %%%%%%%%%%%%%%%%%%%%%%".format(x.size())) 

        x = self.transformer(x)
        # print("%%%%%%%%%%%%% x {} %%%%%%%%%%%%%%%%%%%%%%".format(x.size())) 
        x = x.permute(0, 2, 1) 
        # print("%%%%%%%%%%%%% x {} %%%%%%%%%%%%%%%%%%%%%%".format(x.size())) 
        b, c, hw = x.shape
        x = x.reshape(b, c, int(hw**(0.5)),int(hw**(0.5)))
        # print("%%%%%%%%%%%%% x {} %%%%%%%%%%%%%%%%%%%%%%".format(x.size()))
        # x = self.arrangement(x)
        # print(x.size())
        # x_out = torch.unsqueeze(x[:, 0], 1) + torch.unsqueeze(x[:, 1], 1)
        # x2 = torch.unsqueeze(x[:, 1], 1)

        # x = x1+x2
        # print('----------')
        # print(x_out.size())


        return x


class sp_ViT2(nn.Module):
    def __init__(self, *, image_size, patch_size, dim):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # print(num_patches)
        patch_dim = dim * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        # nn.Conv2d(dim, patch_size[0]**2, 
        self.embedding = nn.Conv2d(dim, dim, kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches , dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = Transformer(dim, depth=2, heads=3, dim_head=64, mlp_dim=128, dropout=0)

        self.to_latent = nn.Identity()
        self.arrangement = nn.PixelShuffle(patch_size[0])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )


    def forward(self, img):
        x = self.to_patch_embedding(img)
        # print("%%%%%%%%%%%%% x {} %%%%%%%%%%%%%%%%%%%%%%".format(x.size())) 
        # x = self.embedding(img)   # (n, c, gh, gw)
        # print("%%%%%%%%%%%%% x {} %%%%%%%%%%%%%%%%%%%%%%".format(x.size())) 
        # x = x.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        # b, h, w, c = x.shape
        # x = x.reshape(b, h * w, c)
        # print(x.size())
        # print(emb.size())
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # print(x.size())
        x += self.pos_embedding[:, :(n)]
        # print(self.pos_embedding.size())
        # print(x.size())
        x = self.dropout(x)
        # print("%%%%%%%%%%%%% x {} %%%%%%%%%%%%%%%%%%%%%%".format(x.size())) 

        x = self.transformer(x)
        # x = self.mlp_head(x) + x 
        # print("%%%%%%%%%%%%% x {} %%%%%%%%%%%%%%%%%%%%%%".format(x.size())) 
        x = x.permute(0, 2, 1) 
        # print("%%%%%%%%%%%%% x {} %%%%%%%%%%%%%%%%%%%%%%".format(x.size())) 
        b, c, hw = x.shape
        # x = torch.softmax(x,dim=-1)
        x = x.reshape(b, c, int(hw**(0.5)),int(hw**(0.5)))
        # print("%%%%%%%%%%%%% x {} %%%%%%%%%%%%%%%%%%%%%%".format(x.size()))
        # x = self.arrangement(x)
        # print("%%%%%%%%%%%%% x {} %%%%%%%%%%%%%%%%%%%%%%".format(x.size()))

        # print(x.size())
        # x_out = torch.unsqueeze(x[:, 0], 1) + torch.unsqueeze(x[:, 1], 1)
        # x2 = torch.unsqueeze(x[:, 1], 1)

        # x = x1+x2
        # print('----------')
        # print(x.size())


        return x

class Classfication(nn.Module):
    def __init__(self, *, image_size, patch_size, depth, heads, mlp_dim, channels, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # print(num_patches)
        patch_dim = channels * patch_height * patch_width
        self.channels = channels
        self.patch_size = patch_size
        self.image_size = image_size
        # print(patch_dim)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            # nn.Linear(patch_dim, image_size*image_size),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches , patch_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer2(patch_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.bn = nn.BatchNorm2d(channels)


    def forward(self, img):
        # print(img.size())
        B,C,H,W = img.shape[0],img.shape[1],img.shape[2],img.shape[3]
        x = self.to_patch_embedding(img)
        # print(x.size())
        # print(x.size())


        b, n, _ = x.shape
       
        x += self.pos_embedding[:, :(n+1)]
        # print(x.size())
        x = self.dropout(x)

        x = self.transformer(x)
        # print(x.size())
        
        # x = x.mean(dim=1,keepdim=True)
        # print(x.size())
        x = x.reshape(x.size(0),-1,self.image_size,self.image_size)
        # x = torch.sigmoid(x)
        # print(x.size())

        x = x.reshape(x.size(0),n,self.patch_size,self.patch_size,C)
        x = x.permute(0,4,1,2,3)
        # x = x.reshape(x.size(0),C,n,self.patch_size,self.patch_size)
        z_h = int(n**0.5)
        x = x.reshape(x.size(0),C,z_h,z_h,self.patch_size,self.patch_size)
        x = x.permute(0,1,2,4,3,5)
        x = x.reshape(x.size(0),C,x.size(2)*x.size(3),-1)

        x = self.bn(x)

        # z_max,index = torch.max(x,dim=1,keepdim=True)
        # z_max = torch.sigmoid(z_max)
        # print(z_max.size())



   
        return x



class TBAM(nn.Module):
    def __init__(self, ch_embed_dim, num_heads, ch_num_patchs,ffn_dim,patch_size):
        super().__init__()
        self.ch_encoder = ch_ViT(ch_embed_dim=ch_embed_dim, num_heads=num_heads, ch_num_patchs=ch_num_patchs,ffn_dim=ffn_dim)
        img_size = int(ch_embed_dim**0.5)
        patch_size_list = (patch_size,patch_size)
        self.sp_encoder = Classfication(image_size = img_size,patch_size = patch_size,depth = 3,heads = 3,
                                        mlp_dim = 128,
                                        channels = ch_num_patchs,
                                        )
        self.bn = nn.BatchNorm2d(ch_num_patchs)
    def forward(self, inputs):
        # print('inputs.size()',inputs.size())

        ch_encoder_out = self.ch_encoder(inputs)+inputs
        sp_encoder_out = self.sp_encoder(ch_encoder_out)+ch_encoder_out
        out = sp_encoder_out
        out = self.bn(out)
       
        
        return out+inputs


class TBAM2(nn.Module):
    def __init__(self, ch_embed_dim, num_heads, ch_num_patchs,ffn_dim,patch_size):
        super().__init__()
        patch_size_list = (patch_size,patch_size)
        # print(ch_num_patchs)
        self.sp_encoder = sp_ViT2(image_size=(int(ch_embed_dim**0.5),int(ch_embed_dim**0.5)),dim=ch_num_patchs,patch_size=patch_size_list)
        #VisionTransformer(image_size=(224, 224),patch_size=(16, 16),in_dim=256,num_heads=3,num_layers=3)
        # self.downsample = F.interpolate(x, size=[128, 128], mode="bilinear")
        # img_size = int(ch_embed_dim**0.5)//patch_size
        self.ch_encoder = ch_ViT(ch_embed_dim=ch_embed_dim, num_heads=num_heads, ch_num_patchs=ch_num_patchs,ffn_dim=ffn_dim)
        # self.conv = nn.Conv2d(in_channels=ch_num_patchs, out_channels=out_dim, kernel_size=1, stride=1)
    def forward(self, inputs):
        # print(inputs.size())

        # ch_encoder_out = self.ch_encoder(inputs)
        # print("*******************************************")
        # print(ch_encoder_out)
        # print(inputs.size(),ch_encoder_out.size())
        sp_encoder_out = self.sp_encoder(inputs)    # yuanlai shi add
        # print(inputs.size(),sp_encoder_out.size())
        ch_encoder_out = self.ch_encoder(sp_encoder_out)
        
        return ch_encoder_out





class TBAM3(nn.Module):
    def __init__(self, ch_embed_dim, num_heads, ch_num_patchs,ffn_dim,patch_size,out_dim):
        super().__init__()
        # ch_ViT(ch_embed_dim=ch_embed_dim, num_heads=num_heads, ch_num_patchs=ch_num_patchs,ffn_dim=ffn_dim)
        
        # img_size = int(ch_embed_dim**0.5)
        # self.sp_encoder = Sp_VisionTransformer(image_size=(img_size,img_size),in_dim=ch_num_patchs,patch_size=patch_size)
        # (image_size=(7,7), patch_size=(7,7), dim=512)
        patch_size_list = (patch_size,patch_size)
        # print(ch_num_patchs)
        self.sp_encoder = sp_ViT(image_size=(int(ch_embed_dim**0.5),int(ch_embed_dim**0.5)),dim=ch_num_patchs,patch_size=patch_size_list,out_dim = out_dim)
        #VisionTransformer(image_size=(224, 224),patch_size=(16, 16),in_dim=256,num_heads=3,num_layers=3)
        # self.downsample = F.interpolate(x, size=[128, 128], mode="bilinear")
        img_size = int(ch_embed_dim**0.5)//patch_size
        self.ch_encoder = ch_ViT(ch_embed_dim=img_size*img_size, num_heads=num_heads, ch_num_patchs=out_dim,ffn_dim=ffn_dim)
        # self.conv = nn.Conv2d(in_channels=ch_num_patchs, out_channels=out_dim, kernel_size=1, stride=1)
    def forward(self, inputs):
        # print(inputs.size())

        # ch_encoder_out = self.ch_encoder(inputs)
        # print("*******************************************")
        # print(ch_encoder_out)
        # print(inputs.size(),ch_encoder_out.size())
        sp_encoder_out = self.sp_encoder(inputs)     # yuanlai shi add
        ch_encoder_out = self.ch_encoder(sp_encoder_out)
        
        return ch_encoder_out

class Cls(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # print(num_patches)
        patch_dim = channels * patch_height * patch_width
        # print(patch_dim)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches , dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer2(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.bn1 = nn.BatchNorm2d(512)
        # print("")

    def forward(self, img):
        # print(img.size())
        x = self.to_patch_embedding(img)
        # print(x.size())
        # print(x.size())


        b, n, _ = x.shape
       

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # print(x.size())
        x += self.pos_embedding[:, :(n+1)]
        # print(self.pos_embedding.size())
        x = self.dropout(x)

        x = self.transformer(x)
        # print(x.size())

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # print(x.size())

        x = self.to_latent(x)
        return self.mlp_head(x)
        # return x




# v = Cls(
#     image_size = 224,
#     patch_size = 16,
#     num_classes = 7,
#     dim = 512,
#     depth = 3,
#     heads = 3,
#     mlp_dim = 512,
#     channels = 3,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )

# img = torch.randn(2, 3, 224, 224)
# print('----------------')
# preds = v(img) # (1, 1000)
# print(preds.size())


# img = torch.randn(2, 3, 224, 224)
# # print('----------------')
# preds = v(img) # (1, 1000)
# print(preds.size())

# v = ViT(
#     image_size = 256,
#     patch_size = 32,
#     num_classes = 7,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )

# ch_encoder = ch_ViT(ch_embed_dim=64*64, num_heads=2, ch_num_patchs=64,ffn_dim=128)
# sp_encoder = sp_ViT2(image_size=(224,224), patch_size=(4,4), dim=3,out_dim=64)
# tbam = TBAM(ch_embed_dim=224*224, num_heads=3, ch_num_patchs=3,ffn_dim=512,patch_size=16)
# img = torch.randn(2, 3, 224, 224)
# preds = tbam(img) 
# print(preds.size())



# img = torch.randn(2, 128, 28, 28)

# # preds = ch_encoder(img) # (1, 1000)

# # preds = sp_encoder(img) # (1, 1000)
# preds = tbam(img) # (1, 1000)

# # preds = v(img) # (1, 1000)

# print(preds.size())
# x = torch.randn(2, 3, 2, 2)
# x1 = torch.unsqueeze(x[:, 0], 1)
# x2 = torch.unsqueeze(x[:, 1], 1)
# print(x)
# print("--------------")
# print(x1)
# print("--------------")
# print(x2)
# print("--------------")
# print(x1+x2)
